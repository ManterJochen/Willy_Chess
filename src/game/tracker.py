"""
Move tracking: the position only changes through a legal move.

Instead of classifying 64 squares with 13 states each, the tracker picks one of the ~30 legal
moves, or "no move" for a piece that was only straightened. Each candidate position is scored
by a negative log-likelihood (NLL) against the observed pieces; the smallest wins, but only
below an absolute gate. Above it the tracker flags the move instead of guessing. The absolute
NLL is the signal, not the lead over the second best: wrong moves are often confidently wrong.

The NLL has a geometric part (where pieces of which colour stand, which ones are hidden) and a
class part (observed piece type against the expected one). Occupancy and colour determine every
legal move except the piece of a promotion, so the gate only looks at the geometric part; the
class part ranks the candidates. A detector that confuses a bishop with a pawn in some frame
then costs nothing, where a gate on the sum flagged 3 of 4 moves at 5 % confused classes.

Two things the image does not show are tracked here:
- The full FEN (side to move, castling rights, en passant, clocks) comes from the move history.
- Logical and physical piece differ after a promotion when no spare piece is available
  (promotion_mode "pawn_stays"): the pawn stays on the board and counts as a queen. The
  observation model compares against the physical piece, the rules use the logical one.
"""
from dataclasses import dataclass

import chess
import cv2 as cv
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.mapping.frame import BoardFrame
from src.mapping.geometry import BoardCalibration, BoardCamera, square_centres
from src.mapping.pieces import OFF_BOARD_MARGIN, piece_positions

PROMOTION_MODES = ("pawn_stays", "piece_swapped")
_INFEASIBLE = 1e9


@dataclass
class TrackResult:
    """
    What one observation explained.

    Attributes:
        move (chess.Move | None): The move found, None for "no move" or when flagged.
        nll (float): Geometric negative log-likelihood of the best explanation (what the gate sees).
        fen (str | None): Full FEN after the move, None when flagged.
        flagged (bool): No explanation below the gate; nothing was committed.
        promotion_ambiguous (bool): A promotion whose piece the image cannot tell (the pawn stayed):
            the queen was assumed, the UI should ask.
    """
    move: chess.Move | None
    nll: float
    fen: str | None
    flagged: bool = False
    promotion_ambiguous: bool = False


def physical_after(board: chess.Board, physical: dict[int, str], move: chess.Move,
                   promotion_mode: str = "pawn_stays") -> dict[int, str]:
    """
    The physical pieces after a move: square -> FEN symbol of the piece that really stands there.

    Args:
        board (chess.Board): Logical position before the move.
        physical (dict[int, str]): Physical pieces before the move.
        move (chess.Move): A legal move in board.
        promotion_mode (str): "pawn_stays" (no spare piece, the pawn counts as the new piece) or
            "piece_swapped" (the promoted piece physically replaces the pawn).
    """
    out = dict(physical)
    moving = out.pop(move.from_square)
    if board.is_en_passant(move):
        out.pop(move.to_square + (-8 if board.turn == chess.WHITE else 8), None)
    else:
        out.pop(move.to_square, None)
    if board.is_castling(move):
        rank = chess.square_rank(move.from_square)
        kingside = board.is_kingside_castling(move)
        rook_from = chess.square(7 if kingside else 0, rank)
        rook_to = chess.square(5 if kingside else 3, rank)
        out[rook_to] = out.pop(rook_from)
    if move.promotion and promotion_mode == "piece_swapped":
        moving = chess.Piece(move.promotion, board.turn).symbol()
    out[move.to_square] = moving
    return out


class MoveTracker:
    """
    Follow a game from observations, one move per observation.

    Args:
        calib (BoardCalibration): Pose of the view the frames come from, square size, piece set.
        sigma (float): Spread of an observed piece around its square centre, in squares.
        gate (float): Largest NLL that is committed.
        hidden_ok (float): Cost of an unseen piece that the other masks could hide.
        hidden_bad (float): Cost of an unseen piece that should have been visible.
        hidden_frac (float): Visible fraction of the piece model below which a piece counts as hidden.
        class_cost (float): Cost of a piece observed with another type than expected (same colour).
            float("inf") forbids it; a finite cost lets the rules overrule a detector mistake.
        promotion_mode (str): See physical_after.
    """

    def __init__(self, calib: BoardCalibration, sigma: float = 0.3, gate: float = 10.0, hidden_ok: float = 1.0,
                 hidden_bad: float = 50.0, hidden_frac: float = 0.15, class_cost: float = 4.0,
                 promotion_mode: str = "pawn_stays"):
        if promotion_mode not in PROMOTION_MODES:
            raise ValueError(f"promotion_mode must be one of {PROMOTION_MODES}, not {promotion_mode!r}")
        self.calib = calib
        self.sigma, self.gate = sigma, gate
        self.hidden_ok, self.hidden_bad, self.hidden_frac = hidden_ok, hidden_bad, hidden_frac
        self.class_cost = class_cost
        self.promotion_mode = promotion_mode
        self.centres = square_centres(calib.square)
        self.board = chess.Board()
        self.physical = _physical_from(self.board)
        self._history: list[dict[int, str]] = []

    def reset(self, fen: str, physical: dict[int, str] | None = None) -> None:
        """Set the logical position and, if they differ from it, the physical pieces."""
        self.board = chess.Board(fen)
        self.physical = dict(physical) if physical is not None else _physical_from(self.board)
        self._history = []

    def step(self, frame: BoardFrame, calib: BoardCalibration | None = None) -> TrackResult:
        """
        Explain one observation by a legal move (or none) and commit it if it passes the gate.

        Args:
            frame (BoardFrame): A view of the board with piece masks.
            calib (BoardCalibration | None): Board pose of this view (a wrist camera sees the board from
                a slightly different pose every time); None uses the tracker's own.

        Returns:
            TrackResult: The move, or flagged when no explanation is good enough.
        """
        ctx = self._context(frame, calib)
        scored = [(*self._nll(self.physical, ctx), None)]
        for move in self.board.legal_moves:
            scored.append((*self._nll(physical_after(self.board, self.physical, move, self.promotion_mode), ctx), move))
        best = min(geo + cls for geo, cls, _ in scored)
        if not np.isfinite(best):
            return TrackResult(None, best, None, flagged=True)
        tied = [(geo, m) for geo, cls, m in scored if geo + cls <= best + 1e-9]
        geo_best = tied[0][0]
        if geo_best > self.gate:
            return TrackResult(None, geo_best, None, flagged=True)

        tied = [m for _, m in tied]
        move = tied[0]
        ambiguous = False
        if len(tied) > 1:
            promotions = [m for m in tied if m is not None and m.promotion]
            same_squares = all(m is not None and (m.from_square, m.to_square) == (tied[0].from_square, tied[0].to_square)
                               for m in tied)
            if not (promotions and same_squares and len(promotions) == len(tied)):
                return TrackResult(None, best, None, flagged=True)
            move = next((m for m in promotions if m.promotion == chess.QUEEN), promotions[0])
            ambiguous = True
        if move is not None:
            self.commit(move)
        return TrackResult(move, geo_best, self.board.fen(), promotion_ambiguous=ambiguous)

    def verify(self, frame: BoardFrame, move: chess.Move, calib: BoardCalibration | None = None) -> tuple[bool, float]:
        """
        Check that a move the robot executed shows up as expected, without committing it.

        The target position is known, so only its NLL is needed. Above the gate the grasp
        failed or a piece was knocked over.

        Returns:
            tuple[bool, float]: Whether the expected position explains the frame, and its geometric NLL.
        """
        ctx = self._context(frame, calib)
        geo, _ = self._nll(physical_after(self.board, self.physical, move, self.promotion_mode), ctx)
        return geo <= self.gate, geo

    def commit(self, move: chess.Move) -> None:
        """Apply a move to the logical and the physical position."""
        self._history.append(self.physical)
        self.physical = physical_after(self.board, self.physical, move, self.promotion_mode)
        self.board.push(move)

    def undo(self) -> chess.Move:
        """Take back the last committed move, e.g. to replace an assumed promotion piece."""
        self.physical = self._history.pop()
        return self.board.pop()

    def _context(self, frame: BoardFrame, calib: BoardCalibration | None = None) -> dict:
        calib = calib or self.calib
        # captured pieces in the graveyard beside the board are no part of the position
        positions = {k: xy for k, xy in piece_positions(frame, calib).items()
                     if np.abs(xy / calib.square).max() <= 4 + OFF_BOARD_MARGIN}
        return {"cam": BoardCamera(frame.K, calib.R, calib.t),
                "obs": [(positions[k], frame.classes[k], k) for k in positions],
                "fg": frame.labels > 0, "hidden": {}}

    def locate(self, frame: BoardFrame, calib: BoardCalibration | None = None) -> dict[int, int]:
        """
        Which observed piece is which piece of the current position: square -> piece label.

        The same matching as the move scoring, against the position as it is; pieces that were not
        seen are missing. More reliable than the free assignment of one image, because it knows
        which pieces must be there.
        """
        located: dict[int, int] = {}
        self._nll(self.physical, self._context(frame, calib), located)
        return located

    def _nll(self, physical: dict[int, str], ctx: dict,
             located: dict[int, int] | None = None) -> tuple[float, float]:
        """
        NLL of a candidate position, split into its geometric and its class part.

        Observed pieces are matched to expected ones of the same colour (Hungarian on squared
        distance plus class cost); unseen expected pieces cost hidden_ok or hidden_bad; an observed
        piece without a counterpart makes the candidate impossible. If `located` is given, it receives
        square -> label of every matched piece.

        Returns:
            tuple[float, float]: (geometric NLL, class NLL); inf for an impossible candidate.
        """
        geo = cls = 0.0
        for white in (True, False):
            squares = [s for s, sym in physical.items() if sym.isupper() == white]
            obs = [o for o in ctx["obs"] if o[1].isupper() == white]
            if len(obs) > len(squares):
                return np.inf, np.inf
            used = set()
            if obs:
                dist = np.empty((len(obs), len(squares)))
                mismatch = np.empty((len(obs), len(squares)), bool)
                for i, (xy, sym, _) in enumerate(obs):
                    dist[i] = np.sum(((xy - self.centres[squares]) / self.calib.square) ** 2, axis=1) / (2 * self.sigma ** 2)
                    mismatch[i] = [physical[s] != sym for s in squares]
                cost = dist + np.where(mismatch, min(self.class_cost, _INFEASIBLE), 0.0)
                rows, cols = linear_sum_assignment(cost)
                if cost[rows, cols].sum() >= _INFEASIBLE:
                    return np.inf, np.inf
                geo += dist[rows, cols].sum()
                cls += cost[rows, cols].sum() - dist[rows, cols].sum()
                used = {squares[c] for c in cols}
                if located is not None:
                    located.update({squares[c]: obs[r][2] for r, c in zip(rows, cols)})
            for s in squares:
                if s not in used:
                    hidden = self._visible_fraction(ctx, s, physical[s]) < self.hidden_frac
                    geo += self.hidden_ok if hidden else self.hidden_bad
        return geo, cls

    def _visible_fraction(self, ctx: dict, square: int, symbol: str) -> float:
        """
        Visible fraction of a piece model (cone frustum) on a square, minimised over where in the
        square it may stand (centre and 8 points 0.18 squares off): pieces rarely stand centred.
        """
        key = (square, symbol.lower())
        cache = ctx["hidden"]
        if key not in cache:
            cam, fg = ctx["cam"], ctx["fg"]
            h, r = self.calib.piece_dims(symbol)
            ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)
            best = np.inf
            offsets = [(0.0, 0.0)] + [(0.18 * np.cos(a), 0.18 * np.sin(a))
                                      for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)]
            for dx, dy in offsets:
                c = self.centres[square] + np.array([dx, dy]) * self.calib.square
                ring = [np.column_stack((c[0] + rr * np.cos(ang), c[1] + rr * np.sin(ang), np.full(len(ang), z)))
                        for z, rr in ((0.0, r), (h, 0.55 * r))]
                hull = cv.convexHull(cam.project(np.vstack(ring)).astype(np.float32))
                silhouette = np.zeros(fg.shape, np.uint8)
                cv.fillPoly(silhouette, [np.round(hull).astype(np.int32)], 1)
                silhouette = silhouette.astype(bool)
                best = min(best, np.count_nonzero(silhouette & ~fg) / max(np.count_nonzero(silhouette), 1))
            cache[key] = best
        return cache[key]


def _physical_from(board: chess.Board) -> dict[int, str]:
    return {s: p.symbol() for s, p in board.piece_map().items()}
