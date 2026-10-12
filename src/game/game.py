"""
The game: calibrate, then take turns until the game ends.

Your turn: you move, press the space bar, Willy looks and recognises the move from the rules
(src/game/tracker.py). Willy's turn: Stockfish chooses, the robot carries the pieces, Willy looks
again and checks that the board shows what it should. Whatever cannot be confirmed is shown and
asked for, never guessed: a flagged move, a moved board, a grasp that did not finish.
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import chess
import numpy as np

from src.game.config import ChessConfig
from src.game.robot_moves import PIECE_NAMES, MoveExecution, RobotMover, plan_transfers
from src.game.tracker import MoveTracker
from src.mapping.chessboard_mapping_3d import BoardObservation, ChessBoardMapping3D
from src.mapping.geometry import square_centres
from src.pieces.record import Recorder

PROMOTION_KEYS = {"q": chess.QUEEN, "r": chess.ROOK, "b": chess.BISHOP, "n": chess.KNIGHT}
INTRUDER_PX = 300           # a hand or an arm over the board covers thousands of pixels, noise none
DRAW_ACCEPT_CP = 50         # Willy accepts a draw unless it stands better than half a pawn
OUTCOMES = {chess.Termination.CHECKMATE: "checkmate", chess.Termination.STALEMATE: "stalemate",
            chess.Termination.INSUFFICIENT_MATERIAL: "insufficient material",
            chess.Termination.SEVENTYFIVE_MOVES: "75-move rule",
            chess.Termination.FIVEFOLD_REPETITION: "fivefold repetition"}


@dataclass
class ViewState:
    """Everything the display shows."""
    board: chess.Board
    physical: dict[int, str]
    human_color: bool
    status: str = ""
    evaluation: str = ""
    observation: BoardObservation | None = None
    uncertain: set[int] = field(default_factory=set)


class View(Protocol):
    """The display and the human's input (src/ui/pygame_view.py, or a scripted one in tests)."""

    def show(self, state: ViewState) -> None: ...

    def wait(self, state: ViewState, accept: set[str]) -> str:
        """
        Block until the human gives one of the accepted answers: "done" (space bar), "recalibrate" (R),
        "quit" (Esc), "resign" (G), "draw" (D), a promotion piece "q", "r", "b", "n", a colour "white" (W)
        or "black" (B), or, if "move" is accepted, "move:<uci>" (two squares clicked).
        """
        ...


class GameAborted(Exception):
    """The human ended the game."""


class GameOver(Exception):
    """The game ended before the board says so: resigned or drawn by agreement."""

    def __init__(self, result: str, reason: str):
        super().__init__(reason)
        self.result, self.reason = result, reason


def describe_wrong_move(board: chess.Board, physical: dict[int, str], seen: dict[int, str],
                        human: bool) -> str | None:
    """
    Why the board shows no legal move, when the picture shows one clear wrong move: a piece of Willy's
    moved, or one of the human's pieces moved from one square to another against the rules.

    Args:
        board (chess.Board): The position before the move.
        physical (dict[int, str]): Square -> piece that stood there before the move.
        seen (dict[int, str]): Square -> piece the camera sees there now.
        human (bool): The human's colour.

    Returns:
        str | None: The message for the human; None if the picture shows no single clear move.
    """
    def changes(white: bool) -> tuple[list[int], list[int]]:
        left = [sq for sq, sym in physical.items() if sym.isupper() == white
                and not (sq in seen and seen[sq].isupper() == white)]
        arrived = [sq for sq, sym in seen.items() if sym.isupper() == white
                   and not (sq in physical and physical[sq].isupper() == white)]
        return left, arrived

    willy_left, willy_arrived = changes(not human)
    if willy_arrived:
        return (f"A piece of Willy's now stands on {chess.square_name(willy_arrived[0])}. "
                f"Please put it back, then press space.")
    left, arrived = changes(human)
    if len(left) != 1 or len(arrived) != 1:
        return None
    piece = board.piece_at(left[0])
    promotion = chess.QUEEN if piece.piece_type == chess.PAWN and chess.square_rank(arrived[0]) in (0, 7) else None
    move = chess.Move(left[0], arrived[0], promotion=promotion)
    if move in board.legal_moves:
        return None
    if not board.is_pseudo_legal(move):
        reason = "this piece cannot move like that"
    elif board.is_into_check(move):
        reason = "your king would be in check"
    else:
        reason = "the rules do not allow it here"
    name = PIECE_NAMES[piece.symbol().lower()].capitalize()
    return (f"{name} {chess.square_name(left[0])}-{chess.square_name(arrived[0])} is not allowed: {reason}. "
            f"Please take it back, then press space.")


class ChessGame:
    """
    Human against Willy on the real board.

    Args:
        mapping (ChessBoardMapping3D): Camera, detector and board geometry.
        engine: Something with best_move(board) and evaluation(board) (src/game/engine.Engine).
        mover (RobotMover): Executes transfers with the robot.
        view (View): Display and input.
        config (ChessConfig): Game settings.
        go_look (Callable[[], Any]): Brings the arm to where the camera sees the board (the look pose
            for a wrist camera, out of the view for a fixed one).
        recorder (Recorder | None): Saves every confirmed view as training data.
    """

    def __init__(self, mapping: ChessBoardMapping3D, engine: Any, mover: RobotMover, view: View,
                 config: ChessConfig, go_look: Callable[[], Any], recorder: Recorder | None = None):
        self.mapping, self.engine, self.mover, self.view = mapping, engine, mover, view
        self.config, self.go_look, self.recorder = config, go_look, recorder
        self.human = chess.WHITE if config.game.human_color == "white" else chess.BLACK
        self.tracker: MoveTracker | None = None
        self.graveyard_used = 0
        # Where each piece was last seen (square -> board x, y in metres): a piece hidden in the current
        # view is gripped where it was last seen, not at its square centre (pieces stand up to ~1 cm off).
        self.piece_xy: dict[int, np.ndarray] = {}
        self.state = ViewState(chess.Board(), {}, self.human)

    # ------------------------------------------------------------------ main loop
    def run(self) -> str:
        """Play one game from the starting position. Returns the result ("1-0", "0-1", "1/2-1/2", "*" aborted)."""
        try:
            self.choose_color()
            self.calibrate()
            while not self.tracker.board.is_game_over():
                if self.tracker.board.turn == self.human:
                    self.human_turn()
                else:
                    self.willy_turn()
            outcome = self.tracker.board.outcome()
            result, reason = outcome.result(), OUTCOMES.get(outcome.termination, "")
        except GameAborted:
            self._status("Game aborted.")
            return "*"
        except GameOver as over:
            result, reason = over.result, over.reason
        self._status(f"Game over: {result} ({reason}).")
        return result

    def ask_new_game(self) -> bool:
        """After a game: another one? The human sets up the starting position and clears the graveyard."""
        return self._ask(f"{self.state.status} New game: set up the starting position, clear the graveyard, "
                         f"press space. Esc: end.", {"done", "quit"}) == "done"

    def choose_color(self) -> None:
        """The human picks a side; the space bar keeps the one preselected in the config."""
        preset = "White" if self.human == chess.WHITE else "Black"
        answer = self._ask(f"Which colour do you play? W = White, B = Black, space = {preset}.",
                           {"white", "black", "done", "quit"})
        if answer == "quit":
            raise GameAborted
        if answer != "done":
            self.human = chess.WHITE if answer == "white" else chess.BLACK
        self.state.human_color = self.human

    def calibrate(self) -> None:
        """Calibrate until it succeeds; in a running game against the position the tracker knows."""
        expected = self.tracker.physical if self.tracker is not None else None
        while True:
            self._status("Willy is calibrating the board ...")
            # a refusal is fail-closed and every grab has new noise, so try a few frames before asking
            for _ in range(3):
                result = self._safely("Calibration", lambda: (self.go_look(), self.mapping.calibrate(expected))[1])
                if result.calibration is not None:
                    break
            if result.calibration is not None:
                break
            answer = self._ask(f"Calibration refused: {result.reason}. Check the board, then press space.",
                               {"done", "quit"})
            if answer == "quit":
                raise GameAborted
        if self.tracker is None:
            t = self.config.tracking
            self.tracker = MoveTracker(result.calibration, sigma=t.sigma_squares, gate=t.gate,
                                       class_cost=t.class_cost, promotion_mode=self.config.game.promotion_mode)
        self.piece_xy = {}
        observation = self._observe()
        if observation is not None:
            self._remember_positions(observation, changed=set())
        self._sync()

    # ------------------------------------------------------------------ turns
    def human_turn(self) -> None:
        """Wait for the human's move and recognise it; ask whenever it cannot be confirmed."""
        prompt = "Your move. Press space when you are done (or click the move)."
        while True:
            answer = self._ask(prompt, {"done", "recalibrate", "quit", "move", "resign", "draw"})
            if answer == "quit":
                raise GameAborted
            if answer == "resign":
                raise GameOver("0-1" if self.human == chess.WHITE else "1-0", "you resigned")
            if answer == "draw":
                if self.engine.score(self.tracker.board, not self.human) <= DRAW_ACCEPT_CP:
                    raise GameOver("1/2-1/2", "draw agreed")
                prompt = "Willy declines the draw. Your move."
                continue
            if answer == "recalibrate":
                self.calibrate()
                continue
            if answer.startswith("move:"):
                if self._commit_typed(answer[5:]):
                    return
                prompt = f"{answer[5:]} is not a legal move here. Try again."
                continue

            observation = self._clear_view(self._observe())
            if observation is None:
                prompt = "The board was moved. Press R to calibrate again."
                continue
            before = dict(self.tracker.physical)
            result = self.tracker.step(observation.frame, observation.calib)
            self.state.observation = observation
            if result.flagged:
                self.state.uncertain = self._disagreeing_squares(observation)
                prompt = describe_wrong_move(self.tracker.board, self.tracker.physical, observation.occupancy(),
                                             self.human) or (f"Move not recognised for sure (NLL {result.nll:.1f}). "
                                                             f"Check the board and press space, or click the move.")
                continue
            if result.move is None:
                prompt = "I see no move. Press space when you have moved."
                continue
            if result.promotion_ambiguous:
                self._ask_promotion(result.move)
            self._remember_positions(observation, self._changed_squares(before))
            self.state.uncertain = set()
            self._sync()
            return

    def willy_turn(self) -> None:
        """Stockfish moves, the robot executes, the camera confirms."""
        board = self.tracker.board
        self._status("Willy is thinking ...")
        move = self.engine.best_move(board)
        san = board.san(move)

        observation = self._clear_view(self._observe())       # nothing but pieces over the board before the arm moves
        while observation is None:
            if self._ask("The board was moved. Press R to calibrate again.", {"recalibrate", "quit"}) == "quit":
                raise GameAborted
            self.calibrate()
            observation = self._clear_view(self._observe())

        self._remember_positions(observation, changed=set(), record=False)
        transfers = plan_transfers(board, self.tracker.physical, move, self.piece_xy, observation.calib.square,
                                   self.graveyard_used, self.config.robot.graveyard)
        self._status(f"Willy plays {san}: " + ", ".join(t.describe() for t in transfers))
        try:
            execution = self.mover.execute(transfers, observation)
        except Exception as err:  # noqa: BLE001 (never repeated: a piece may have moved already)
            execution = MoveExecution(ok=False, holding=True, message=f"Willy's move aborted ({err})")
        self.graveyard_used += sum(1 for t in execution.done if t.to_square is None)

        if not execution.ok:
            if execution.holding:
                # never open the hand by itself: the piece would fall onto the board
                self._ask(f"{execution.message}. If Willy still holds a piece, please put a hand under the gripper; "
                          f"space opens it.", {"done"})
                self._safely("Opening the gripper", self.mover.release)
            self._ask(f"{execution.message}. Please finish {san} by hand, then press space.", {"done"})
        elif move.promotion and self.config.game.promotion_mode == "piece_swapped":
            name = PIECE_NAMES[chess.piece_symbol(move.promotion)]
            self._ask(f"Please replace the pawn on {chess.square_name(move.to_square)} with the new piece ({name}), "
                      f"then press space.", {"done"})
        self._confirm_own_move(move, san)

    def _confirm_own_move(self, move: chess.Move, san: str) -> None:
        """Willy's move is known, so the camera only has to confirm it."""
        while True:
            observation = self._observe()
            if observation is not None:
                ok, nll = self.tracker.verify(observation.frame, move, observation.calib)
                self.state.observation = observation
                if ok:
                    before = dict(self.tracker.physical)
                    self.tracker.commit(move)
                    self._remember_positions(observation, self._changed_squares(before))
                    self.state.uncertain = set()
                    self._sync()
                    return
                prompt = (f"After {san} the board looks different than expected (NLL {nll:.1f}). "
                          f"Please fix it, then press space.")
            else:
                prompt = "The board was moved. Press R to calibrate again."
            answer = self._ask(prompt, {"done", "recalibrate", "quit"})
            if answer == "quit":
                raise GameAborted
            if answer == "recalibrate":
                self.calibrate()

    # ------------------------------------------------------------------ helpers
    def _observe(self) -> BoardObservation | None:
        """Look at the board; None if it moved since the calibration."""
        observation = self._safely("Looking at the board", lambda: (self.go_look(), self.mapping.observe())[1])
        self.state.observation = observation
        return None if observation.board_moved else observation

    def _clear_view(self, observation: BoardObservation | None) -> BoardObservation | None:
        """
        Wait until nothing but pieces is over the board (a hand, an arm). After the second request the
        space bar goes on anyway: a piece the detector missed looks the same.
        """
        for attempt in range(3):
            if observation is None or attempt == 2 or observation.intruders() <= INTRUDER_PX:
                return observation
            extra = "" if attempt == 0 else " If there is nothing: space goes on anyway."
            if self._ask(f"Something is over the board, a hand? Please take it away, then press space.{extra}",
                         {"done", "quit"}) == "quit":
                raise GameAborted
            observation = self._observe()
        return observation

    def _safely(self, what: str, action: Callable[[], Any]) -> Any:
        """Run an action of the camera or the arm; if it fails, show why and let the human try again."""
        while True:
            try:
                return action()
            except (GameAborted, GameOver):
                raise
            except Exception as err:  # noqa: BLE001 (the person at the board decides, the game goes on)
                if self._ask(f"{what} failed: {err}. Space: try again, Esc: end.", {"done", "quit"}) == "quit":
                    raise GameAborted from err

    def _remember_positions(self, observation: BoardObservation, changed: set[int], record: bool = True) -> None:
        """
        Update where each piece of the tracked position stands: as seen now if it was matched, else as last
        seen, else (a piece that just arrived and is hidden) its square centre. A confirmed view also goes
        to the recorder, labelled with the pieces the tracker knows are there.
        """
        located = self.tracker.locate(observation.frame, observation.calib)
        if record and self.recorder is not None:
            self.recorder.save(observation.frame, {k: self.tracker.physical[sq] for sq, k in located.items()})
        seen = {sq: observation.grasp_positions[k] for sq, k in located.items() if k in observation.grasp_positions}
        centres = square_centres(observation.calib.square)
        self.piece_xy = {sq: seen[sq] if sq in seen else
                         (self.piece_xy[sq] if sq in self.piece_xy and sq not in changed else centres[sq])
                         for sq in self.tracker.physical}

    def _changed_squares(self, before: dict[int, str]) -> set[int]:
        """Squares whose physical piece changed with the last move (arrived, left or was replaced)."""
        after = self.tracker.physical
        return {sq for sq in set(before) | set(after) if before.get(sq) != after.get(sq)}

    def _ask_promotion(self, move: chess.Move) -> None:
        answer = self._ask("Promote to? Q = queen, R = rook, B = bishop, N = knight.", set(PROMOTION_KEYS))
        if PROMOTION_KEYS[answer] != move.promotion:
            self.tracker.undo()
            self.tracker.commit(chess.Move(move.from_square, move.to_square, promotion=PROMOTION_KEYS[answer]))

    def _commit_typed(self, uci: str) -> bool:
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            return False
        board = self.tracker.board
        if move not in board.legal_moves:
            promoted = chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)
            if promoted not in board.legal_moves:
                return False
            self.tracker.commit(promoted)
            self._ask_promotion(promoted)
        else:
            self.tracker.commit(move)
        self._sync()
        return True

    def _disagreeing_squares(self, observation: BoardObservation) -> set[int]:
        """Squares where the free-mode view and the tracked position disagree on the colour, for the display."""
        seen = {sq: sym.isupper() for sq, sym in observation.occupancy().items()}
        known = {sq: sym.isupper() for sq, sym in self.tracker.physical.items()}
        return {sq for sq in set(seen) | set(known) if seen.get(sq) != known.get(sq)}

    def _sync(self) -> None:
        self.state.board = self.tracker.board.copy() if self.tracker is not None else chess.Board()
        self.state.physical = dict(self.tracker.physical) if self.tracker is not None else {}
        try:
            self.state.evaluation = self.engine.evaluation(self.state.board)
        except Exception:  # noqa: BLE001 (the display can live without an evaluation)
            self.state.evaluation = ""
        self.view.show(self.state)

    def _status(self, text: str) -> None:
        self.state.status = text
        self.view.show(self.state)

    def _ask(self, text: str, accept: set[str]) -> str:
        self.state.status = text
        return self.view.wait(self.state, accept)
