"""A whole game in simulation: human moves, Willy's moves carried out by a fake robot, the camera checking both.

The fake robot really moves the simulated pieces, but only if it grips within 6 mm of where a piece
stands; its graveyard is anything off the board. The human plays random legal moves and only tells
the game "done". Willy's brain is a seeded random player (no Stockfish needed). Promotion runs in
"pawn_stays" mode: there is no spare queen, the pawn stays on the board.

    cd tests
    python -m unittest test_game_sim -v
"""
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

import chess
import numpy as np

TESTS = Path(__file__).resolve().parent
for p in (TESTS, TESTS.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from bench import sim  # noqa: E402
from src.game.config import ChessConfig  # noqa: E402
from src.game.game import ChessGame, ViewState  # noqa: E402
from src.game.robot_moves import RobotMover  # noqa: E402
from src.game.tracker import physical_after  # noqa: E402
from src.mapping.frame import SYMBOL_TO_CLASS  # noqa: E402
from src.mapping.geometry import square_centres  # noqa: E402
from test_mapping_sim import BOARD_TO_BASE, SimDetector, SimPose, SimRig  # noqa: E402

# The open jaws centre a piece along the closing direction; across it only the finger width helps.
GRIP_ALONG_M = 0.008
GRIP_ACROSS_M = 0.005


class GameRig(SimRig):
    """The simulated table: logical game (truth) and the pieces that physically stand on the board."""

    def __init__(self, seed: int):
        super().__init__(seed)
        self.logical = chess.Board()
        self.physical = {s: p.symbol() for s, p in self.logical.piece_map().items()}
        self.graveyard: list[tuple[str, np.ndarray]] = []      # captured pieces beside the board, in view
        self.board = self._render_board()
        self.pose_error_mm = 1.5

    def _carriers(self) -> dict[int, int]:
        """The renderer draws one piece per square: each graveyard piece borrows a free square and an offset."""
        free = [s for s in chess.SQUARES if s not in self.physical]
        return {free[i]: i for i in range(len(self.graveyard))}

    def _render_board(self) -> chess.Board:
        b = chess.Board(None)
        pieces = {s: chess.Piece.from_symbol(sym) for s, sym in self.physical.items()}
        pieces.update({s: chess.Piece.from_symbol(self.graveyard[i][0]) for s, i in self._carriers().items()})
        b.set_piece_map(pieces)
        return b

    def set_physical(self, physical: dict[int, str], placed: dict[int, np.ndarray] | None = None) -> None:
        """New physical pieces; moved ones get a fresh offset (or the one the robot placed them at)."""
        prev_board, prev_offsets = self.board, self.offsets
        self.physical = physical
        self.board = self._render_board()
        self.offsets = sim.jitter_offsets(self.rng, self.board, prev_offsets, prev_board)
        for sq, off in (placed or {}).items():
            self.offsets[sq] = off
        centres = square_centres(self.square)
        for sq, i in self._carriers().items():
            self.offsets[sq] = (self.graveyard[i][1] - centres[sq]) / self.square

    def human_moves(self, move: chess.Move) -> None:
        physical = physical_after(self.logical, self.physical, move, "pawn_stays")
        self.logical.push(move)
        self.set_physical(physical)

    def piece_xy(self, sq: int) -> np.ndarray:
        return square_centres(self.square)[sq] + self.offsets[sq] * self.square


@dataclass
class Report:
    ok: bool
    outcome: str = "SUCCEEDED"


@dataclass
class FakePose:
    xyz: np.ndarray
    heading_deg: float


class FakeRobot:
    """Moves simulated pieces where Willy's poses say, if the grip is close enough to a piece."""

    def __init__(self, rig: GameRig):
        self.rig = rig
        self.held: str | None = None
        self.misses = 0

    def tool_down(self, x, y, z, *, closing_axis="x", yaw_deg=0.0):
        assert closing_axis == "x"
        return FakePose(np.array([x, y, z]), yaw_deg)

    def _board_xy(self, pose: FakePose) -> np.ndarray:
        X = np.linalg.inv(BOARD_TO_BASE) @ np.r_[pose.xyz / 1000.0, 1.0]
        return X[:2]

    def pick(self, pose, width_mm, **kw):
        xy = self._board_xy(pose)
        sq = min(self.rig.physical, key=lambda s: np.linalg.norm(self.rig.piece_xy(s) - xy))
        h = np.deg2rad(pose.heading_deg)
        closing = (BOARD_TO_BASE[:3, :3].T @ np.array([np.cos(h), np.sin(h), 0.0]))[:2]
        off = self.rig.piece_xy(sq) - xy
        along, across = abs(off @ closing), abs(off[0] * closing[1] - off[1] * closing[0])
        if along > GRIP_ALONG_M or across > GRIP_ACROSS_M:
            self.misses += 1
            return Report(False, f"NOTHING_HELD ({along * 1000:.1f} mm along, {across * 1000:.1f} mm across)")
        physical = dict(self.rig.physical)
        self.held = physical.pop(sq)
        self.rig.set_physical(physical)
        return Report(True)

    def place(self, pose, **kw):
        xy = self._board_xy(pose)
        s = self.rig.square
        physical = dict(self.rig.physical)
        placed = {}
        if np.abs(xy / s).max() < 4:                                  # on the board
            sq = chess.square(int(np.floor(xy[0] / s + 4)), int(np.floor(xy[1] / s + 4)))
            physical[sq] = self.held
            placed[sq] = (xy - square_centres(s)[sq]) / s
        else:                                                         # the graveyard, still in view
            self.rig.graveyard.append((self.held, xy))
        self.held = None
        self.rig.set_physical(physical, placed)
        return Report(True)


class RandomEngine:
    def __init__(self, rig: GameRig, seed: int):
        self.rig, self.rng = rig, np.random.default_rng(seed)

    def best_move(self, board: chess.Board) -> chess.Move:
        moves = list(board.legal_moves)
        move = moves[self.rng.integers(len(moves))]
        self.rig.logical.push(move)                                  # the truth Willy is about to make
        return move

    def evaluation(self, board):
        return ""

    def score(self, board, color):
        return 0                                                      # level: a draw offer is accepted


class ScriptedHuman:
    """The display and the human: plays random legal moves, says "done", types the move if not recognised."""

    def __init__(self, rig: GameRig, seed: int, max_plies: int):
        self.rig, self.rng, self.max_plies = rig, np.random.default_rng(seed), max_plies
        self.made: chess.Move | None = None
        self.retries = 0
        self.flags = 0
        self.typed = 0
        self.disagreements = 0
        self.statuses: list[str] = []

    def show(self, state: ViewState) -> None:
        self.statuses.append(state.status)

    def wait(self, state: ViewState, accept: set[str]) -> str:
        self.statuses.append(state.status)
        if self.rig.logical.ply() >= self.max_plies or self.rig.logical.is_game_over():
            return "quit"
        if accept == {"q", "r", "b", "n"}:
            return "q"
        if "white" in accept:
            return "done"                                             # keep the preselected colour
        if "move" in accept:
            if len(state.board.move_stack) == self.rig.logical.ply():      # the game knows every move: play
                moves = list(self.rig.logical.legal_moves)
                self.made = moves[self.rng.integers(len(moves))]
                if self.made.promotion:
                    self.made = chess.Move(self.made.from_square, self.made.to_square, promotion=chess.QUEEN)
                self.rig.human_moves(self.made)
                self.retries = 0
                return "done"
            self.flags += 1                                                  # my move was not recognised
            self.retries += 1
            if self.retries <= 1:
                return "done"
            if self.retries > 4:
                return "quit"
            self.typed += 1
            return f"move:{self.made.uci()}"
        self.disagreements += 1
        return "done" if self.disagreements < 20 else "quit"


class GameSimTest(unittest.TestCase):
    def play(self, seed: int, plies: int, recorder=None, human_cls=None, setup=None):
        rig = GameRig(seed)
        config = ChessConfig()
        from src.mapping.chessboard_mapping_3d import ChessBoardMapping3D
        mapping = ChessBoardMapping3D(rig, SimDetector(rig), SimPose(rig))
        robot = FakeRobot(rig)
        human = (human_cls or ScriptedHuman)(rig, seed + 100, plies)
        game = ChessGame(mapping, RandomEngine(rig, seed + 200), RobotMover(robot, mapping, config.robot), human,
                         config, go_look=lambda: rig.look(35.0, 20.0), recorder=recorder)
        if setup is not None:
            setup(rig, mapping)
        game.run()
        return rig, game, robot, human

    def test_games_stay_in_sync_with_the_real_board(self):
        for seed in range(3):
            rig, game, robot, human = self.play(seed, plies=40)
            with self.subTest(seed=seed):
                self.assertEqual(robot.misses, 0, "the robot gripped next to a piece")
                self.assertEqual(game.tracker.board.fen(), rig.logical.fen())
                self.assertEqual(game.tracker.physical, rig.physical)
                self.assertEqual(human.disagreements, 0, "a move of Willy's was not confirmed by the camera")
                self.assertLessEqual(human.typed, 1, f"{human.typed} human moves had to be typed")
                # where Willy would grip each piece now, against where it really stands (board frame)
                err_mm = [np.linalg.norm(game.piece_xy[s] - rig.piece_xy(s)) * 1000 for s in rig.physical]
                self.assertLess(np.quantile(err_mm, 0.95), 5.0, f"grip positions: {np.round(sorted(err_mm)[-5:], 1)}")


    def test_recorded_labels_name_the_tracked_pieces(self):
        import tempfile

        from src.mapping.frame import CLASSES
        from src.pieces.record import Recorder

        with tempfile.TemporaryDirectory() as folder:
            rig, game, _, _ = self.play(1, plies=6, recorder=Recorder(folder))
            labels = sorted(Path(folder, "labels").glob("*.txt"))
            self.assertEqual(len(labels), len(list(Path(folder, "images").glob("*.png"))))
            self.assertGreaterEqual(len(labels), 7)                   # calibration + one view per confirmed move
            last = [CLASSES[int(line.split()[0])] for line in labels[-1].read_text().splitlines()]
            want = sorted(SYMBOL_TO_CLASS[s] for s in rig.physical.values())
            self.assertLessEqual(len(last), len(want))
            self.assertTrue(set(last) <= set(want))


    def test_an_illegal_move_is_named_and_can_be_taken_back(self):
        class CheatingHuman(ScriptedHuman):
            """Moves the knight g1-g3 first, takes it back when told, then plays on."""
            told: list[str] = []

            def wait(self, state, accept):
                if "not allowed" in state.status:
                    self.told.append(state.status)
                    self.rig.set_physical(self.before)
                elif "move" in accept and not hasattr(self, "before"):
                    self.before = dict(self.rig.physical)
                    physical = dict(self.rig.physical)
                    physical[chess.G3] = physical.pop(chess.G1)
                    self.rig.set_physical(physical)
                    return "done"
                return super().wait(state, accept)

        rig, game, _, human = self.play(2, plies=4, human_cls=CheatingHuman)
        self.assertTrue(any("Knight g1-g3 is not allowed" in t for t in human.told), human.told)
        self.assertEqual(game.tracker.board.fen(), rig.logical.fen())


    def test_resign_and_draw_end_the_game(self):
        for answer, ending in (("resign", "0-1 (you resigned)"), ("draw", "1/2-1/2 (draw agreed)")):
            class Ending(ScriptedHuman):
                def wait(self, state, accept):
                    return answer if answer in accept else super().wait(state, accept)

            _, _, _, human = self.play(3, plies=10, human_cls=Ending)
            self.assertTrue(any(ending in st for st in human.statuses), ending)

    def test_a_hand_over_the_board_is_waited_for(self):
        class Reaching(ScriptedHuman):
            """Leaves the hand over the board after the first move and takes it away when asked."""
            asked = 0

            def wait(self, state, accept):
                if "a hand?" in state.status:
                    Reaching.asked += 1
                    self.rig.hand = None
                    return "done"
                answer = super().wait(state, accept)
                if answer == "done" and self.rig.logical.ply() == 1 and not Reaching.asked:
                    self.rig.hand = ((260, 180, 380, 260), 0.08)
                return answer

        rig, game, _, _ = self.play(4, plies=4, human_cls=Reaching)
        self.assertEqual(Reaching.asked, 1)
        self.assertEqual(game.tracker.board.fen(), rig.logical.fen())

    def test_a_failing_camera_is_reported_and_retried(self):
        def flaky(rig, mapping):
            observe, calls = mapping.observe, []

            def once_broken():
                calls.append(1)
                if len(calls) == 3:
                    raise RuntimeError("camera gone")
                return observe()
            mapping.observe = once_broken

        rig, game, _, human = self.play(5, plies=4, setup=flaky)
        self.assertTrue(any("Looking at the board failed: camera gone" in st for st in human.statuses))
        self.assertEqual(game.tracker.board.fen(), rig.logical.fen())

    def test_a_finished_game_is_saved_as_pgn(self):
        import tempfile

        from src.game.play import save_pgn

        board = chess.Board()
        for san in ("f3", "e5", "g4", "Qh4#"):
            board.push_san(san)
        with tempfile.TemporaryDirectory() as folder:
            text = save_pgn(board, "0-1", chess.WHITE, Path(folder)).read_text()
        self.assertIn('[White "Human"]', text)
        self.assertIn("2. g4 Qh4# 0-1", text)


class WrongMoveTest(unittest.TestCase):
    """The message when the board shows a move the rules forbid."""

    def explain(self, fen: str, uci: str, human=chess.WHITE):
        from src.game.game import describe_wrong_move

        board = chess.Board(fen)
        physical = {s: p.symbol() for s, p in board.piece_map().items()}
        seen = dict(physical)
        move = chess.Move.from_uci(uci)
        seen[move.to_square] = seen.pop(move.from_square)
        return describe_wrong_move(board, physical, seen, human)

    def test_a_piece_moved_against_its_rules(self):
        self.assertIn("this piece cannot move like that", self.explain(chess.STARTING_FEN, "g1g3"))

    def test_a_move_into_check(self):
        msg = self.explain("4r1k1/8/8/8/8/8/4B3/4K3 w - - 0 1", "e2d3")       # the bishop is pinned
        self.assertIn("king would be in check", msg)

    def test_a_piece_of_willys_moved(self):
        self.assertIn("piece of Willy's", self.explain(chess.STARTING_FEN, "e7e5"))

    def test_a_legal_move_is_no_wrong_move(self):
        self.assertIsNone(self.explain(chess.STARTING_FEN, "e2e4"))


if __name__ == "__main__":
    unittest.main()
