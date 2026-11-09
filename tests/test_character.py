"""Willy as a character: what he says, how he thinks out loud, and how the window draws him.

    cd tests
    python -m unittest test_character -v
"""
import os
import sys
import unittest
from pathlib import Path

import chess

TESTS = Path(__file__).resolve().parent
for p in (TESTS, TESTS.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.game.config import ChessConfig, EngineConfig  # noqa: E402
from src.game.engine import Candidate, Thought  # noqa: E402
from src.game.game import ChessGame, ViewState  # noqa: E402
from src.game.persona import LINES, Persona, decided_event, human_move_event, willy_pawns  # noqa: E402
from src.game.robot_moves import RobotMover  # noqa: E402
from src.mapping.chessboard_mapping_3d import ChessBoardMapping3D  # noqa: E402
from test_game_sim import FakeRobot, GameRig, RandomEngine, ScriptedHuman  # noqa: E402
from test_mapping_sim import SimDetector, SimPose  # noqa: E402


class PersonaTest(unittest.TestCase):
    def test_the_same_moment_never_sounds_the_same_twice_in_a_row(self):
        persona = Persona(seed=1)
        for event, (_, lines) in LINES.items():
            if len(lines) > 1:
                said = [persona.react(event, san="Nf3", piece="knight")[1] for _ in range(20)]
                self.assertTrue(all(a != b for a, b in zip(said, said[1:])), event)

    def test_every_mood_has_a_face(self):
        from src.ui.face import BROWS

        self.assertLessEqual({mood for mood, _ in LINES.values()}, set(BROWS))

    def test_willy_remarks_on_what_the_human_did(self):
        board = chess.Board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
        self.assertEqual(human_move_event(board, chess.Move.from_uci("e4d5"), 0.0, -1.0),
                         ("lost_piece", {"piece": "pawn"}))
        check = chess.Board("rnbqkbnr/ppp2ppp/8/3pp3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 3")
        self.assertEqual(human_move_event(check, chess.Move.from_uci("f1b5"), 0.0, 0.0)[0], "in_check")
        quiet = chess.Move.from_uci("g1f3")
        self.assertEqual(human_move_event(check, quiet, 0.0, -2.0)[0], "good_move")
        self.assertEqual(human_move_event(check, quiet, 0.0, 2.0)[0], "blunder")
        self.assertEqual(human_move_event(check, quiet, 0.0, 0.3)[0], "seen")

    def test_willy_announces_his_own_move(self):
        mate = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2")
        self.assertEqual(decided_event(mate, mate.parse_san("Qh4#")), ("decided_mate", {"san": "Qh4#"}))
        board = chess.Board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
        self.assertEqual(decided_event(board, board.parse_san("exd5")),
                         ("decided_capture", {"san": "exd5", "piece": "pawn"}))
        self.assertEqual(decided_event(board, board.parse_san("Nf3"))[0], "decided")

    def test_the_evaluation_is_read_from_willys_side(self):
        self.assertEqual(willy_pawns("+0.35", chess.WHITE), 0.35)
        self.assertEqual(willy_pawns("+0.35", chess.BLACK), -0.35)
        self.assertEqual(willy_pawns("M3", chess.BLACK), -100.0)
        self.assertEqual(willy_pawns("M-2", chess.BLACK), 100.0)
        self.assertIsNone(willy_pawns("", chess.WHITE))


class ThinkingEngine(RandomEngine):
    """The random engine, thinking out loud first: two candidates, the second one taking the lead."""

    def think(self, board, on_thought):
        moves = list(board.legal_moves)[:2]
        lines = [Candidate(m, board.san(m), 20 - 10 * i) for i, m in enumerate(moves)]
        for depth, order in ((4, lines), (9, lines[::-1])):
            on_thought(Thought(depth, tuple(order)))
        return self.best_move(board)


class ThinkingGameTest(unittest.TestCase):
    def test_willy_thinks_out_loud_and_the_game_stays_in_sync(self):
        rig = GameRig(3)
        config = ChessConfig()
        config.game.think_show_s = 1.5            # past Willy's word on the human's move, time to weigh again
        mapping = ChessBoardMapping3D(rig, SimDetector(rig), SimPose(rig))

        class Watching(ScriptedHuman):
            seen: list = []

            def show(self, state):
                Watching.seen.append((state.mood, state.focus, state.thought, state.intent))
                super().show(state)

        human = Watching(rig, 103, 4)
        game = ChessGame(mapping, ThinkingEngine(rig, 203), RobotMover(FakeRobot(rig), mapping, config.robot), human,
                         config, go_look=lambda: rig.look(35.0, 20.0))
        game.run()
        self.assertEqual(game.tracker.board.fen(), rig.logical.fen())
        thinking = [s for s in Watching.seen if s[0] == "thinking" and s[1] is not None and s[2] is not None]
        self.assertTrue(thinking, "Willy never looked at a candidate while thinking")
        self.assertTrue(any(intent is not None for *_, intent in Watching.seen), "no arrow while the arm moved")
        self.assertIsNone(game.state.intent)


class WindowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    def test_every_mood_is_drawn(self):
        from src.ui.face import BROWS
        from src.ui.pygame_view import PygameView

        board = chess.Board()
        board.push_san("e4")
        thought = Thought(12, tuple(Candidate(m, board.san(m), 30 - i) for i, m in enumerate(list(board.legal_moves)[:3])))
        with PygameView() as view:
            for mood in BROWS:
                for human in (chess.WHITE, chess.BLACK):
                    state = ViewState(board, {}, human, "Willy is thinking ...", "+0.30", mood=mood, speech="Hmm...",
                                      thought=thought, focus=thought.candidates[0].move)
                    view.show(state)
            from src.game.game import TakenOff
            from src.game.tracker import board_without

            taken = TakenOff(board.copy(), chess.G1, chess.Piece(chess.KNIGHT, chess.WHITE), 4)
            for strikes in (1, 3, 4):                       # the patience runs out, a piece is taken off
                view.show(ViewState(board_without(board, chess.G1), {}, chess.WHITE, "Willy takes your knight off.",
                                    "+3.10", mood="angry", speech="That's it!", taking=chess.G1, strikes=strikes,
                                    patience=3, earlier=[taken]))
            view._asking, view._waiting_since = True, 0.0                         # long enough to doze off
            view._draw(ViewState(board, {}, chess.BLACK, "Your move.", "M-3", mood="idle", speech="Your move."))
            board.push_san("e5")
            view.show(ViewState(board, {}, chess.WHITE, "", "", intent=chess.Move.from_uci("g1f3")))   # slides, then returns


@unittest.skipIf(__import__("shutil").which("stockfish") is None and sys.platform != "win32", "no Stockfish here")
class StockfishThinkingTest(unittest.TestCase):
    def test_the_thoughts_are_legal_moves_and_the_answer_is_one(self):
        from src.game.engine import Engine

        board = chess.Board()
        thoughts = []
        with Engine(EngineConfig(think_time_s=0.3)) as engine:
            move = engine.think(board, thoughts.append)
        self.assertIn(move, board.legal_moves)
        lines = [c for t in thoughts for c in t.candidates]
        self.assertTrue(lines)
        self.assertTrue(all(c.move in board.legal_moves and board.san(c.move) == c.san for c in lines))
        self.assertGreaterEqual(max(t.depth for t in thoughts), 2)


if __name__ == "__main__":
    unittest.main()
