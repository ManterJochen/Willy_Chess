"""What can only go wrong at the cell, held down at the desk: dead ends, the robot's clearances, the detector's duplicates.

    cd tests
    python -m unittest test_regressions -v
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import chess
import numpy as np

TESTS = Path(__file__).resolve().parent
for p in (TESTS, TESTS.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.game.config import ChessConfig, RobotConfig, load_config  # noqa: E402
from src.game.game import ChessGame, GameAborted  # noqa: E402
from src.game.play import play_games  # noqa: E402
from src.game.robot_moves import JawGeometry, RobotMover, Transfer, plan_transfers  # noqa: E402
from src.game.tracker import MoveTracker  # noqa: E402
from src.mapping.chessboard_mapping_3d import ChessBoardMapping3D  # noqa: E402
from src.mapping.frame import BoardFrame  # noqa: E402
from src.mapping.geometry import BoardCalibration, square_centres  # noqa: E402
from test_game_sim import FakeRobot, GameRig, RandomEngine, Report, ScriptedHuman  # noqa: E402
from test_mapping_sim import SimDetector, SimPose  # noqa: E402

def new_game(seed: int, plies: int, human_cls=ScriptedHuman, robot_cls=FakeRobot, engine=None, config=None):
    rig = GameRig(seed)
    config = config or ChessConfig()
    mapping = ChessBoardMapping3D(rig, SimDetector(rig), SimPose(rig))
    robot = robot_cls(rig)
    human = human_cls(rig, seed + 100, plies)
    game = ChessGame(mapping, engine or RandomEngine(rig, seed + 200), RobotMover(robot, mapping, config.robot), human,
                     config, go_look=lambda: rig.look(35.0, 20.0))
    return rig, game, robot, human


class DeadEndTest(unittest.TestCase):
    """A game must not end because the camera keeps disagreeing with a board that is right."""

    def test_the_person_at_the_board_can_overrule_the_camera_on_willys_move(self):
        class Overrules(ScriptedHuman):
            offered = 0

            def wait(self, state, accept):
                if "force" in accept:
                    Overrules.offered += 1
                    return "force"
                return super().wait(state, accept)

        rig, game, _, _ = new_game(1, plies=6, human_cls=Overrules)
        with mock.patch.object(MoveTracker, "verify", lambda *args, **kwargs: (False, 99.0)):    # never confirmed
            game.run()
        self.assertGreater(Overrules.offered, 0)
        self.assertGreaterEqual(game.tracker.board.ply(), 4, "the game went on past Willy's moves")
        played = game.tracker.board.move_stack
        self.assertEqual(played, rig.logical.move_stack[:len(played)])

    def test_a_board_bumped_during_willys_move_can_be_calibrated_again(self):
        """The position to calibrate against is the one after Willy's move, not the one before it."""
        asked = []
        rig, game, _, human = new_game(2, plies=2)
        game.choose_color = lambda: None
        real = game.mapping.calibrate

        def spy(expected=None):
            asked.append(None if expected is None else dict(expected))
            return real(expected)

        game.mapping.calibrate = spy
        game.calibrate()
        move = chess.Move.from_uci("e2e4")
        shown = dict(game.tracker.physical)
        shown[chess.E4] = shown.pop(chess.E2)
        rig.logical.push(move)
        rig.set_physical(shown)
        real_observe, bumps = game.mapping.observe, []

        def bumped_once():
            observation = real_observe()
            if not bumps:
                bumps.append(1)
                observation.pose_status = "moved"
            return observation

        game.mapping.observe = bumped_once
        human.wait = lambda state, accept: "recalibrate" if "recalibrate" in accept else "done"
        game._confirm_own_move(move, "e4")
        self.assertEqual(asked[-1], shown)
        self.assertEqual(game.tracker.board.peek(), move)

    def test_a_crashed_game_keeps_its_moves(self):
        class Crash(ScriptedHuman):
            def wait(self, state, accept):
                if self.rig.logical.ply() >= 2 and "move" in accept:
                    raise KeyError("something nobody expected")
                return super().wait(state, accept)

        rig, game, _, human = new_game(3, plies=10, human_cls=Crash)
        with tempfile.TemporaryDirectory() as folder:
            config = ChessConfig()
            config.game.pgn_dir = folder
            cell = SimpleNamespace(mapping=game.mapping, mover=game.mover, go_look=game.go_look)
            with self.assertRaises(KeyError):
                play_games(cell, game.engine, human, config)
            games = list(Path(folder).glob("*.pgn"))
            self.assertEqual(len(games), 1)
            self.assertIn('[Result "*"]', games[0].read_text())

    def test_a_crashed_engine_is_asked_again_not_the_end_of_the_game(self):
        rig, game, _, human = new_game(4, plies=3)
        engine, calls = game.engine, []

        def flaky(board):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("engine gone")
            return RandomEngine.best_move(engine, board)

        game.engine = SimpleNamespace(best_move=flaky, evaluation=engine.evaluation, score=engine.score)
        game.run()
        self.assertTrue(any("Stockfish failed: engine gone" in st for st in human.statuses))
        self.assertEqual(game.tracker.board.fen(), rig.logical.fen())


class GraveyardTest(unittest.TestCase):
    def test_a_slot_is_used_up_even_if_the_arm_did_not_finish(self):
        """An exception after the capture was carried away lost the count: the next piece went onto the same slot."""
        rig, game, _, human = new_game(5, plies=2)
        game.choose_color = lambda: None
        game.calibrate()
        for uci in ("e2e4", "d7d5"):
            move = chess.Move.from_uci(uci)
            shown = dict(game.tracker.physical)
            shown[move.to_square] = shown.pop(move.from_square)
            rig.logical.push(move)
            rig.set_physical(shown)
            game.tracker.commit(move)
        game.human = chess.BLACK
        capture = chess.Move.from_uci("e4d5")

        def execute(transfers, observation):
            raise RuntimeError("protective stop on the way")

        game.mover.execute, game.mover.release = execute, lambda: None
        game.engine = SimpleNamespace(best_move=lambda board: capture, evaluation=lambda board: "", score=lambda *a: 0)

        def finished_by_hand(state, accept):
            if "by hand" in state.status:
                rig.logical.push(capture)
                shown = dict(rig.physical)
                shown[capture.to_square] = shown.pop(capture.from_square)
                rig.set_physical(shown)
            return "done"

        human.wait = finished_by_hand
        game.willy_turn()
        self.assertEqual(game.graveyard_used, 1)


class RobotTest(unittest.TestCase):
    def rig(self, physical: dict[int, str]):
        """A mover with a recording robot, and a view of these pieces on a board of 57 mm squares."""
        square = 0.057
        centres = square_centres(square)
        keys = {sq: i + 1 for i, sq in enumerate(physical)}
        frame = BoardFrame(np.zeros((4, 4, 3), np.uint8), np.eye(3), None, np.zeros((4, 4), np.int32),
                           {keys[sq]: sym for sq, sym in physical.items()})
        observation = SimpleNamespace(calib=BoardCalibration(np.eye(3), np.zeros(3), square), frame=frame,
                                      grasp_positions={keys[sq]: centres[sq] for sq in physical},
                                      positions={}, squares={})
        calls = []

        class Robot:
            def tool_down(self, x, y, z, *, closing_axis="x", yaw_deg=0.0):
                return SimpleNamespace(xyz=(x, y, z), yaw=yaw_deg)

            def pick(self, pose, width_mm, **kw):
                calls.append(("pick", pose, kw))
                return Report(True)

            def place(self, pose, **kw):
                calls.append(("place", pose, kw))
                return Report(True)

        mapping = SimpleNamespace(grasp_width=lambda symbol: 0.02,
                                  board_to_base_mm=lambda xy, z=0.0: np.array([xy[0], xy[1], z]) * 1000.0,
                                  board_axis_base=lambda d: np.asarray(d, float))
        return RobotMover(Robot(), mapping, RobotConfig(), JawGeometry()), observation, calls, centres, square

    @staticmethod
    def along_the_rank(pose) -> float:
        """How much the jaws of a pose open along the rank: 1 along it, 0 along the file."""
        return abs(np.cos(np.deg2rad(pose.yaw)))

    def test_a_carried_piece_clears_the_tallest_piece(self):
        """Foot at the 80 mm standoff above the board, a king stands 95 mm: the arm would carry a piece into it."""
        mover, observation, calls, centres, _ = self.rig({chess.E2: "P", chess.E1: "K"})
        transfer = Transfer("P", centres[chess.E2], centres[chess.E4], from_square=chess.E2, to_square=chess.E4)
        self.assertTrue(mover.execute([transfer], observation).ok)
        tallest_mm = 1000 * observation.calib.piece_dims("k")[0]
        self.assertEqual(len(calls), 2)
        for _, _, kw in calls:
            self.assertGreater(kw["standoff_mm"], tallest_mm)

    def test_the_rook_of_a_castling_is_gripped_with_the_king_where_it_stands_now(self):
        """The view is from before the king moved: without g1 the jaws opened along the rank, into the king."""
        board = chess.Board("4k3/8/8/8/8/8/6PP/4K2R w K - 0 1")
        physical = {s: p.symbol() for s, p in board.piece_map().items()}
        mover, observation, calls, _, square = self.rig(physical)
        transfers = plan_transfers(board, physical, chess.Move.from_uci("e1g1"), {}, square, 0,
                                   ChessConfig().robot.graveyard)
        self.assertTrue(mover.execute(transfers, observation).ok)
        rook_pick = [pose for verb, pose, _ in calls if verb == "pick"][1]
        self.assertLess(self.along_the_rank(rook_pick), 0.8)

    def test_the_hand_opens_at_the_target_where_its_neighbours_leave_room(self):
        """Gripped along the rank (a neighbour on the file), set down between two pieces on the rank: not into them."""
        mover, observation, calls, centres, _ = self.rig({chess.A1: "R", chess.A2: "P", chess.C5: "p", chess.E5: "p"})
        transfer = Transfer("R", centres[chess.A1], centres[chess.D5], from_square=chess.A1, to_square=chess.D5)
        self.assertTrue(mover.execute([transfer], observation).ok)
        (_, picked, _), (_, placed, _) = calls
        self.assertGreater(self.along_the_rank(picked), 0.8)
        self.assertLess(self.along_the_rank(placed), 0.8)


class DuplicateTest(unittest.TestCase):
    def mapping(self, objects):
        detector = SimpleNamespace(detect=lambda crop, segment, threshold: objects)
        return ChessBoardMapping3D(None, detector, None)

    def obj(self, label, score, mask):
        return SimpleNamespace(label=label, score=score, mask=mask)

    def test_one_piece_seen_twice_counts_once_and_keeps_the_better_score(self):
        """RT-DETR has no NMS: a second box of a piece made one piece too many, and with it every move impossible."""
        a = np.zeros((40, 40), bool)
        a[5:25, 5:15] = True
        twice = a.copy()
        twice[5:26, 5:15] = True                                       # the same piece, its box a little different
        other = np.zeros((40, 40), bool)
        other[5:25, 20:30] = True
        frame = BoardFrame(np.zeros((40, 40, 3), np.uint8), np.eye(3))
        seen = self.mapping([self.obj("white pawn", 0.6, twice), self.obj("white pawn", 0.9, a),
                             self.obj("black pawn", 0.8, other)])._with_pieces(frame, None)
        self.assertEqual(sorted(seen.classes.values()), ["P", "p"])
        self.assertEqual(sorted(seen.scores.values()), [0.8, 0.9])

    def test_a_piece_in_front_of_another_stays(self):
        back = np.zeros((40, 40), bool)
        back[5:25, 5:15] = True
        front = np.zeros((40, 40), bool)
        front[15:35, 5:15] = True
        frame = BoardFrame(np.zeros((40, 40, 3), np.uint8), np.eye(3))
        seen = self.mapping([self.obj("white pawn", 0.9, back), self.obj("white queen", 0.8, front)])._with_pieces(frame, None)
        self.assertEqual(sorted(seen.classes.values()), ["P", "Q"])


class CalibrationCropTest(unittest.TestCase):
    def rig_and_mapping(self, seed: int):
        rig = GameRig(seed)
        rig.look(35.0, 20.0)
        return rig, SimDetector(rig), ChessBoardMapping3D

    def test_the_detector_sees_the_board_not_the_room(self):
        rig, detector, mapping_cls = self.rig_and_mapping(6)
        mapping = mapping_cls(rig, detector, SimPose(rig))
        image = np.asarray(rig.grab().color)
        h, w = image.shape[:2]
        room = np.zeros((2 * h, 2 * w, 3), np.uint8)
        room[:h, :w] = image
        x0, y0, x1, y1 = mapping._grid_roi(BoardFrame(room, rig.get_intrinsics()))
        self.assertLess((x1 - x0) * (y1 - y0), 0.5 * room.shape[0] * room.shape[1])
        self.assertIsNone(mapping._grid_roi(BoardFrame(np.zeros((h, w, 3), np.uint8), rig.get_intrinsics())))

    def test_the_first_try_on_the_crop_is_enough_where_the_grid_is_right(self):
        rig, detector, mapping_cls = self.rig_and_mapping(6)
        crops, real = [], detector.detect

        def spy(crop, *, segment=False, threshold=None):
            crops.append(crop.shape[:2])
            return real(crop, segment=segment, threshold=threshold)

        detector.detect = spy
        self.assertEqual(mapping_cls(rig, detector, SimPose(rig)).calibrate().reason, "ok")
        self.assertEqual(len(crops), 1)

    def test_a_crop_that_goes_wrong_falls_back_to_the_whole_image(self):
        rig, detector, mapping_cls = self.rig_and_mapping(7)
        mapping = mapping_cls(rig, detector, SimPose(rig))
        mapping._grid_roi = lambda frame: (0, 0, 40, 40)               # a corner of the image, no piece in it
        self.assertEqual(mapping.calibrate().reason, "ok")


class ConfigTest(unittest.TestCase):
    def test_a_config_file_that_is_not_there_is_refused(self):
        with self.assertRaises(ValueError) as refused:
            load_config("config/no_such_file.yaml")
        self.assertIn("no_such_file", str(refused.exception))

    def test_no_config_named_keeps_the_shipped_one(self):
        self.assertIsNotNone(load_config(None).game)


class InputTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    def test_a_key_pressed_before_the_question_answers_nothing(self):
        """A stray space bar while the arm moved opened the gripper the moment the game asked for a hand under it."""
        import pygame

        from src.game.game import ViewState
        from src.ui.pygame_view import PygameView

        with PygameView() as view:
            state = ViewState(chess.Board(), {}, chess.WHITE)
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))

            class Later:
                def tick(self, _fps):                                  # a key pressed after the question came
                    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))

            view.clock = Later()
            self.assertEqual(view.wait(state, {"done", "quit"}), "quit")

    def test_a_closed_window_is_not_thrown_away(self):
        import pygame

        from src.game.game import ViewState
        from src.ui.pygame_view import PygameView

        with PygameView() as view:
            pygame.event.post(pygame.event.Event(pygame.QUIT))
            with self.assertRaises(GameAborted):
                view.wait(ViewState(chess.Board(), {}, chess.WHITE), {"done"})

    def test_f_is_only_an_answer_where_it_is_offered(self):
        import pygame

        from src.game.game import ViewState
        from src.ui.pygame_view import PygameView

        with PygameView() as view:
            state = ViewState(chess.Board(), {}, chess.WHITE)
            f = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_f)
            self.assertIsNone(view._answer(f, state, {"done", "recalibrate"}))
            self.assertEqual(view._answer(f, state, {"done", "force"}), "force")


class WeightsTest(unittest.TestCase):
    """The shipped checkpoint and the threshold belong together: docs/weights/.../config.json and the yaml."""

    def test_the_threshold_is_a_real_score_again(self):
        self.assertGreaterEqual(load_config(None).detector.threshold, 0.3)

    def test_the_faulty_last_layer_of_this_checkpoint_stays_off(self):
        import json

        folder = TESTS.parent / load_config(None).detector.weights
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        if manifest.get("metrics", {}).get("true_positive_score") is None:     # trained before Willy's loss repair
            self.assertEqual(config["decoder_layers"], 5)


if __name__ == "__main__":
    unittest.main()
