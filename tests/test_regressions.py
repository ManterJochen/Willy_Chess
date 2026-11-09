"""What can only go wrong at the cell, held down at the desk: dead ends, the robot's clearances, the detector's duplicates.

    cd tests
    python -m unittest test_regressions -v
"""
import io
import os
import sys
import tempfile
import unittest
from contextlib import contextmanager, redirect_stdout
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
from src.game import mapping_test  # noqa: E402
from src.game.play import Cell, look_stops, play_games  # noqa: E402
from src.game.robot_moves import JawGeometry, RobotMover, Transfer, graveyard_slot, plan_transfers  # noqa: E402
from src.game.tracker import MoveTracker, TrackResult  # noqa: E402
from src.mapping.calibration import measure_grasp_widths  # noqa: E402
from src.mapping.chessboard_mapping_3d import ChessBoardMapping3D  # noqa: E402
from src.mapping.frame import BoardFrame  # noqa: E402
from src.mapping.geometry import BoardCalibration, square_centres  # noqa: E402
from src.mapping.pieces import fuse  # noqa: E402
from test_game_sim import FakeRobot, GameRig, RandomEngine, Report, ScriptedHuman  # noqa: E402
from test_mapping_sim import SimDetector, SimPose  # noqa: E402

LOOKS = ((35.0, 20.0), (35.0, 200.0), (35.0, 110.0))      # tilt, azimuth: the first pose and two from other sides


def new_game(seed: int, plies: int, human_cls=ScriptedHuman, robot_cls=FakeRobot, engine=None, config=None, views=1):
    rig = GameRig(seed)
    config = config or ChessConfig()
    mapping = ChessBoardMapping3D(rig, SimDetector(rig), SimPose(rig))
    robot = robot_cls(rig)
    human = human_cls(rig, seed + 100, plies)

    def go_look(view: int = 0) -> None:
        rig.visited = getattr(rig, "visited", []) + [view]
        rig.look(*LOOKS[view])

    game = ChessGame(mapping, engine or RandomEngine(rig, seed + 200), RobotMover(robot, mapping, config.robot), human,
                     config, go_look=go_look, views=views)
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
        with mock.patch.object(MoveTracker, "explains", lambda *args, **kwargs: (False, 99.0)):  # never confirmed
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
            cell = SimpleNamespace(mapping=game.mapping, mover=game.mover, go_look=game.go_look, views=1)
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

    def test_neighbouring_slots_leave_room_for_the_housing(self):
        """The Hand-E's housing reaches 37.5 mm to each side, a queen's crown about 15 mm: one square of 45 mm apart,
        a piece set down beside a captured queen put the housing into it, where no camera sees and nothing refuses."""
        config = ChessConfig().robot.graveyard
        slots = [graveyard_slot(i, 0.045, config) for i in range(2 * 8)]
        nearest = min(np.linalg.norm(a - b) for i, a in enumerate(slots) for b in slots[i + 1:])
        self.assertGreater(nearest, 0.0375 + 0.015)


class StopTest(unittest.TestCase):
    def test_a_stopped_arm_holds_the_game_until_the_stop_is_cleared(self):
        """The controller stopped the arm on contact. Told to finish by hand as after a refusal, a person reached in
        beside an arm that had just met something, and the game moved it again with the same numbers."""
        rig, game, robot, human = new_game(5, plies=2)
        game.choose_color = lambda: None
        game.calibrate()
        game.human = chess.BLACK
        move = chess.Move.from_uci("e2e4")
        game.engine = SimpleNamespace(best_move=lambda board: move, evaluation=lambda board: "", score=lambda *a: 0)
        reads = [True, True]                    # the stop, still there after the first space, cleared after the second

        def status():
            stopped = reads.pop(0) if reads else False
            return SimpleNamespace(protective_stopped=stopped, emergency_stopped=False, controller_ready=not stopped)

        robot.arm = SimpleNamespace(get_robot_status=status)
        robot.pick = lambda pose, width_mm, **kw: Report(False, "MOTION_REFUSED")
        asked = []

        def person(state, accept):
            asked.append(state.status)
            if "by hand" in state.status:
                rig.logical.push(move)
                shown = dict(rig.physical)
                shown[move.to_square] = shown.pop(move.from_square)
                rig.set_physical(shown)
            return "done"

        human.wait = person
        game.willy_turn()
        stops = [i for i, text in enumerate(asked) if text.startswith("The arm stopped (protective stop)")]
        self.assertEqual(len(stops), 2)
        self.assertLess(stops[-1], next(i for i, text in enumerate(asked) if "by hand" in text))
        self.assertEqual(game.tracker.board.fen(), rig.logical.fen())


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
                calls.append(("pick", pose, dict(kw, width_mm=width_mm)))
                return Report(True)

            def place(self, pose, **kw):
                calls.append(("place", pose, kw))
                return Report(True)

        mapping = SimpleNamespace(grasp_width=lambda symbol: 0.02, clear_width=lambda symbol: 0.02,
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

    def test_the_jaws_open_past_a_knights_head_and_close_on_its_neck(self):
        """Closed to the head's width the jaws stop short of the neck and hold nothing; opened to the neck's, the
        fingers come down onto the head."""
        mover, observation, calls, centres, _ = self.rig({chess.G1: "N"})
        mover.mapping.clear_width = lambda symbol: 0.03
        transfer = Transfer("N", centres[chess.G1], centres[chess.F3], from_square=chess.G1, to_square=chess.F3)
        self.assertTrue(mover.execute([transfer], observation).ok)
        (_, _, picked), _ = calls
        self.assertAlmostEqual(picked["width_mm"], 20.0)
        self.assertAlmostEqual(picked["pre_open_mm"], 30.0 + 2 * mover.config.finger_clearance_mm)

    def test_a_knight_turned_since_the_calibration_is_not_gripped_across_its_head(self):
        """Measured once at the calibration, the knights' width said nothing of a head that now reaches 26 mm forward
        along the rank: jaws closing along the rank came down onto it (a protective stop, 2026-11-05)."""
        mover, observation, calls, centres, _ = self.rig({chess.G1: "N"})
        height = observation.calib.piece_dims("n")[0]
        angles = np.linspace(0, 2 * np.pi, 72, endpoint=False)
        neck = np.column_stack((0.007 * np.cos(angles), 0.007 * np.sin(angles), np.full(72, 0.5 * height)))
        head = np.column_stack((np.linspace(-0.008, 0.026, 72), np.zeros(72), np.full(72, 0.9 * height)))
        points = np.vstack((neck, head)) + np.r_[centres[chess.G1], 0.0]
        observation.squares, observation.piece_points = {1: chess.G1}, lambda label: points
        transfer = Transfer("N", centres[chess.G1], centres[chess.F3], from_square=chess.G1, to_square=chess.F3)
        self.assertTrue(mover.execute([transfer], observation).ok)
        (_, picked, _), _ = calls
        self.assertLess(self.along_the_rank(picked), 0.8)


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


class GraspWidthTest(unittest.TestCase):
    def test_the_jaws_close_on_a_knights_neck_and_open_past_its_head(self):
        """Its head overhangs the neck: with one width from the grasp height up the jaws closed on nothing."""
        angles = np.linspace(0, 2 * np.pi, 72, endpoint=False)
        rings = [np.column_stack(((0.007 if z < 0.04 else 0.015) * np.cos(angles),
                                  (0.007 if z < 0.04 else 0.008) * np.sin(angles), np.full(len(angles), z)))
                 for z in np.linspace(0.0, 0.06, 61)]
        camera = SimpleNamespace(board_points=lambda depth, mask: np.vstack(rings), foot=np.array([0.0, -0.5]))
        frame = BoardFrame(np.zeros((4, 4, 3), np.uint8), np.eye(3), np.ones((4, 4), np.float32),
                           np.ones((4, 4), np.int32), {1: "N"})
        calib = BoardCalibration(np.eye(3), np.zeros(3), 0.045, {"n": (0.06, 0.016)})
        with mock.patch("src.mapping.calibration.BoardCamera", return_value=camera):
            at, above = measure_grasp_widths(frame, calib, {"n": 0.6})
        self.assertAlmostEqual(at["n"], 0.014, delta=0.001)
        self.assertAlmostEqual(above["n"], 0.030, delta=0.001)


class PositionTest(unittest.TestCase):
    """Foot point and depth, each wrong in its own way (measured on a real set, see fuse)."""
    F = np.zeros(2)

    def test_the_depth_gives_the_direction_and_both_the_distance(self):
        np.testing.assert_allclose(fuse(np.array([0.29, 0.01]), np.array([0.30, 0.0]), self.F), [0.295, 0.0])

    def test_a_foot_behind_the_depth_says_nothing(self):
        depth = np.array([0.30, 0.0])
        np.testing.assert_allclose(fuse(np.array([0.302, 0.0]), depth, self.F), depth)
        np.testing.assert_allclose(fuse(np.array([0.33, 0.01]), depth, self.F), depth)

    def test_without_depth_the_foot_alone(self):
        foot = np.array([0.29, 0.01])
        np.testing.assert_allclose(fuse(foot, None, self.F), foot)

    def test_no_mean_of_two_that_contradict_each_other(self):
        foot, depth = np.array([0.26, 0.0]), np.array([0.30, 0.0])
        np.testing.assert_allclose(fuse(foot, depth, self.F), foot)
        np.testing.assert_allclose(fuse(foot, depth, self.F, for_grasp=True), depth)


class OneSquareOffTest(unittest.TestCase):
    """A whole square is less than the gate on the sum: it has to be refused piece by piece."""

    def tracked(self):
        rig = GameRig(11)
        rig.look(35.0, 20.0)
        rig.offsets = {s: np.zeros(2) for s in rig.offsets}            # every other piece in the middle of its square
        rig.board = rig._render_board()
        mapping = ChessBoardMapping3D(rig, SimDetector(rig), SimPose(rig))
        self.assertEqual(mapping.calibrate().reason, "ok")
        return rig, mapping, MoveTracker(mapping.last_calibration.calibration)

    def on(self, rig, mapping, moved: dict[int, int]):
        physical = dict(rig.physical)
        for start, end in moved.items():
            physical[end] = physical.pop(start)
        rig.physical = physical
        rig.board = rig._render_board()
        rig.offsets = {s: np.zeros(2) for s in physical}
        return mapping.observe()

    def test_a_knight_beside_the_legal_squares_is_no_legal_move(self):
        rig, mapping, tracker = self.tracked()
        observation = self.on(rig, mapping, {chess.G1: chess.G3})      # not f3, not h3
        result = tracker.step(observation.frame, observation.calib)
        self.assertTrue(result.flagged, f"taken for {result.move}")
        self.assertEqual(tracker.board.ply(), 0)

    def test_a_piece_set_down_one_square_off_is_not_confirmed(self):
        rig, mapping, tracker = self.tracked()
        observation = self.on(rig, mapping, {chess.E2: chess.E5})
        ok, _ = tracker.verify(observation.frame, chess.Move.from_uci("e2e4"), observation.calib)
        self.assertFalse(ok)

    def test_a_piece_well_inside_its_square_is_fine(self):
        rig, mapping, tracker = self.tracked()
        rig.offsets = {s: np.zeros(2) for s in rig.physical}
        rig.offsets[chess.E2] = np.array([0.35, 0.0])
        observation = mapping.observe()
        result = tracker.step(observation.frame, observation.calib)
        self.assertFalse(result.flagged)
        self.assertIsNone(result.move)


class KnownPositionTest(unittest.TestCase):
    """The calibration knows which piece stands where: a detector that swaps bishops and knights changes nothing."""

    def calibrated(self, swaps: int):
        rig = GameRig(12)
        rig.look(35.0, 20.0)
        detector = SimDetector(rig)
        real = detector.detect
        names = {"white bishop": "white knight", "white knight": "white bishop",
                 "black bishop": "black knight", "black knight": "black bishop"}

        def swapped(crop, *, segment=False, threshold=None):
            found, left = real(crop, segment=segment, threshold=threshold), swaps
            for obj in found:
                if left and obj.label in names:
                    obj.label, left = names[obj.label], left - 1
            return found

        detector.detect = swapped
        return rig, ChessBoardMapping3D(rig, detector, SimPose(rig))

    def test_the_usual_swaps_are_no_refusal(self):
        _, mapping = self.calibrated(swaps=4)
        result = mapping.calibrate()
        self.assertEqual(result.reason, "ok")
        self.assertEqual(result.class_mismatches, 4)

    def test_the_set_is_measured_by_the_pieces_that_stand_there(self):
        """Most of them swapped: knights measured as bishops would make the knights the taller ones."""
        rig, mapping = self.calibrated(swaps=6)
        self.assertEqual(mapping.calibrate().reason, "ok")
        self.assertGreater(mapping.dims["b"][0], mapping.dims["n"][0])

    def test_many_wrong_types_are_still_another_position(self):
        _, mapping = self.calibrated(swaps=8)
        result = mapping.calibrate()
        self.assertIsNone(result.calibration)
        self.assertIn("another type", result.reason)


class LookViewsTest(unittest.TestCase):
    """What one view cannot settle is looked at from the next pose before anybody is asked. Off by default."""

    def test_every_look_but_the_first_goes_by_way_of_the_first(self):
        looks = ((1,), (2,), (3,))
        self.assertEqual(look_stops(looks, 0), ((1,),))
        self.assertEqual(look_stops(looks, 2), ((1,), (3,)))

    def test_one_view_never_leaves_the_first_pose(self):
        rig, game, _, _ = new_game(1, plies=4)
        game.run()
        self.assertEqual(set(rig.visited), {0})

    def test_a_refused_calibration_tries_the_next_pose_not_the_same_again(self):
        rig, game, _, human = new_game(2, plies=2, views=3)
        real, calls = game.mapping.calibrate, []

        def twice_refused(expected=None):
            calls.append(rig.visited[-1])
            return SimpleNamespace(calibration=None, reason="bad frame") if len(calls) < 3 else real(expected)

        game.mapping.calibrate = twice_refused
        game.choose_color = lambda: None
        game.calibrate()
        self.assertEqual(calls, [0, 1, 2])
        self.assertFalse(any("refused" in st for st in human.statuses))

    def test_the_cell_calibrates_the_same_way_for_the_test_programs(self):
        visited = []
        results = [SimpleNamespace(calibration=None, reason="bad frame"), SimpleNamespace(calibration=1, reason="ok")]
        cell = Cell(None, SimpleNamespace(calibrate=lambda expected: results.pop(0)), None,
                    lambda view=0: visited.append(view), {}, looks=2, views=2)
        self.assertEqual(cell.calibrate().reason, "ok")
        self.assertEqual(visited, [0, 1])

    def test_a_move_not_recognised_is_looked_at_again_before_the_human_is_asked(self):
        rig, game, _, human = new_game(3, plies=2, views=2)
        real, seen = MoveTracker.step, []

        def blind_from_the_first_pose(tracker, frame, calib=None):
            seen.append(rig.visited[-1])
            if rig.visited[-1] == 0:
                return TrackResult(None, 99.0, None, flagged=True)
            return real(tracker, frame, calib)

        with mock.patch.object(MoveTracker, "step", blind_from_the_first_pose):
            game.run()
        self.assertIn(1, seen)
        self.assertEqual(human.flags, 0, "the human was asked although the second pose saw the move")
        self.assertGreaterEqual(game.tracker.board.ply(), 1)

    def test_willys_move_is_confirmed_from_the_next_pose_before_anybody_is_asked(self):
        rig, game, _, human = new_game(4, plies=4, views=2)
        real, seen = MoveTracker.explains, []

        def blind_from_the_first_pose(tracker, frame, physical, calib=None):
            seen.append(rig.visited[-1])
            return (False, 99.0) if rig.visited[-1] == 0 else real(tracker, frame, physical, calib)

        with mock.patch.object(MoveTracker, "explains", blind_from_the_first_pose):
            game.run()
        self.assertIn(1, seen, "Willy's move was never looked at from the second pose")
        self.assertEqual(human.disagreements, 0)
        self.assertGreaterEqual(game.tracker.board.ply(), 2)

    def test_a_look_views_of_zero_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "c.yaml"
            path.write_text("game:\n  look_views: 0\n")
            with self.assertRaises(ValueError) as refused:
                load_config(path)
            self.assertIn("look_views", str(refused.exception))


class MappingTestProgramTest(unittest.TestCase):
    """The program the cell is judged with: a wrong type must not read as a wrong square."""

    def run_program(self, *args, swaps=4):
        rig = GameRig(13)
        detector = SimDetector(rig)
        real = detector.detect
        names = {"white bishop": "white knight", "white knight": "white bishop"}

        def swapped(crop, *, segment=False, threshold=None):
            found, left = real(crop, segment=segment, threshold=threshold), swaps
            for obj in found:
                if left and obj.label in names:
                    obj.label, left = names[obj.label], left - 1
            return found

        detector.detect = swapped
        mapping = ChessBoardMapping3D(rig, detector, SimPose(rig))
        visited = []

        def go_look(view=0):
            visited.append(view)
            rig.look(*LOOKS[view])

        @contextmanager
        def sim_cell(_tree, _config):
            yield Cell(None, mapping, None, go_look, {}, looks=3, views=1)

        with tempfile.TemporaryDirectory() as folder, mock.patch.object(mapping_test, "open_cell", sim_cell), \
                mock.patch.object(mapping_test, "load_cell_tree", lambda _: SimpleNamespace(ok=True)), \
                redirect_stdout(io.StringIO()) as out:
            code = mapping_test.main(["--out", folder, *args])
            files = sorted(p.name for p in Path(folder).iterdir())
            kept = dict(np.load(Path(folder) / "look1.npz")) if "look1.npz" in files else {}
        return code, out.getvalue(), files, kept, visited

    def test_a_bishop_seen_as_a_knight_stands_on_the_right_square(self):
        code, text, _, _, _ = self.run_program("--views", "2")
        self.assertEqual(code, 0)
        self.assertIn("right square: 64 of 64", text)
        self.assertRegex(text, r"right type:\s+56 of 64, (B seen as N|N seen as B) \d+x")

    def test_all_looks_visits_every_pose_and_keeps_what_it_saw(self):
        code, text, files, kept, visited = self.run_program("--all-looks", "--save", swaps=0)
        self.assertEqual(code, 0)
        self.assertEqual(sorted(set(visited)), [0, 1, 2])
        self.assertIn("the board as seen from here against look 0", text)
        self.assertIn("between the views", text)
        self.assertIn("free board height", text)
        self.assertEqual([f for f in files if f.endswith(".npz")], ["look0.npz", "look1.npz", "look2.npz"])
        self.assertEqual(kept["depth"].shape, kept["labels"].shape)


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
