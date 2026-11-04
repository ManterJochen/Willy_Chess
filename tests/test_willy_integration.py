"""Willy_Chess against the real Workaholic-Willy at a desk: the dummy arm of the console_dummy profile, Willy's
calibration and keep-out types. Skipped where Willy is not installed.

    cd tests
    python -m unittest test_willy_integration -v
"""
import sys
import unittest
from pathlib import Path

import chess
import numpy as np

TESTS = Path(__file__).resolve().parent
for p in (TESTS, TESTS.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    import willy  # noqa: F401
except ImportError:
    willy = None


@unittest.skipIf(willy is None, "Workaholic-Willy is not installed")
class WillyIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from willy import Robot, load_tree

        from src.game.config import ChessConfig
        from src.mapping.chessboard_mapping_3d import ChessBoardMapping3D
        from test_game_sim import GameRig
        from test_mapping_sim import SimDetector, SimPose

        cls.tree = load_tree("console_dummy")
        cls.robot = Robot.from_tree(cls.tree)
        cls.config = ChessConfig()
        cls.rig = GameRig(5)
        cls.rig.look(35.0, 20.0)
        cls.mapping = ChessBoardMapping3D(cls.rig, SimDetector(cls.rig), SimPose(cls.rig))
        assert cls.mapping.calibrate().reason == "ok"

    def test_a_capture_runs_on_willys_arm(self):
        """Every pick and place of a planned capture is accepted by Willy's robot (no camera world: declined)."""
        from src.game.robot_moves import RobotMover, plan_transfers

        board = chess.Board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
        self.rig.set_physical({s: p.symbol() for s, p in board.piece_map().items()})
        observation = self.mapping.observe()
        transfers = plan_transfers(board, self.rig.physical, chess.Move.from_uci("e4d5"), {},
                                   observation.calib.square, 0, self.config.robot.graveyard)
        mover = RobotMover(self.robot, self.mapping, self.config.robot, camera_world=False)
        with self.robot.connected():
            execution = mover.execute(transfers, observation)
        self.assertTrue(execution.ok, execution.message)
        self.assertEqual(len(execution.done), 2)                      # the pawn to the graveyard, then e4 -> d5
        self.assertTrue(all(report.ok for report in execution.reports))

    def test_the_keep_out_offer_passes_willys_validation(self):
        """With a camera world the piece to grip goes to Willy as a SegmentationOffer: (N, 3) BASE millimetres."""
        from src.game.robot_moves import RobotMover, Transfer
        from src.robot.core.keep_out import SegmentationOffer

        self.rig.set_physical({s: p.symbol() for s, p in chess.Board().piece_map().items()})
        observation = self.mapping.observe()
        mover = RobotMover(self.robot, self.mapping, self.config.robot, camera_world=True)
        sq = chess.G1
        offer = mover._keep_out(Transfer("N", np.zeros(2), np.zeros(2), from_square=sq), observation)["keep_out"]
        self.assertIsInstance(offer, SegmentationOffer)
        self.assertEqual(offer.target_points_base_mm.shape[1], 3)
        self.assertGreater(len(offer.named_points_base_mm), 20)
        centre = self.mapping.square_center_base(sq)
        self.assertLess(np.linalg.norm(np.median(offer.target_points_base_mm, axis=0)[:2] - centre[:2]), 0.6 * 1000 * self.mapping.square)

    def test_a_frame_comes_through_willys_capture_as_the_median_of_a_burst(self):
        """WillyCameraPose through willy.capture (a fixed camera, placed by its calibration): the depth in metres,
        the per-pixel median of the burst, so a flying pixel that a third of the grabs read is gone."""
        from src.calibration.rig_calibration import RigCalibration
        from src.camera.setup.image_taking.frames import RGBDFrame
        from src.geometry import Frame
        from src.geometry.transform import Transform

        from src.mapping.chessboard_mapping_3d import WillyCameraPose

        camera_to_base = np.eye(4)
        camera_to_base[:3, 3] = [400.0, 0.0, 750.0]
        calibration = RigCalibration("overhead", "eye_to_hand", "unused.json",
                                     Transform.from_matrix(camera_to_base, from_frame=Frame.CAMERA, to_frame=Frame.BASE))

        class Handle:
            rig_id, grabs = "overhead", 0

            def grab(self):
                self.grabs += 1
                depth = np.full((48, 64), 700, np.uint16)
                depth[10, 10] = 900 if self.grabs % 3 == 0 else 700
                return RGBDFrame(color=np.zeros((48, 64, 3), np.uint8), depth=depth, captured_at_s=1.0)

            def get_intrinsics(self):
                return np.array([[60.0, 0.0, 32.0], [0.0, 60.0, 24.0], [0.0, 0.0, 1.0]])

        class FixedCamera:
            rig_id, source, the_handle = "overhead", "rgbd", Handle()

            def handle(self):
                return self.the_handle

            def calibration(self):
                return calibration

        pose = WillyCameraPose(FixedCamera())
        frame, placed = pose.grab(FixedCamera())
        self.assertFalse(pose.on_wrist)
        self.assertGreaterEqual(FixedCamera.the_handle.grabs, 7)                 # a burst, not one grab
        np.testing.assert_allclose(frame.depth, 0.7)
        np.testing.assert_allclose(placed, camera_to_base)


@unittest.skipIf(willy is None, "Workaholic-Willy is not installed")
class CellTreeTest(unittest.TestCase):
    def test_a_willy_data_directory_loads_without_a_profile(self):
        from src.game.play import load_cell_tree

        self.assertTrue(load_cell_tree("external/Workaholic-Willy/config").ok)

    def test_a_missing_data_directory_is_refused(self):
        from src.game.play import load_cell_tree

        for data_dir in (None, "no/such/dir"):
            with self.assertRaises(SystemExit) as refused:
                load_cell_tree(data_dir)
            self.assertIn("Willy data directory", str(refused.exception))

    def test_the_keys_the_game_reads_are_in_willys_tree(self):
        """What open_cell and JawGeometry read of a tree: a key Willy renames fails here, not at the cell."""
        from src.game.play import load_cell_tree
        from src.game.robot_moves import JawGeometry

        tree = load_cell_tree("external/Workaholic-Willy/config")
        jaw = JawGeometry.from_tree(tree)
        self.assertGreater(jaw.finger_thickness, 0.0)
        self.assertAlmostEqual(jaw.max_open, tree.robot.gripper.max_width_mm / 1000.0)
        self.assertTrue(hasattr(tree.robot.motion, "look_deg"))
        self.assertTrue(tree.app_config.camera.cameras.primary_rig_id)


@unittest.skipIf(willy is None, "Workaholic-Willy is not installed")
class LookPoseTest(unittest.TestCase):
    LOOKS = ((0.0, -90.0, 90.0, -90.0, -90.0, 0.0), (10.0, -90.0, 90.0, -90.0, -90.0, 0.0))

    def test_every_look_is_accepted_by_willys_arm(self):
        from willy import Robot, load_tree

        from src.game.play import go_to_look
        from src.game.robot_moves import DECLINE

        robot = Robot.from_tree(load_tree("console_dummy"))
        with robot.connected():
            go_to_look(robot, self.LOOKS, 1, {"decline": DECLINE})
            go_to_look(robot, self.LOOKS, 0, {"decline": DECLINE})
            go_to_look(robot, (), 0, {"decline": DECLINE})             # a fixed camera: home

    def test_no_further_than_the_first_stop_it_did_not_reach(self):
        from src.game.play import go_to_look

        commanded = []

        class Stuck:
            def move_joints(self, joints, **kw):
                commanded.append(joints)
                return type("Report", (), {"ok": False})()

        with self.assertRaises(RuntimeError):
            go_to_look(Stuck(), self.LOOKS, 1, {})
        self.assertEqual(len(commanded), 1)


@unittest.skipIf(willy is None, "Workaholic-Willy is not installed")
class TouchTestRunTest(unittest.TestCase):
    def test_the_touch_test_reaches_every_square_and_graveyard_slot_on_willys_arm(self):
        """Every point lifted over and come down to through Willy's move, the graveyard's corners among them."""
        from contextlib import ExitStack, contextmanager
        from unittest import mock

        from willy import Robot, load_tree

        from src.game import touch_test
        from src.game.config import ChessConfig
        from src.game.play import Cell
        from src.game.robot_moves import DECLINE, RobotMover
        from src.mapping.chessboard_mapping_3d import ChessBoardMapping3D
        from test_game_sim import GameRig
        from test_mapping_sim import SimDetector, SimPose

        tree = load_tree("console_dummy")
        robot = Robot.from_tree(tree)
        rig = GameRig(5)
        rig.look(35.0, 20.0)
        mapping = ChessBoardMapping3D(rig, SimDetector(rig), SimPose(rig))
        mover = RobotMover(robot, mapping, ChessConfig().robot, camera_world=False)

        @contextmanager
        def dummy_cell(_tree, _config):
            with ExitStack() as stack:
                stack.enter_context(robot.connected())
                yield Cell(robot, mapping, mover, lambda: None, {"decline": DECLINE})

        points = []
        with mock.patch.object(touch_test, "open_cell", dummy_cell), mock.patch.object(touch_test, "load_cell_tree", lambda _: tree), \
                mock.patch("builtins.input", lambda text: points.append(text) or ""):
            self.assertEqual(touch_test.main(["--height-mm", "30"]), 0)
            self.assertEqual(len(points), len(touch_test.SQUARES) + len(touch_test.GRAVEYARD_SLOTS))
            self.assertTrue(any("graveyard slot 15" in text for text in points))
            points.clear()
            self.assertEqual(touch_test.main(["--height-mm", "30", "--corners"]), 0)
            self.assertEqual(len(points),
                             len(touch_test.SQUARES) + len(touch_test.CORNERS) + len(touch_test.GRAVEYARD_SLOTS))
            self.assertTrue(any("a8 (take the rook off first)" in text for text in points))


@unittest.skipIf(willy is None, "Workaholic-Willy is not installed")
class ExportedDatasetTest(unittest.TestCase):
    def test_willys_training_reads_what_the_export_writes(self):
        """Classes, splits and boxes as DetectorTraining will load them, from two logged games."""
        import tempfile

        from src.mapping.frame import CLASSES
        from src.models.detection.closed_set.training.datasets import load_dataset
        from src.pieces.export import export
        from test_game_log import played

        with tempfile.TemporaryDirectory() as folder:
            for seed in (21, 22):
                played(Path(folder, "logs"), seed, plies=2)
            report = export(Path(folder, "logs"), Path(folder, "dataset"))
            dataset = load_dataset(Path(folder, "dataset"), format="yolo", classes=CLASSES)
        self.assertEqual(tuple(dataset.classes), CLASSES)
        self.assertEqual(dataset.val_source, "declared")
        self.assertEqual(len(dataset.train), report["train"])
        self.assertEqual(len(dataset.val), report["val"])
        self.assertGreater(sum(len(image.boxes) for image in dataset.train), 20 * report["train"])


class WeightsCheckTest(unittest.TestCase):
    def test_a_git_lfs_pointer_is_refused(self):
        import tempfile

        from src.game.play import check_weights

        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "model.safetensors").write_text("version https://git-lfs.github.com/spec/v1\noid sha256:0\n")
            with self.assertRaises(SystemExit) as refused:
                check_weights(Path(folder))
            self.assertIn("git lfs pull", str(refused.exception))


@unittest.skipIf(willy is None, "Workaholic-Willy is not installed")
class DeskRunTest(unittest.TestCase):
    def test_a_whole_game_runs_at_the_desk(self):
        """desk_run: play_games with Stockfish, Willy's dummy arm and the Pygame view, up to the saved PGN."""
        import tempfile

        import desk_run
        from src.game.engine import find_stockfish

        try:
            find_stockfish()
        except FileNotFoundError:
            self.skipTest("no Stockfish on this machine")
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(desk_run.main(["--headless", "--seed", "2", "--pgn-dir", folder]), 0)
            self.assertEqual(len(list(Path(folder).glob("*.pgn"))), 1)


if __name__ == "__main__":
    unittest.main()
