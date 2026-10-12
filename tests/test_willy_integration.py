"""Willy_Chess against the real Workaholic-Willy at a desk: the dummy arm of the console_dummy profile, Willy's
calibration and keep-out types. Skipped where Willy is not installed.

    cd tests
    python -m unittest test_willy_integration -v
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

    def test_a_wrist_camera_is_placed_by_the_tool_pose_at_its_shutter(self):
        """WillyCameraPose with Willy's RigCalibration (eye_in_hand) and the dummy arm's TCP reader."""
        from src.calibration.rig_calibration import RigCalibration
        from src.geometry import Frame
        from src.geometry.transform import Transform

        from src.mapping.chessboard_mapping_3d import WillyCameraPose

        camera_to_tool = np.eye(4)
        camera_to_tool[:3, 3] = [0.0, 60.0, 40.0]
        calibration = RigCalibration("wrist", "eye_in_hand", "unused.json",
                                     Transform.from_matrix(camera_to_tool, from_frame=Frame.CAMERA, to_frame=Frame.TOOL),
                                     shutter_motion_tolerance_mm=1.0, shutter_motion_tolerance_deg=0.5)

        @dataclass
        class WristCamera:
            def calibration(self):
                return calibration

            def grab(self):
                return "frame"

        with self.robot.connected():
            self.robot.home(decline="Schach: Test am Schreibtisch")
            pose = WillyCameraPose(WristCamera(), tcp_reader=self.robot.arm.get_tcp_pose)
            frame, camera_to_base = pose.grab(WristCamera())
            tool_to_base = np.asarray(self.robot.arm.get_tcp_pose().to_matrix())
        self.assertEqual(frame, "frame")
        np.testing.assert_allclose(camera_to_base, tool_to_base @ camera_to_tool, atol=1e-6)


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


class WeightsCheckTest(unittest.TestCase):
    def test_a_git_lfs_pointer_is_refused(self):
        import tempfile

        from src.game.play import check_weights

        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "model.safetensors").write_text("version https://git-lfs.github.com/spec/v1\noid sha256:0\n")
            with self.assertRaises(SystemExit) as refused:
                check_weights(Path(folder))
            self.assertIn("git lfs pull", str(refused.exception))


if __name__ == "__main__":
    unittest.main()
