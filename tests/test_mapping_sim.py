"""End-to-end test of ChessBoardMapping3D on simulated scenes, without Willy and without a robot.

A fake camera renders bench.sim scenes the way willy.Camera delivers them (BGR, uint16 depth in mm),
a fake detector cuts the masks out of the scene's labels for whatever crop it is handed, like
willy.ObjectDetector with segment=True. The board stands at a known pose in the robot base frame;
the arm's camera pose comes with a few millimetres of kinematics and hand-eye error.

    cd tests
    python -m unittest test_mapping_sim -v
"""
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

import chess
import cv2 as cv
import numpy as np

TESTS = Path(__file__).resolve().parent
for p in (TESTS, TESTS.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from bench import sim  # noqa: E402
from src.mapping.chessboard_mapping_3d import ChessBoardMapping3D  # noqa: E402
from src.mapping.frame import SYMBOL_TO_CLASS  # noqa: E402
from src.mapping.geometry import pose_matrix  # noqa: E402

# The board in the robot base frame: 45 cm in front of the robot, turned by 25 degrees, 2 cm up.
_a = np.deg2rad(25.0)
BOARD_TO_BASE = pose_matrix(np.array([[np.cos(_a), -np.sin(_a), 0], [np.sin(_a), np.cos(_a), 0], [0, 0, 1]]),
                            np.array([0.45, 0.05, 0.02]))


@dataclass
class RGBD:
    color: np.ndarray
    depth: np.ndarray


@dataclass
class Detection:
    label: str
    score: float
    mask: np.ndarray


class SimRig:
    """Fake willy.Camera + camera pose: renders the current board from the current look pose."""

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.square = self.rng.uniform(0.045, 0.065)
        self.board = chess.Board()
        self.offsets = sim.jitter_offsets(self.rng, self.board)
        self.board_shift = np.zeros(3)          # the board pushed away from where it was locked
        self.pose_error_mm = 0.0                # hand-eye error of the reported camera pose (fixed per rig)
        self.hand: tuple | None = None          # ((x0, y0, x1, y1) pixels, height above the board in m)
        self._pose_error = (self.rng.normal(0, 1, 3), cv.Rodrigues(self.rng.normal(0, np.deg2rad(0.1), 3))[0])
        self.look(40.0, 30.0)
        self.last = None

    def look(self, tilt: float, azim: float) -> None:
        self.tilt, self.azim = tilt, azim

    # willy.Camera
    def grab(self):
        R, t, C = sim.look_at(self.tilt, self.azim)
        setup = sim.Setup(R, t, C, square=self.square)
        frame, truth = sim.render_pieces(self.rng, setup, self.board, self.offsets)
        if self.hand is not None:
            self._add_hand(frame, setup)
        self.last = (frame, truth, setup)
        depth_mm = np.round(frame.depth * 1000.0).astype(np.uint16)
        return RGBD(frame.bgr, depth_mm)

    def _add_hand(self, frame, setup) -> None:
        """A flat hand at a height over the board: nearer than what it covers, in depth, colour and masks."""
        (x0, y0, x1, y1), height = self.hand
        vs, us = np.mgrid[y0:y1, x0:x1]
        rays = np.stack([us, vs, np.ones_like(us)], -1).reshape(-1, 3) @ np.linalg.inv(sim.K).T
        depth = ((height - setup.C[2]) / (rays @ setup.R)[:, 2]).reshape(us.shape).astype(np.float32)
        region = frame.depth[y0:y1, x0:x1]
        nearer = depth < np.where(region > 0, region, np.inf)
        region[nearer] = depth[nearer]
        frame.bgr[y0:y1, x0:x1][nearer] = (120, 150, 200)
        frame.labels[y0:y1, x0:x1][nearer] = 0

    def get_intrinsics(self):
        return sim.K.copy()

    def get_distortion(self):
        return None

    # CameraPose
    def grab_with_pose(self, camera):
        rgbd = self.grab()
        _, _, setup = self.last
        board_to_camera = pose_matrix(setup.R, setup.t)
        true_board = BOARD_TO_BASE.copy()
        true_board[:3, 3] += self.board_shift
        camera_to_base = true_board @ np.linalg.inv(board_to_camera)
        camera_to_base[:3, 3] *= 1000.0
        if self.pose_error_mm:
            # a hand-eye error is the same in every view, the arm's repeatability adds a little jitter
            shift, rot = self._pose_error
            camera_to_base[:3, 3] += shift * self.pose_error_mm + self.rng.normal(0, 0.05, 3)
            camera_to_base[:3, :3] = camera_to_base[:3, :3] @ rot
        return rgbd, camera_to_base


class SimPose:
    def __init__(self, rig: SimRig):
        self.rig = rig

    def grab(self, camera):
        return self.rig.grab_with_pose(camera)


class SimDetector:
    """Masks and classes of the scene for the crop it is handed (found by exact template match)."""

    def __init__(self, rig: SimRig):
        self.rig = rig

    def detect(self, image_bgr, *, segment=False, threshold=None):
        frame, _, _ = self.rig.last
        full = cv.cvtColor(frame.bgr, cv.COLOR_BGR2GRAY)
        crop = cv.cvtColor(image_bgr, cv.COLOR_BGR2GRAY)
        _, _, (x0, y0), _ = cv.minMaxLoc(cv.matchTemplate(full, crop, cv.TM_SQDIFF))
        h, w = crop.shape
        out = []
        for k, symbol in frame.classes.items():
            mask = frame.labels[y0:y0 + h, x0:x0 + w] == k
            if mask.any():
                out.append(Detection(SYMBOL_TO_CLASS[symbol], 0.9, mask))
        return out


def make_mapping(rig: SimRig) -> ChessBoardMapping3D:
    return ChessBoardMapping3D(rig, SimDetector(rig), SimPose(rig))


def truth_squares(rig: SimRig) -> dict[int, int]:
    _, truth, _ = rig.last
    return truth.square_of


class ChessBoardMapping3DTest(unittest.TestCase):
    def test_calibration_locks_the_board_in_the_base_frame(self):
        for seed in range(3):
            rig = SimRig(seed)
            mapping = make_mapping(rig)
            result = mapping.calibrate()
            self.assertEqual(result.reason, "ok", f"seed {seed}")
            self.assertLess(abs(mapping.square / rig.square - 1), 0.01)
            err_mm = np.linalg.norm(mapping.board_to_base[:3, 3] - BOARD_TO_BASE[:3, 3]) * 1000
            self.assertLess(err_mm, 3.0, f"seed {seed}: board origin {err_mm:.1f} mm off")
            a1 = mapping.square_center_base(chess.A1)
            a1_true = (BOARD_TO_BASE @ np.r_[-3.5 * rig.square, -3.5 * rig.square, 0, 1])[:3] * 1000
            self.assertLess(np.linalg.norm(a1 - a1_true), 3.0)

    def test_observe_from_other_look_poses_with_pose_error(self):
        rig = SimRig(7)
        mapping = make_mapping(rig)
        self.assertEqual(mapping.calibrate().reason, "ok")
        rig.board = sim.random_position(rig.rng, 30)
        rig.offsets = sim.jitter_offsets(rig.rng, rig.board)
        rig.pose_error_mm = 2.0
        right = total = 0
        for tilt, azim in ((30.0, 10.0), (45.0, 80.0), (35.0, -60.0)):
            rig.look(tilt, azim)
            obs = mapping.observe()
            self.assertFalse(obs.board_moved, f"look {tilt}/{azim}: residual {obs.corner_residual_px}")
            self.assertGreater(obs.corners_used, 6)
            want = truth_squares(rig)
            right += sum(obs.squares.get(k) == s for k, s in want.items())
            total += len(want)
        self.assertGreaterEqual(right / total, 0.99, f"{right}/{total} pieces on the right square")

    def test_a_moved_board_is_reported(self):
        rig = SimRig(11)
        mapping = make_mapping(rig)
        self.assertEqual(mapping.calibrate().reason, "ok")
        rig.board_shift = np.array([0.03, -0.01, 0.0])
        self.assertTrue(mapping.observe().board_moved)

    def test_a_board_moved_by_exactly_one_square_is_reported(self):
        # every corner then lies on a neighbouring corner of the prediction
        rig = SimRig(12)
        mapping = make_mapping(rig)
        self.assertEqual(mapping.calibrate().reason, "ok")
        rig.board_shift = BOARD_TO_BASE[:3, :3] @ np.array([rig.square, 0.0, 0.0])
        self.assertTrue(mapping.observe().board_moved)

    def test_a_nudged_board_is_followed(self):
        rig = SimRig(13)
        mapping = make_mapping(rig)
        self.assertEqual(mapping.calibrate().reason, "ok")
        rig.board_shift = np.array([0.005, 0.003, 0.0])
        obs = mapping.observe()
        self.assertEqual(obs.pose_status, "corrected")
        want = (BOARD_TO_BASE[:3, 3] + rig.board_shift) * 1000
        self.assertLess(np.linalg.norm(mapping.board_to_base[:3, 3] * 1000 - want), 3.0)

    def test_a_hand_over_the_board_is_seen(self):
        rig = SimRig(17)
        mapping = make_mapping(rig)
        self.assertEqual(mapping.calibrate().reason, "ok")
        rig.board = sim.random_position(rig.rng, 20)
        rig.offsets = sim.jitter_offsets(rig.rng, rig.board)
        self.assertEqual(mapping.observe().intruders(), 0)
        rig.hand = ((260, 180, 380, 260), 0.08)
        self.assertGreater(mapping.observe().intruders(), 3000)

    def test_detector_only_sees_the_board_region(self):
        rig = SimRig(3)
        mapping = make_mapping(rig)
        self.assertEqual(mapping.calibrate().reason, "ok")
        seen = []
        detector = mapping.detector
        original = detector.detect

        def spy(image_bgr, **kw):
            seen.append(image_bgr.shape[:2])
            return original(image_bgr, **kw)

        detector.detect = spy
        mapping.observe()
        h, w = seen[-1]
        self.assertLess(h * w, 640 * 480)


if __name__ == "__main__":
    unittest.main()
