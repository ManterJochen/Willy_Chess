"""
ChessBoardMapping3D: the real board as the robot sees it, square by square, in its base frame.

One class from the camera frame to robot coordinates:
1. Grab a frame and where the camera stood (fixed camera or eye in hand, through Willy's
   shutter stamp), undistort colour and depth once, depth to metres.
2. Find the pieces only where the board is: the board volume projected into the image gives
   the crop for the detector (fewer pixels for RT-DETR and SAM2, more pixels per piece).
3. Calibrate once in the starting position (src/mapping/calibration.py) and lock the board pose
   in the robot base frame. It stays valid as long as nobody moves the board.
4. Every later view predicts the board from the lock and the camera pose, corrects the
   prediction with the visible corners (kinematics and hand-eye error), and checks the residual:
   a jump means the board was moved and has to be calibrated again.
5. Pieces to squares (src/mapping/pieces.py), and every square, piece and graveyard slot as a
   point in the robot base frame in millimetres, the unit of Willy's poses.
"""
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Protocol

import cv2 as cv
import numpy as np

from src.board.grid import detect_corners, detect_grid, gradient_magnitude, line_energy, to_gray
from src.mapping.calibration import CalibrationResult, calibrate_board, measure_grasp_widths
from src.mapping.frame import CLASS_TO_SYMBOL, BoardFrame, labels_from_masks
from src.mapping.geometry import BoardCalibration, BoardCamera, pose_matrix, solve_pnp, square_centres
from src.mapping.pieces import assign_squares, piece_positions


class Camera(Protocol):
    """What ChessBoardMapping3D needs from a camera (Willy's Camera fits)."""

    def grab(self) -> Any: ...                          # RGBDFrame: .color BGR uint8, .depth uint16 mm
    def get_intrinsics(self) -> np.ndarray | None: ...
    def get_distortion(self) -> np.ndarray | None: ...


class Detector(Protocol):
    """What ChessBoardMapping3D needs from a detector (Willy's ObjectDetector fits)."""

    def detect(self, image_bgr: np.ndarray, *, segment: bool = ..., threshold: float | None = ...) -> Iterable[Any]: ...


class CameraPose(Protocol):
    """Grabs a frame together with where the camera stood: CAMERA -> BASE, 4x4, millimetres."""

    def grab(self, camera: Camera) -> tuple[Any, np.ndarray]: ...


class StaticCameraPose:
    """A camera whose pose in the base frame is known and fixed (also for simulation and tests)."""

    def __init__(self, camera_to_base_mm: np.ndarray):
        self.camera_to_base_mm = np.asarray(camera_to_base_mm, np.float64)

    def grab(self, camera: Camera) -> tuple[Any, np.ndarray]:
        return camera.grab(), self.camera_to_base_mm


class WillyCameraPose:
    """
    Camera pose per frame through Workaholic-Willy, for a fixed camera and a camera on the wrist.

    A fixed camera (eye_to_hand) is placed by its calibration. A wrist camera (eye_in_hand) is
    placed by the tool pose read at its shutter; a frame taken while the arm moved beyond the
    rig's tolerance is grabbed again and refused if the arm never held still.

    Args:
        camera: An opened willy.Camera.
        tcp_reader (Callable | None): The arm's TCP reader (robot.arm.get_tcp_pose), needed for a wrist camera.
        attempts (int): Grabs after the first while the arm still moves.
    """

    def __init__(self, camera: Any, tcp_reader: Callable[[], Any] | None = None, attempts: int = 3):
        from src.robot.core.shutter_motion import ShutterStamp  # Willy, imported only when it is used

        self.calibration = camera.calibration()
        self.stamp = None
        if self.calibration.mounting_mode == "eye_in_hand":
            if tcp_reader is None:
                raise ValueError("a wrist camera needs the arm's TCP reader (robot.arm.get_tcp_pose)")
            tolerance = (self.calibration.shutter_motion_tolerance_mm or 1.0,
                         self.calibration.shutter_motion_tolerance_deg or 0.5)
            self.stamp = ShutterStamp.of(tcp_reader, motion_tolerance=tolerance, attempts=attempts)

    def grab(self, camera: Camera) -> tuple[Any, np.ndarray]:
        from src.robot.core.shutter_motion import camera_to_base_at_shutter

        if self.stamp is None:
            return camera.grab(), camera_to_base_at_shutter(self.calibration, None)
        stamped = self.stamp.grab(camera.grab)
        if stamped.tool_pose is None:
            raise RuntimeError("the arm moved during every grab; hold it still at the look pose")
        return stamped.frame, camera_to_base_at_shutter(self.calibration, stamped.tool_pose)


@dataclass
class BoardObservation:
    """
    One view of the board after calibration.

    Attributes:
        frame (BoardFrame): Undistorted image, depth and piece masks.
        calib (BoardCalibration): Board pose for this view (predicted from the lock, corrected by corners).
        camera_to_base (np.ndarray): CAMERA -> BASE of this view, 4x4, metres.
        board_to_base (np.ndarray): BOARD -> BASE as seen in this view, 4x4, metres.
        positions (dict[int, np.ndarray]): Piece label -> (x, y) on the board, metres, board frame
            (the estimate for squares, robust to hidden feet).
        grasp_positions (dict[int, np.ndarray]): Piece label -> (x, y) to grip at (the depth estimate,
            millimetre-accurate where the piece has depth).
        squares (dict[int, int]): Piece label -> square 0..63 (free mode, one image, no history).
        corners_used (int): Visible corners that corrected the predicted pose.
        corner_residual_px (float): Median reprojection error of those corners after the correction.
        pose_status (str): "corrected" (corners confirm the board, the lock follows it), "predicted"
            (too few corners visible, the lock alone) or "moved" (the board is not where it was
            locked: calibrate again before trusting any square).
        captured_at_s (float): Host time of the grab (time.time() clock).
    """
    frame: BoardFrame
    calib: BoardCalibration
    camera_to_base: np.ndarray
    board_to_base: np.ndarray
    positions: dict[int, np.ndarray]
    grasp_positions: dict[int, np.ndarray]
    squares: dict[int, int]
    corners_used: int
    corner_residual_px: float
    pose_status: str
    captured_at_s: float = 0.0

    @property
    def board_moved(self) -> bool:
        return self.pose_status == "moved"

    def occupancy(self) -> dict[int, str]:
        """Square -> FEN symbol of the piece seen there."""
        return {sq: self.frame.classes[k] for k, sq in self.squares.items()}

    def intruders(self, min_height: float = 0.015) -> int:
        """
        Pixels of something above the playing area that is no detected piece: a hand or an arm.

        Counted are depth points higher than min_height over the 8x8 squares and off every piece mask.
        Sensor noise and flying pixels come as single pixels and are opened away; a hand is a blob.
        A piece the detector missed counts as well, which errs on the safe side.
        """
        depth = self.frame.depth
        if depth is None:
            return 0
        cam = BoardCamera(self.frame.K, self.calib.R, self.calib.t)
        ys, xs = np.nonzero(depth > 0)
        P = cam.board_points_at(depth, xs, ys)
        above = (np.abs(P[:, :2]).max(axis=1) <= 4 * self.calib.square) & (P[:, 2] > min_height)
        blob = np.zeros(depth.shape, np.uint8)
        blob[ys[above], xs[above]] = 1
        pieces = cv.dilate((self.frame.labels > 0).astype(np.uint8), np.ones((9, 9), np.uint8))
        return int(cv.morphologyEx(blob & (1 - pieces), cv.MORPH_OPEN, np.ones((5, 5), np.uint8)).sum())

    def piece_points(self, label: int) -> np.ndarray:
        """
        Depth points of one piece in the board frame, (N, 3). Only points above the board and below
        1.3 x the tallest piece: flying pixels at the mask edge lie almost on the board.
        """
        if self.frame.depth is None:
            return np.empty((0, 3))
        cam = BoardCamera(self.frame.K, self.calib.R, self.calib.t)
        P = cam.board_points(self.frame.depth, self.frame.labels == label)
        top = 1.3 * max(self.calib.piece_dims(s)[0] for s in "pnbrqk")
        return P[(P[:, 2] > 0.002) & (P[:, 2] < top)]


class ChessBoardMapping3D:
    """
    The chessboard in the camera image and in the robot's base frame.

    Args:
        camera (Camera): Colour and aligned depth (willy.Camera).
        detector (Detector): The piece detector with segmentation (willy.ObjectDetector).
        camera_pose (CameraPose): Where the camera stood for each frame (WillyCameraPose).
        detector_threshold (float | None): Score a piece must clear; None keeps the detector's own.
        roi_margin_px (int): Margin around the projected board volume for the detector crop.
        max_corner_residual_px (float): Median corner residual above which the board counts as moved.
        max_correction_squares (float): Largest correction of the predicted board pose, in squares,
            before the board counts as moved (the kinematics and hand-eye budget). Corners are matched
            within the same radius; a board moved by a whole square matches the neighbouring corners
            and shows up as a correction of one square.
        grasp_height_fraction (dict[str, float] | None): Grasp height per piece type as a fraction of
            its height; the grasp width is measured there during calibration.
    """

    def __init__(self, camera: Camera, detector: Detector, camera_pose: CameraPose,
                 detector_threshold: float | None = None, roi_margin_px: int = 12,
                 max_corner_residual_px: float = 1.5, max_correction_squares: float = 0.35,
                 grasp_height_fraction: dict[str, float] | None = None):
        self.camera = camera
        self.detector = detector
        self.camera_pose = camera_pose
        self.detector_threshold = detector_threshold
        self.roi_margin_px = roi_margin_px
        self.max_corner_residual_px = max_corner_residual_px
        self.max_correction_squares = max_correction_squares
        self.grasp_height_fraction = grasp_height_fraction or {s: 0.45 for s in "pnbrqk"}

        self.board_to_base: np.ndarray | None = None        # 4x4, metres: the lock
        self.square: float | None = None
        self.dims: dict[str, tuple[float, float]] | None = None
        self.grasp_widths: dict[str, float] = {}
        self.last_calibration: CalibrationResult | None = None
        self._undistort: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None   # map1, map2, K_new

    # ------------------------------------------------------------------ calibration
    @property
    def is_calibrated(self) -> bool:
        return self.board_to_base is not None

    def calibrate(self, expected: dict[int, str] | None = None) -> CalibrationResult:
        """
        Calibrate from a known position and lock the board pose in the base frame.

        Args:
            expected (dict[int, str] | None): Square -> FEN symbol of the pieces that physically stand on
                the board; None is the starting position. In a running game: the tracker's physical pieces.

        Returns:
            CalibrationResult: With the calibration of this view, or the reason it was refused;
                on refusal an earlier lock stays as it was.
        """
        frame, camera_to_base, _ = self._grab()
        frame = self._with_pieces(frame, roi=None)
        result = calibrate_board(frame, expected)
        self.last_calibration = result
        if result.calibration is not None:
            calib = result.calibration
            self.board_to_base = camera_to_base @ pose_matrix(calib.R, calib.t)
            self.square, self.dims = calib.square, calib.dims
            self.grasp_widths = measure_grasp_widths(frame, calib, self.grasp_height_fraction)
        return result

    def grasp_width(self, symbol: str) -> float:
        """Width of a piece across the jaws at its grasp height, metres (measured, else 1.6 x foot radius)."""
        key = symbol.lower()
        if key in self.grasp_widths:
            return self.grasp_widths[key]
        return 1.6 * BoardCalibration(np.eye(3), np.zeros(3), self.square, self.dims).piece_dims(key)[1]

    def unlock(self) -> None:
        """Forget the board pose, e.g. after the board was moved on purpose."""
        self.board_to_base = None

    # ------------------------------------------------------------------ observation
    def observe(self) -> BoardObservation:
        """
        Grab a view and map every piece on the board to its square.

        Returns:
            BoardObservation: Pieces, squares and the pose of this view.
        """
        if not self.is_calibrated:
            raise RuntimeError("calibrate the board in the starting position first")
        frame, camera_to_base, captured_at_s = self._grab()
        predicted = self.view_calibration(camera_to_base, frame.K)
        # Pieces first, in the predicted board region (its margin covers the kinematics error): the pose
        # correction then ignores corners and edge samples that lie on a piece.
        frame = self._with_pieces(frame, roi=self._board_roi(frame, predicted, extra_squares=0.5))
        calib, used, residual, status = self._correct_pose(frame, predicted)
        board_to_base = camera_to_base @ pose_matrix(calib.R, calib.t)
        if status == "corrected":
            # Follow the board: every confirmed view is an absolute measurement, so nothing accumulates,
            # and a board nudged by a few millimetres is picked where it now stands.
            self.board_to_base = board_to_base
        positions = piece_positions(frame, calib)
        return BoardObservation(frame, calib, camera_to_base, board_to_base, positions,
                                piece_positions(frame, calib, for_grasp=True), assign_squares(positions, calib.square),
                                used, residual, status, captured_at_s)

    def view_calibration(self, camera_to_base: np.ndarray, K: np.ndarray | None = None) -> BoardCalibration:
        """The locked board seen from a camera pose: R, t of BOARD -> CAMERA, the square and the piece set."""
        board_to_camera = np.linalg.inv(camera_to_base) @ self.board_to_base
        return BoardCalibration(board_to_camera[:3, :3], board_to_camera[:3, 3], self.square, self.dims)

    # ------------------------------------------------------------------ robot coordinates
    def board_to_base_mm(self, xy: np.ndarray, z: float = 0.0) -> np.ndarray:
        """
        A point of the board frame in the robot base frame.

        Args:
            xy (np.ndarray): (x, y) on the board in metres, or (N, 2).
            z (float): Height above the board surface in metres.

        Returns:
            np.ndarray: (x, y, z) in millimetres in BASE, shape (3,) or (N, 3).
        """
        xy = np.atleast_2d(np.asarray(xy, np.float64))
        X = np.column_stack((xy, np.full(len(xy), z), np.ones(len(xy))))
        out = (X @ self.board_to_base.T)[:, :3] * 1000.0
        return out[0] if out.shape[0] == 1 else out

    def board_points_to_base_mm(self, P: np.ndarray) -> np.ndarray:
        """3D points of the board frame (N, 3), metres, in the robot base frame in millimetres."""
        X = np.column_stack((np.asarray(P, np.float64), np.ones(len(P))))
        return (X @ self.board_to_base.T)[:, :3] * 1000.0

    def square_center_base(self, square: int, z: float = 0.0) -> np.ndarray:
        """Centre of a square (python-chess numbering) at height z above the board, millimetres in BASE."""
        return self.board_to_base_mm(square_centres(self.square)[square], z)

    def board_axis_base(self, axis_xy: np.ndarray) -> np.ndarray:
        """A direction on the board (board frame) as a unit vector in the base frame's xy plane."""
        d = self.board_to_base[:3, :3] @ np.array([axis_xy[0], axis_xy[1], 0.0])
        return d[:2] / np.linalg.norm(d[:2])

    # ------------------------------------------------------------------ internals
    def _grab(self) -> tuple[BoardFrame, np.ndarray, float]:
        """Grab, place and undistort one frame: BoardFrame without pieces, CAMERA -> BASE in metres, grab time."""
        before = time.time()
        rgbd, camera_to_base_mm = self.camera_pose.grab(self.camera)
        captured_at_s = getattr(rgbd, "captured_at_s", None) or before
        color = np.asarray(rgbd.color)
        depth = np.asarray(rgbd.depth)
        depth_m = depth.astype(np.float32) / 1000.0 if depth.dtype == np.uint16 else depth.astype(np.float32)
        K = self.camera.get_intrinsics()
        if K is None:
            raise RuntimeError("the camera reports no intrinsics; the board needs a pinhole model")
        color, depth_m, K = self._undistorted(color, depth_m, np.asarray(K, np.float64))

        camera_to_base = np.asarray(camera_to_base_mm, np.float64).copy()
        camera_to_base[:3, 3] /= 1000.0
        return BoardFrame(color, K, depth_m if depth_m.size else None), camera_to_base, captured_at_s

    def _undistorted(self, color: np.ndarray, depth: np.ndarray, K: np.ndarray):
        """
        Remove lens distortion once, right after the grab: from here on the pinhole model holds
        everywhere, for the corners and the foot points alike. Depth is remapped with nearest
        neighbour, an interpolated depth at an edge would invent a point between two surfaces.
        """
        dist = self.camera.get_distortion()
        if dist is None or not np.any(np.asarray(dist)):
            return color, depth, K
        h, w = color.shape[:2]
        if self._undistort is None:
            K_new, _ = cv.getOptimalNewCameraMatrix(K, np.asarray(dist, np.float64), (w, h), 0)
            m1, m2 = cv.initUndistortRectifyMap(K, np.asarray(dist, np.float64), None, K_new, (w, h), cv.CV_32FC1)
            self._undistort = (m1, m2, K_new)
        m1, m2, K_new = self._undistort
        color = cv.remap(color, m1, m2, cv.INTER_LINEAR)
        if depth.size:
            depth = cv.remap(depth, m1, m2, cv.INTER_NEAREST)
        return color, depth, K_new

    def _with_pieces(self, frame: BoardFrame, roi: tuple[int, int, int, int] | None) -> BoardFrame:
        """Run the detector with segmentation (inside roi = x0, y0, x1, y1, if given) and add the masks."""
        h, w = frame.bgr.shape[:2]
        x0, y0, x1, y1 = roi if roi is not None else (0, 0, w, h)
        crop = np.ascontiguousarray(frame.bgr[y0:y1, x0:x1])
        masks, classes, scores = [], {}, {}
        for obj in self.detector.detect(crop, segment=True, threshold=self.detector_threshold):
            symbol = CLASS_TO_SYMBOL.get(obj.label.strip().lower())
            if symbol is None or obj.mask is None or not np.any(obj.mask):
                continue
            full = np.zeros((h, w), bool)
            full[y0:y1, x0:x1] = obj.mask
            masks.append(full)
            classes[len(masks)] = symbol
            scores[len(masks)] = float(obj.score)
        labels = labels_from_masks((h, w), masks)
        present = set(np.unique(labels)) - {0}
        return BoardFrame(frame.bgr, frame.K, frame.depth, labels,
                          {k: s for k, s in classes.items() if k in present},
                          {k: s for k, s in scores.items() if k in present})

    def _board_roi(self, frame: BoardFrame, calib: BoardCalibration,
                   extra_squares: float = 0.0) -> tuple[int, int, int, int]:
        """Bounding box of the board volume (squares plus a quarter square, up to the tallest piece)."""
        cam = BoardCamera(frame.K, calib.R, calib.t)
        e = (4.25 + extra_squares) * calib.square
        top = 1.3 * max(calib.piece_dims(s)[0] for s in "pnbrqk")
        X = np.array([[x, y, z] for x in (-e, e) for y in (-e, e) for z in (0.0, top)])
        uv = cam.project(X)
        h, w = frame.bgr.shape[:2]
        m = self.roi_margin_px
        x0, y0 = np.floor(uv.min(axis=0)).astype(int) - m
        x1, y1 = np.ceil(uv.max(axis=0)).astype(int) + m
        return max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)

    def _correct_pose(self, frame: BoardFrame, predicted: BoardCalibration):
        """
        Correct the predicted board pose with the visible inner corners.

        Detected corners within max_correction_squares of a predicted one are matched; with enough of
        them the pose is refined by Levenberg-Marquardt starting at the prediction. A large correction
        or residual means the board is no longer where it was locked. With too few matches the image
        decides: a board grid elsewhere in the image means it was moved, none means the corners are
        just hidden and the prediction stands.

        Returns:
            tuple: (calibration of this view, corners used, median residual px, pose status)
        """
        s = predicted.square
        ij = np.array([(i, j) for j in range(-3, 4) for i in range(-3, 4)], np.float64)
        cam = BoardCamera(frame.K, predicted.R, predicted.t)
        expected = cam.project(np.column_stack((ij * s, np.zeros(len(ij)))))
        detected = detect_corners(to_gray(frame.bgr))
        visible = self._board_visible(frame)
        if len(detected):
            px = np.rint(detected).astype(int)
            inside = (px[:, 0] >= 0) & (px[:, 1] >= 0) & (px[:, 0] < visible.shape[1]) & (px[:, 1] < visible.shape[0])
            detected = detected[inside & visible[np.clip(px[:, 1], 0, visible.shape[0] - 1),
                                                 np.clip(px[:, 0], 0, visible.shape[1] - 1)]]

        ok = np.zeros(len(ij), bool)
        nearest = np.zeros(len(ij), int)
        if len(detected):
            # the matching radius in pixels, from the local corner spacing of the prediction
            right = cam.project(np.column_stack(((ij + [1, 0]) * s, np.zeros(len(ij)))))
            up = cam.project(np.column_stack(((ij + [0, 1]) * s, np.zeros(len(ij)))))
            spacing = np.minimum(np.linalg.norm(right - expected, axis=1), np.linalg.norm(up - expected, axis=1))
            d = np.linalg.norm(expected[:, None] - detected[None], axis=2)
            nearest = d.argmin(axis=1)
            ok = d[np.arange(len(ij)), nearest] < self.max_correction_squares * spacing
        if ok.sum() < 6:
            grid = detect_grid(frame.bgr)
            status = "moved" if grid is not None and len(grid.indices) >= 12 else "predicted"
            return predicted, int(ok.sum()), float("nan"), status

        obj, img = ij[ok] * s, detected[nearest[ok]]
        R, t = solve_pnp(frame.K, obj, img, guess=(predicted.R, predicted.t))
        # second round without the matches that do not fit (saddles at piece edges near a hidden corner)
        res = self._residuals(frame.K, R, t, obj, img)
        keep = res < 2.0
        if 6 <= keep.sum() < len(res):
            obj, img = obj[keep], img[keep]
            R, t = solve_pnp(frame.K, obj, img, guess=(R, t))
            res = self._residuals(frame.K, R, t, obj, img)
        refined = BoardCalibration(R, t, s, predicted.dims)
        residual = float(np.median(res))
        # how far the board centre moved between prediction and correction, in squares
        shift = np.linalg.norm(predicted.R.T @ (t - predicted.t)) / s
        moved = (residual > self.max_corner_residual_px or shift > self.max_correction_squares
                 or self._window_shifted(frame, refined, visible))
        return refined, len(obj), residual, "moved" if moved else "corrected"

    @staticmethod
    def _residuals(K: np.ndarray, R: np.ndarray, t: np.ndarray, obj_xy: np.ndarray, pixels: np.ndarray) -> np.ndarray:
        proj = BoardCamera(K, R, t).project(np.column_stack((obj_xy, np.zeros(len(obj_xy)))))
        return np.linalg.norm(proj - pixels, axis=1)

    @staticmethod
    def _board_visible(frame: BoardFrame) -> np.ndarray:
        """False on the pieces, dilated by 3 px: their outlines are edges and saddles that belong to no corner."""
        if frame.labels is None:
            return np.ones(frame.bgr.shape[:2], bool)
        pieces = cv.dilate((frame.labels > 0).astype(np.uint8), np.ones((7, 7), np.uint8))
        return pieces == 0

    @staticmethod
    def _window_shifted(frame: BoardFrame, calib: BoardCalibration, visible: np.ndarray | None = None) -> bool:
        """
        A board moved by exactly one square puts every inner corner onto a neighbouring one, so the
        corners cannot see it; only the board edge breaks the symmetry. For each side compare the line
        one square beyond the board (+-5) with the opposite board edge (-+4): locked, the edge is a real
        edge and the line beyond lies in the frame; moved by one square towards that side, it is the
        other way round. (Simulated: locked at most 0.66, moved at least 2.0.) Samples on pieces are
        skipped, and a side whose lines are mostly hidden or outside the image is not judged.
        """
        s = calib.square
        H = frame.K @ np.column_stack((calib.R[:, 0] * s, calib.R[:, 1] * s, calib.t))
        grad = gradient_magnitude(to_gray(frame.bgr))
        for axis in (0, 1):
            for sign in (1, -1):
                beyond = line_energy(grad, H, axis, 5 * sign, visible)
                edge = line_energy(grad, H, axis, -4 * sign, visible)
                if beyond > edge:                                   # False whenever one of them is NaN
                    return True
        return False
