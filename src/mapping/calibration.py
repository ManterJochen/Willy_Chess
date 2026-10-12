"""
Calibrate the board from a known position: pose, orientation, square size and piece set.

Normally the starting position; in a running game the position the tracker knows (after the
board was moved), as long as it is not symmetric enough to fit two orientations.

The grid from src/board is labelled only up to the 8 symmetries of the square plus a shift,
and each needs its own evidence:
1. Mirror: image coordinates are left-handed. The camera has to be above the board, C_z > 0;
   otherwise the x index is mirrored and the pose solved again.
2. Rotation and window: one search over 4 rotations x 25 shifts. The labelling under which the
   most visible pieces stand on an expected square of their colour (and their type) wins.
3. Cross-check a1 is dark: the median brightness of the free pixels per square against (-1)^(f+r).
4. Scale: PnP runs in square units, so the camera height comes out in squares; a plane through
   the free board pixels gives it in metres. Their ratio is the square size, no board size needed.
5. Piece set: height and foot radius per type from the depth.
Fail closed: if any check disagrees, there is no calibration and the reason says why.
"""
from dataclasses import dataclass

import chess
import cv2 as cv
import numpy as np

from src.board.grid import detect_grid, to_gray
from src.mapping.frame import BoardFrame
from src.mapping.geometry import BoardCalibration, BoardCamera, fit_plane, reprojection_rms, solve_pnp
from src.mapping.pieces import assign, foot_xy

ROT90 = np.array([[0, -1], [1, 0]])


@dataclass
class CalibrationResult:
    """
    Outcome of a calibration attempt.

    Attributes:
        calibration (BoardCalibration | None): The calibration, None if refused.
        reason (str): "ok", or why the calibration was refused.
        corners (int): Inlier corners the pose rests on.
        reprojection_px (float): RMS reprojection error of those corners.
        normal_deg (float): Angle between the depth plane normal and the board z axis of the pose.
        class_mismatches (int): Pieces on a start square of their colour but of another type.
    """
    calibration: BoardCalibration | None
    reason: str
    corners: int = 0
    reprojection_px: float = float("nan")
    normal_deg: float = float("nan")
    class_mismatches: int = 0


def calibrate_board(frame: BoardFrame, expected: dict[int, str] | None = None, min_corners: int = 12,
                    max_normal_deg: float = 3.0, slack: int = 3, margin: int = 6,
                    max_class_mismatches: int = 2) -> CalibrationResult:
    """
    Calibrate from one view of a known position.

    Args:
        frame (BoardFrame): Image, depth and piece masks.
        expected (dict[int, str] | None): Square -> FEN symbol of the pieces on the board (physical
            pieces); None is the starting position.
        min_corners (int): Fewest inlier corners for the pose.
        max_normal_deg (float): Largest angle between depth plane and pose.
        slack (int): Pieces whose foot may miss its start square (hidden feet).
        margin (int): Lead of the best labelling over the second best (colour + type matches).
        max_class_mismatches (int): Pieces the detector may classify wrongly in the final check.

    Returns:
        CalibrationResult: The calibration, or the reason it was refused.
    """
    grid = detect_grid(frame.bgr)
    if grid is None:
        return CalibrationResult(None, "no board grid found")
    pts, idx = grid.points, grid.indices
    if len(idx) < min_corners:
        return CalibrationResult(None, f"only {len(idx)} corners visible, {min_corners} needed", len(idx))
    if frame.labels is None or not frame.classes:
        return CalibrationResult(None, "no pieces detected, the starting position is needed", len(idx))

    R, t = solve_pnp(frame.K, idx, pts)
    if (-R.T @ t)[2] < 0:                                    # camera below the board: mirrored labelling
        idx = idx * np.array([-1, 1])
        R, t = solve_pnp(frame.K, idx, pts)

    keys = [k for k in frame.classes if (frame.labels == k).any()]
    cam = BoardCamera(frame.K, R, t)
    feet = np.array([foot_xy(cam, frame.labels == k, 0.25) for k in keys]).reshape(-1, 2)
    symbols = [frame.classes[k] for k in keys]

    if expected is None:
        expected = {s: p.symbol() for s, p in chess.Board().piece_map().items()}
    ranked = _rank_labellings(feet, symbols, idx, expected)
    best, second = ranked[0], ranked[1]
    colour_hits = best[1]
    if colour_hits < len(symbols) - slack:
        return CalibrationResult(None, f"only {colour_hits} of {len(symbols)} pieces stand on an expected square "
                                       f"of their colour: not the expected position?", len(idx))
    if best[0] - second[0] < margin:
        return CalibrationResult(None, "two orientations of the board fit about equally well", len(idx))

    M = np.linalg.matrix_power(ROT90, best[3])
    idx = idx @ M.T - np.array(best[4:6])
    R, t = solve_pnp(frame.K, idx, pts)
    rms = reprojection_rms(frame.K, R, t, idx, pts)
    if _parity(frame, R, t) > 0:
        return CalibrationResult(None, "a1 is not dark under the chosen orientation", len(idx), rms)

    square, normal_deg = _square_from_depth(frame, R, t)
    if square is None:
        return CalibrationResult(None, "too few free board pixels with depth for the scale", len(idx), rms)
    if normal_deg > max_normal_deg:
        return CalibrationResult(None, f"depth plane and pose disagree by {normal_deg:.1f} degrees",
                                 len(idx), rms, normal_deg)

    t = square * t
    calib = BoardCalibration(R, t, float(square), _measure_dims(frame, BoardCamera(frame.K, R, t)))

    mismatches = 0
    for k, sq in assign(frame, calib).items():
        want = expected.get(sq)
        symbol = frame.classes[k]
        if want is None or want.isupper() != symbol.isupper():
            return CalibrationResult(None, f"a {symbol} lands on {chess.square_name(sq)}, where no piece of its "
                                           f"colour is expected", len(idx), rms, normal_deg)
        mismatches += want != symbol
    if mismatches > max_class_mismatches:
        return CalibrationResult(None, f"{mismatches} pieces stand on a square expected for another type",
                                 len(idx), rms, normal_deg, mismatches)
    return CalibrationResult(calib, "ok", len(idx), rms, normal_deg, mismatches)


def calibrate(frame: BoardFrame) -> BoardCalibration | None:
    """The calibration of calibrate_board, or None if it was refused."""
    return calibrate_board(frame).calibration


def _rank_labellings(feet: np.ndarray, symbols: list[str], idx: np.ndarray, expected: dict[int, str]) -> list[tuple]:
    """
    Score all 4 rotations x 25 shifts of the grid labelling against the expected position.

    Returns:
        list[tuple]: (colour + type hits, colour hits, inlier corners in the window, rotation, dx, dy),
            best first.
    """
    ranked = []
    for rot in range(4):
        M = np.linalg.matrix_power(ROT90, rot)
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                shift = np.array([dx, dy])
                squares = np.floor(feet @ M.T - shift + 4).astype(int)
                colour = kind = 0
                for (f, r), symbol in zip(squares, symbols):
                    if 0 <= f < 8 and 0 <= r < 8 and (want := expected.get(chess.square(f, r))) is not None:
                        if want.isupper() == symbol.isupper():
                            colour += 1
                            kind += want == symbol
                inside = int(np.sum(np.abs(idx @ M.T - shift).max(axis=1) <= 3))
                ranked.append((colour + kind, colour, inside, rot, dx, dy))
    ranked.sort(reverse=True)
    return ranked


def _parity(frame: BoardFrame, R: np.ndarray, t_sq: np.ndarray) -> float:
    """> 0 if the squares with even file + rank are light, i.e. a1 is not dark."""
    cam = BoardCamera(frame.K, R, t_sq)
    gray = to_gray(frame.bgr).astype(np.float32)
    h, w = gray.shape
    offsets = np.array([(a, b) for a in np.linspace(-0.3, 0.3, 5) for b in np.linspace(-0.3, 0.3, 5)])
    medians, signs = [], []
    for f in range(8):
        for r in range(8):
            c = np.array([f - 3.5, r - 3.5]) + offsets
            uv = np.rint(cam.project(np.column_stack((c, np.zeros(len(c)))))).astype(int)
            uv = uv[(uv[:, 0] >= 0) & (uv[:, 1] >= 0) & (uv[:, 0] < w) & (uv[:, 1] < h)]
            free = frame.labels[uv[:, 1], uv[:, 0]] == 0
            if free.sum() >= 3:
                medians.append(np.median(gray[uv[free, 1], uv[free, 0]]))
                signs.append(1 if (f + r) % 2 == 0 else -1)
    medians = np.array(medians)
    return float(np.sum(np.array(signs) * (medians - np.median(medians)))) if len(medians) else 0.0


def _square_from_depth(frame: BoardFrame, R: np.ndarray, t_sq: np.ndarray) -> tuple[float | None, float]:
    """
    Square size = camera height in metres (plane through the free board pixels) / camera height in squares.

    Returns:
        tuple[float | None, float]: Square size in metres (None with too little depth) and the angle
            in degrees between the depth plane normal and the board z axis of the pose.
    """
    if frame.depth is None:
        return None, 90.0
    cam = BoardCamera(frame.K, R, t_sq)
    border = cam.project(np.array([[-4, -4, 0], [4, -4, 0], [4, 4, 0], [-4, 4, 0]], float))
    area = np.zeros(frame.labels.shape, np.uint8)
    cv.fillPoly(area, [np.round(border).astype(np.int32)], 1)
    free = cv.erode(((area > 0) & (frame.labels == 0)).astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    ys, xs = np.nonzero(free & (frame.depth > 0))
    if len(xs) < 500:
        return None, 90.0
    step = max(1, len(xs) // 6000)
    ys, xs = ys[::step], xs[::step]
    z = frame.depth[ys, xs].astype(np.float64)
    X = (np.linalg.inv(frame.K) @ np.vstack((xs, ys, np.ones(len(xs))))).T * z[:, None]
    n, c0 = fit_plane(X)
    angle = float(np.degrees(np.arccos(min(1.0, abs(n @ R[:, 2])))))
    return abs(c0 @ n) / abs(cam.C[2]), angle


def measure_grasp_widths(frame: BoardFrame, calib: BoardCalibration,
                         fractions: dict[str, float]) -> dict[str, float]:
    """
    Width of each piece type across the jaws at its grasp height, from the depth: the lateral extent
    (perpendicular to the viewing direction, the side the camera sees in full) of the points in a band
    around fraction x height, median over the pieces of a type.

    Returns:
        dict[str, float]: Lower-case piece symbol -> width in metres, for the types that were measured.
    """
    if frame.depth is None:
        return {}
    cam = BoardCamera(frame.K, calib.R, calib.t)
    widths = {}
    for k, symbol in frame.classes.items():
        key = symbol.lower()
        h = calib.piece_dims(key)[0]
        z = fractions.get(key, 0.45) * h
        P = cam.board_points(frame.depth, frame.labels == k)
        band = P[np.abs(P[:, 2] - z) < max(0.003, 0.05 * h)]
        if len(band) < 15:
            continue
        u = np.median(band[:, :2], axis=0) - cam.foot
        u /= np.linalg.norm(u)
        lateral = band[:, :2] @ np.array([-u[1], u[0]])
        widths.setdefault(key, []).append(np.quantile(lateral, 0.98) - np.quantile(lateral, 0.02))
    return {s: float(np.median(w)) for s, w in widths.items()}


def _measure_dims(frame: BoardFrame, cam: BoardCamera) -> dict[str, tuple[float, float]]:
    """Height = 95 % quantile of z, foot radius = half the lateral extent in the lowest eighth, per type."""
    heights, radii = {}, {}
    F = cam.foot
    for k, symbol in frame.classes.items():
        P = cam.board_points(frame.depth, frame.labels == k)
        P = P[P[:, 2] > 0.004]
        if len(P) < 30:
            continue
        h = np.quantile(P[:, 2], 0.95)
        heights.setdefault(symbol.lower(), []).append(h)
        u = np.median(P[:, :2], axis=0) - F
        u /= np.linalg.norm(u)
        v = np.array([-u[1], u[0]])
        band = P[P[:, 2] < 0.12 * h]
        if len(band) >= 15:
            lateral = band[:, :2] @ v
            radii.setdefault(symbol.lower(), []).append((np.quantile(lateral, 0.98) - np.quantile(lateral, 0.02)) / 2)
    return {s: (float(np.median(heights[s])), float(np.median(radii[s]))) for s in heights if s in radii}
