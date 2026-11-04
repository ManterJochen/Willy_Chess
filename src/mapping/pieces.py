"""
Piece to square: where each piece stands on the board, and which square that is.

The homography maps a pixel exactly onto the board only if the pixel shows a point ON the
board. A body point at height z lands behind the piece, by z * d / (h_cam - z) (d: horizontal
distance to the camera foot F, h_cam: camera height). At 60 degrees a king's mask centroid lands
about 1.7 squares off. Only the foot touches the board, so:

- Foot point: the mask pixel nearest to F on the board plane is the front rim of the foot,
  one foot radius further along the viewing direction is the centre.
- Depth: every mask pixel as a 3D point; each visible surface point lies in front of the
  piece axis, so the axis is estimated from the points plus their distance to the axis.
- Censored fusion: a foot hidden behind another piece can only push the foot estimate away
  from the camera, never towards it. If it lies behind the depth estimate, the depth estimate
  wins. Otherwise the depth gives the direction from the camera and both together the distance
  (see fuse).
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.mapping.frame import BoardFrame
from src.mapping.geometry import BoardCalibration, BoardCamera, square_centres

FOOT_QUANTILE = 0.02        # nearest 2 % of the mask instead of the minimum: one stray pixel moves nothing
MIN_DEPTH_POINTS = 30       # fewer valid depth points: the foot point alone
FUSION_MAX_GAP = 0.025      # metres: further apart than this, one of the two is wrong (a real set: at most ~13 mm)
OFF_BOARD_MARGIN = 0.75     # squares beyond the board edge before a piece counts as off the board (graveyard: 1.5)


def foot_xy(cam: BoardCamera, mask: np.ndarray, radius: float, quantile: float = FOOT_QUANTILE) -> np.ndarray:
    """
    Foot centre of a piece on the board plane from its mask alone.

    Args:
        cam (BoardCamera): Camera relative to the board.
        mask (np.ndarray): Boolean mask of the piece.
        radius (float): Foot radius in the unit of cam.

    Returns:
        np.ndarray: Foot centre (x, y) in the board frame.
    """
    ys, xs = np.nonzero(mask)
    g = cam.to_plane(np.column_stack((xs, ys)).astype(np.float64))
    F = cam.foot
    d = np.linalg.norm(g - F, axis=1)
    k = np.argsort(d)[int(quantile * len(d))]
    return g[k] + radius * (g[k] - F) / max(d[k], 1e-12)


def depth_xy(points: np.ndarray, F: np.ndarray, height: float, radius: float,
             n_min: int = MIN_DEPTH_POINTS) -> np.ndarray | None:
    """
    Piece axis on the board plane from the piece's depth points.

    Points between 4 mm and 1.3 x the height above the board are kept, which drops the
    flying pixels at the mask edge (they lie almost on the board). Split relative to the
    viewing direction u into lateral l and longitudinal lambda: a visible surface point
    lies sqrt(r(z)^2 - (l - l_hat)^2) in front of the axis, r(z) a coarse cone frustum.

    Args:
        points (np.ndarray): The piece's depth points in the board frame, (N, 3).
        F (np.ndarray): Camera foot on the board plane, (2,).
        height (float): Piece height in metres.
        radius (float): Foot radius in metres.

    Returns:
        np.ndarray | None: Axis position (x, y), or None with too little depth.
    """
    P = points[(points[:, 2] > 0.004) & (points[:, 2] < 1.3 * height)]
    if len(P) < n_min:
        return None
    u = np.median(P[:, :2], axis=0) - F
    u /= np.linalg.norm(u)
    v = np.array([-u[1], u[0]])
    lateral, longitudinal = (P[:, :2] - F) @ v, (P[:, :2] - F) @ u
    l_hat = np.median(lateral)
    r_z = radius * (1 - 0.45 * np.clip(P[:, 2] / height, 0, 1))
    behind = np.sqrt(np.maximum(r_z ** 2 - (lateral - l_hat) ** 2, 0))
    return F + np.median(longitudinal + behind) * u + l_hat * v


def fuse(x_foot: np.ndarray, x_depth: np.ndarray | None, F: np.ndarray, max_gap: float = FUSION_MAX_GAP,
         for_grasp: bool = False) -> np.ndarray:
    """
    Foot point and depth estimate as one position.

    Measured on a real set (30 pieces, 2026-11-04): sideways the depth repeats to a millimetre between views
    where the foot point jumps by up to 14 mm with its mask, so the depth gives the direction from the camera.
    Along that direction their errors have opposite signs (foot -3 mm, depth +3.5 mm and up to +12.6 mm on
    the far tall pieces), so the distance is their mean: 1 piece of 30 more than 8 mm off, 7 with the depth alone.

    Args:
        x_foot (np.ndarray): Foot point, (2,).
        x_depth (np.ndarray | None): Depth estimate, (2,); None with too little depth, then the foot point alone.
        F (np.ndarray): Camera foot on the board plane, (2,).
        max_gap (float): Further apart than this they contradict each other: no mean of a wrong value.
        for_grasp (bool): Which one is believed then: the depth for a grasp, which needs the piece's own surface;
            the foot point for a square, which must survive a depth that does not fit.
    """
    if x_depth is None:
        return x_foot
    u = (x_depth - F) / np.linalg.norm(x_depth - F)
    behind = (x_foot - x_depth) @ u
    if behind > 0:                      # at least partly hidden: all a foot can say is that the piece is not nearer
        return x_depth
    if np.linalg.norm(x_foot - x_depth) > max_gap:
        return x_depth if for_grasp else x_foot
    return x_depth + 0.5 * behind * u


def piece_positions(frame: BoardFrame, calib: BoardCalibration, for_grasp: bool = False) -> dict[int, np.ndarray]:
    """
    Position of every detected piece on the board plane.

    The square needs half a square of accuracy and must survive a depth that does not fit, the grasp
    needs millimetres: the same fusion, which only differs in what it believes when foot and depth
    contradict each other (see fuse).

    Args:
        for_grasp (bool): Positions to grip at instead of positions to assign squares with.

    Returns:
        dict[int, np.ndarray]: Piece label -> (x, y) in the board frame, metres.
    """
    cam = BoardCamera(frame.K, calib.R, calib.t)
    positions = {}
    for k, symbol in frame.classes.items():
        mask = frame.labels == k
        if not mask.any():
            continue
        h, r = calib.piece_dims(symbol)
        x_foot = foot_xy(cam, mask, r)
        x_depth = depth_xy(cam.board_points(frame.depth, mask), cam.foot, h, r) if frame.depth is not None else None
        positions[k] = fuse(x_foot, x_depth, cam.foot, for_grasp=for_grasp)
    return positions


def assign_squares(positions: dict[int, np.ndarray], square: float,
                   off_board_margin: float = OFF_BOARD_MARGIN) -> dict[int, int]:
    """
    Free mode, one image and no history: every piece on exactly one square, at most one
    piece per square. A linear assignment (Hungarian) on squared distances to the centres.

    Pieces further than off_board_margin squares outside the board (captured pieces at the
    side) are left out.

    Returns:
        dict[int, int]: Piece label -> square 0..63.
    """
    keys = [k for k, xy in positions.items() if np.abs(xy / square).max() <= 4 + off_board_margin]
    if not keys:
        return {}
    X = np.array([positions[k] for k in keys]) / square
    centres = square_centres(square) / square
    rows, cols = linear_sum_assignment(((X[:, None] - centres[None]) ** 2).sum(axis=-1))
    return {keys[i]: int(j) for i, j in zip(rows, cols)}


def assign(frame: BoardFrame, calib: BoardCalibration) -> dict[int, int]:
    """Piece label -> square for every piece in the frame (free mode)."""
    return assign_squares(piece_positions(frame, calib), calib.square)
