"""
Find the board in one image: inner corners, lattice, homography and the 8x8 window.

Everything here works in square units. The grid has its inner corners on the integers
-3..3 and the board edge at +-4; which corner is a1 and how large a square is in metres
is decided later from the pose, the pieces and the depth (src/mapping/calibration.py).
"""
from dataclasses import dataclass

import cv2 as cv
import numpy as np

from src.board.saddle import SaddlePointDetector
from src.board.symetric_solver import ChessboardSolver

# The four lattice steps a basis can be built from: the two sides and the two diagonals.
_STEPS = [np.array(step) for step in ((1, 0), (0, 1), (1, 1), (1, -1))]
# The lattice basis comes from the strongest corners only: in a real room every weak saddle of the
# background enters the displacement histogram, and the basis tips onto a diagonal (recorded webcam
# frames: 30 of 57 frames of a hand-held board right with all corners, 56 of 57 with the strongest 60).
BASIS_CORNERS = 60


@dataclass(frozen=True)
class GridDetection:
    """
    The board grid found in one image.

    Attributes:
        H_grid_to_image (np.ndarray): 3x3 homography from grid coordinates to pixels.
            The 49 inner corners sit on the integers (i, j) with -3 <= i, j <= 3.
        points (np.ndarray): Inlier corners in pixels, shape (N, 2).
        indices (np.ndarray): Their integer grid coordinates, shape (N, 2).
    """
    H_grid_to_image: np.ndarray
    points: np.ndarray
    indices: np.ndarray

    @property
    def H_image_to_grid(self) -> np.ndarray:
        H = np.linalg.inv(self.H_grid_to_image)
        return H / H[2, 2]


def to_gray(image: np.ndarray) -> np.ndarray:
    """Grayscale view of a BGR, BGRA or grayscale image."""
    if image.ndim == 2:
        return image
    code = cv.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv.COLOR_BGR2GRAY
    return cv.cvtColor(image, code)


def transform(H: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a homography to (N, 2) points."""
    pts = np.asarray(points, np.float64).reshape(-1, 1, 2)
    return cv.perspectiveTransform(pts, np.asarray(H, np.float64)).reshape(-1, 2)


def detect_corners(image: np.ndarray) -> np.ndarray:
    """
    Inner board corners with subpixel accuracy.

    Args:
        image (np.ndarray): BGR or grayscale image.

    Returns:
        np.ndarray: Corner candidates in pixels, shape (N, 2), clutter included.
    """
    return SaddlePointDetector(to_gray(image), 0, True).detect()


def detect_grid(image: np.ndarray, ransac_px: float = 2.0, min_inliers: int = 12) -> GridDetection | None:
    """
    Find the board grid in an image.

    1. Saddle points with cornerSubPix refinement.
    2. Lattice basis from the histogram of pairwise displacements of the strongest corners
       (ChessboardSolver); every corner gets its coordinates in that basis.
    3. Side or diagonal: a photometric test picks the basis made of the two board axes.
    4. RANSAC grid -> image with a threshold in pixels, where the corner noise lives.
    5. The 8x8 window: a lattice is shift invariant, only the board edge breaks the
       symmetry. Most inliers in the 7x7 window, ties broken by the edge energy on the
       18 predicted grid lines.

    Args:
        image (np.ndarray): BGR or grayscale image.
        ransac_px (float): RANSAC inlier threshold in pixels.
        min_inliers (int): Fewest inlier corners to accept a grid (grids made up from a floor or a shelf
            had at most 9 in the recordings, real boards at least 26).

    Returns:
        GridDetection | None: The grid, or None if no board was found.
    """
    gray = to_gray(image)
    points = detect_corners(gray)
    if len(points) < 8:
        return None

    strongest = points[:BASIS_CORNERS]                 # detect_corners returns the strongest first
    basis, _ = ChessboardSolver(strongest).estimate_lattice(strongest)
    if abs(np.linalg.det(basis)) < 1e-6:
        return None
    lattice = _choose_axes(gray, points, _lattice_coordinates(points, basis))

    H, _ = cv.findHomography(np.round(lattice).astype(np.float32), points.astype(np.float32), cv.RANSAC,
                             ransac_px, maxIters=5000, confidence=0.999)
    if H is None:
        return None
    indices, inliers = _inliers(H, points, ransac_px)
    if inliers.sum() < min_inliers:
        return None

    H = _choose_window(gray, H, indices[inliers])
    indices, inliers = _inliers(H, points, ransac_px)
    inliers &= np.abs(indices).max(axis=1) <= 3
    if inliers.sum() < min_inliers:
        return None

    return GridDetection(H / H[2, 2], points[inliers], indices[inliers])


def board_homography(image: np.ndarray) -> np.ndarray | None:
    """Homography image -> grid (inner corners on -3..3), or None if no board was found."""
    grid = detect_grid(image)
    return None if grid is None else grid.H_image_to_grid


def _lattice_coordinates(points: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """
    Coordinates of the points in a lattice basis, shifted so that lattice points lie on integers. The
    fractional parts live on a torus, so their offset is a circular mean.
    """
    lattice = points @ np.linalg.inv(basis)
    offset = np.angle(np.mean(np.exp(2j * np.pi * lattice), axis=0)) / (2 * np.pi)
    return lattice - offset


def _inliers(H_grid_to_image: np.ndarray, points: np.ndarray, ransac_px: float) -> tuple[np.ndarray, np.ndarray]:
    """Nearest grid index of every point and whether it reprojects within ransac_px."""
    indices = np.round(transform(np.linalg.inv(H_grid_to_image), points))
    error = np.linalg.norm(transform(H_grid_to_image, indices) - points, axis=1)
    return indices, error < ransac_px


def _choose_axes(gray: np.ndarray, points: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    """
    Re-express the lattice coordinates in the basis of the two board axes.

    Every basis B M with integer M and det M = +-1 spans the same points, and seen at an
    angle a diagonal can be shorter than a side. Along a side the midpoint of two
    neighbouring corners lies on an edge, with the gradient perpendicular to the step t;
    along a diagonal it lies inside a square. Score per step: median of |g x t| - |g . t|.
    """
    smooth = cv.GaussianBlur(gray.astype(np.float32), (0, 0), 1.0)
    gx, gy = cv.Sobel(smooth, cv.CV_32F, 1, 0), cv.Sobel(smooth, cv.CV_32F, 0, 1)
    h, w = smooth.shape

    nearest = np.round(lattice).astype(int)
    on_lattice = np.linalg.norm(lattice - nearest, axis=1) < 0.25
    lookup = {tuple(k): i for i, k in zip(np.nonzero(on_lattice)[0], nearest[on_lattice])}

    scores = []
    for step in _STEPS:
        values = []
        for k, i in lookup.items():
            j = lookup.get((k[0] + step[0], k[1] + step[1]))
            if j is None:
                continue
            d = points[j] - points[i]
            length = np.linalg.norm(d)
            mx, my = np.round(0.5 * (points[i] + points[j])).astype(int)
            if length < 1e-6 or not (0 <= mx < w and 0 <= my < h):
                continue
            g, t = np.array([gx[my, mx], gy[my, mx]]), d / length
            values.append(abs(g[0] * t[1] - g[1] * t[0]) - abs(g @ t))
        scores.append(np.median(values) if len(values) >= 3 else -np.inf)

    best, M = -np.inf, np.eye(2)
    for a in range(4):
        for b in range(a + 1, 4):
            candidate = np.array([_STEPS[a], _STEPS[b]], float)
            if abs(round(np.linalg.det(candidate))) == 1 and min(scores[a], scores[b]) > best:
                best, M = min(scores[a], scores[b]), candidate
    return lattice @ np.linalg.inv(M)


def gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    """Gradient magnitude of a lightly smoothed grayscale image."""
    smooth = cv.GaussianBlur(gray.astype(np.float32), (0, 0), 1.0)
    return cv.magnitude(cv.Sobel(smooth, cv.CV_32F, 1, 0), cv.Sobel(smooth, cv.CV_32F, 0, 1))


def line_energy(grad: np.ndarray, H_grid_to_image: np.ndarray, axis: int, k: float,
                visible: np.ndarray | None = None) -> float:
    """
    Median gradient magnitude along one grid line, sampled at the 8 square midpoints of the board:
    the line x = k (axis 0) or y = k (axis 1) in grid coordinates (board edges at +-4).

    Args:
        visible (np.ndarray | None): Boolean image, False where the board is hidden (pieces); those
            samples are skipped.

    Returns:
        float: The median, NaN with fewer than 4 usable samples.
    """
    h, w = grad.shape
    along = np.arange(-3.5, 4.0, 1.0)
    pts = np.column_stack((np.full(8, k), along)) if axis == 0 else np.column_stack((along, np.full(8, k)))
    p = np.rint(transform(H_grid_to_image, pts)).astype(int)
    ok = (p[:, 0] >= 0) & (p[:, 1] >= 0) & (p[:, 0] < w) & (p[:, 1] < h)
    if visible is not None:
        ok[ok] = visible[p[ok, 1], p[ok, 0]]
    return float(np.median(grad[p[ok, 1], p[ok, 0]])) if ok.sum() >= 4 else float("nan")


def _edge_energy(grad: np.ndarray, H_grid_to_image: np.ndarray, cx: int, cy: int) -> float:
    """Gradient magnitude along the 18 grid lines of the window centred at (cx, cy)."""
    h, w = grad.shape
    along = np.arange(-3.5, 4.0, 1.0)
    total = 0.0
    for k in range(-4, 5):
        for line in (np.column_stack((np.full(8, cx + k), cy + along)),
                     np.column_stack((cx + along, np.full(8, cy + k)))):
            p = np.rint(transform(H_grid_to_image, line)).astype(int)
            ok = (p[:, 0] >= 0) & (p[:, 1] >= 0) & (p[:, 0] < w) & (p[:, 1] < h)
            total += float(np.median(grad[p[ok, 1], p[ok, 0]])) if ok.sum() >= 4 else 0.0
    return total


def _choose_window(gray: np.ndarray, H_grid_to_image: np.ndarray, indices: np.ndarray, band: int = 3) -> np.ndarray:
    """
    Shift the grid so that the board's 7x7 inner corners sit on -3..3.

    Candidates within `band` inliers of the best count compete on edge energy: in the
    right window all 18 lines lie on real edges, a shifted one puts a line into the frame.
    """
    grad = gradient_magnitude(gray)

    idx = indices.astype(int)
    lo, hi = idx.min(axis=0), idx.max(axis=0)
    candidates = [(int(np.sum((np.abs(idx[:, 0] - cx) <= 3) & (np.abs(idx[:, 1] - cy) <= 3))), cx, cy)
                  for cx in range(lo[0] - 3, hi[0] + 4) for cy in range(lo[1] - 3, hi[1] + 4)]
    most = max(c[0] for c in candidates)
    _, cx, cy = max((_edge_energy(grad, H_grid_to_image, cx, cy), cx, cy)
                    for n, cx, cy in candidates if n >= most - band)
    return H_grid_to_image @ np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1.0]])
