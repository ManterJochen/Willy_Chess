"""
Board geometry: the board frame, the camera that looks at it, PnP and plane fits.

Conventions, shared by the whole pipeline and by tests/bench:
- Board frame: origin in the board centre, x towards the h-file, y towards rank 8, z up, metres.
- Pose: X_cam = R @ X_board + t.
- Squares in python-chess numbering (a1 = 0, h8 = 63), centre ((file - 3.5) * s, (rank - 3.5) * s).
- Depth: z-depth along the optical axis in metres, 0 = invalid.
"""
from dataclasses import dataclass

import chess
import cv2 as cv
import numpy as np

# Piece height and foot radius in metres on 57 mm squares (about Staunton). Only a prior:
# calibration measures the real set from the depth in the starting position.
NOMINAL_DIMS = {"p": (0.050, 0.0145), "n": (0.060, 0.016), "b": (0.070, 0.016),
                "r": (0.055, 0.016), "q": (0.085, 0.018), "k": (0.095, 0.020)}
NOMINAL_SQUARE = 0.057


@dataclass
class BoardCalibration:
    """
    Where the board is for one camera view, how large a square is and how large the pieces are.

    Attributes:
        R (np.ndarray): Rotation board -> camera, shape (3, 3).
        t (np.ndarray): Translation board -> camera in metres, shape (3,).
        square (float): Square size in metres.
        dims (dict[str, tuple[float, float]] | None): Lower-case piece symbol -> (height, foot radius)
            in metres, measured during calibration; None falls back to NOMINAL_DIMS scaled to the square.
    """
    R: np.ndarray
    t: np.ndarray
    square: float
    dims: dict[str, tuple[float, float]] | None = None

    def piece_dims(self, symbol: str) -> tuple[float, float]:
        """Height and foot radius of a piece (FEN symbol, either colour) in metres."""
        key = symbol.lower()
        if self.dims and key in self.dims:
            return self.dims[key]
        h, r = NOMINAL_DIMS[key]
        k = self.square / NOMINAL_SQUARE
        return h * k, r * k


class BoardCamera:
    """
    A camera relative to the board plane z = 0.

    The unit is the unit of t: metres for a calibrated pose, squares for a pose solved
    on the grid indices before the scale is known.
    """

    def __init__(self, K: np.ndarray, R: np.ndarray, t: np.ndarray):
        self.K = np.asarray(K, np.float64)
        self.R = np.asarray(R, np.float64)
        self.t = np.asarray(t, np.float64).reshape(3)
        self.C = -self.R.T @ self.t
        H = self.K @ np.column_stack((self.R[:, 0], self.R[:, 1], self.t))
        self.H_plane_to_image = H / H[2, 2]
        self.H_image_to_plane = np.linalg.inv(self.H_plane_to_image)

    @property
    def foot(self) -> np.ndarray:
        """The camera centre dropped onto the board plane, shape (2,)."""
        return self.C[:2]

    def to_plane(self, uv: np.ndarray) -> np.ndarray:
        """Pixels (N, 2) to the point where their ray meets the board plane, (N, 2)."""
        p = np.column_stack((uv, np.ones(len(uv)))) @ self.H_image_to_plane.T
        return p[:, :2] / p[:, 2:]

    def project(self, X: np.ndarray) -> np.ndarray:
        """Board points (N, 3) to pixels (N, 2)."""
        p = (self.K @ (self.R @ np.asarray(X, np.float64).T + self.t[:, None])).T
        return p[:, :2] / p[:, 2:]

    def board_points(self, depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        The valid depth pixels of a mask as 3D points in the board frame.

        Args:
            depth (np.ndarray): z-depth in metres, 0 = invalid, shape (H, W).
            mask (np.ndarray): Boolean mask, shape (H, W).

        Returns:
            np.ndarray: Points in the board frame, shape (N, 3).
        """
        ys, xs = np.nonzero(mask & (depth > 0))
        return self.board_points_at(depth, xs, ys)

    def board_points_at(self, depth: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """Board-frame 3D points of the depth pixels (xs, ys)."""
        z = depth[ys, xs].astype(np.float64)
        X = (np.linalg.inv(self.K) @ np.vstack((xs, ys, np.ones(len(xs))))).T * z[:, None]
        return (self.R.T @ (X - self.t).T).T


def square_centres(square: float) -> np.ndarray:
    """Centres of the 64 squares in the board frame, python-chess order, shape (64, 2)."""
    return np.array([[(chess.square_file(s) - 3.5) * square, (chess.square_rank(s) - 3.5) * square]
                     for s in range(64)])


def solve_pnp(K: np.ndarray, plane_xy: np.ndarray, pixels: np.ndarray,
              guess: tuple[np.ndarray, np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Pose of a planar target from its points on z = 0 and their pixels.

    IPPE for the planar case, then Levenberg-Marquardt on the reprojection error. With a
    guess (a pose predicted from the robot's kinematics) only the refinement runs, which
    also settles the two-fold ambiguity of planar targets in favour of the prediction.

    Returns:
        tuple[np.ndarray, np.ndarray]: R (3, 3) and t (3,), X_cam = R @ X_plane + t.
    """
    obj = np.column_stack((plane_xy, np.zeros(len(plane_xy)))).astype(np.float64)
    img = np.asarray(pixels, np.float64)
    if guess is None:
        _, rvec, tvec = cv.solvePnP(obj, img, K, None, flags=cv.SOLVEPNP_IPPE)
    else:
        rvec, tvec = cv.Rodrigues(guess[0])[0], np.asarray(guess[1], np.float64).reshape(3, 1)
    rvec, tvec = cv.solvePnPRefineLM(obj, img, K, None, rvec, tvec)
    return cv.Rodrigues(rvec)[0], tvec.ravel()


def reprojection_rms(K: np.ndarray, R: np.ndarray, t: np.ndarray, plane_xy: np.ndarray, pixels: np.ndarray) -> float:
    """RMS reprojection error in pixels of plane points under a pose."""
    cam = BoardCamera(K, R, t)
    proj = cam.project(np.column_stack((plane_xy, np.zeros(len(plane_xy)))))
    return float(np.sqrt(np.mean(np.sum((proj - pixels) ** 2, axis=1))))


def fit_plane(X: np.ndarray, rounds: int = 3, k_sigma: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Total least squares plane through 3D points, trimmed at k_sigma a few times.

    The normal is the singular vector of the smallest singular value of the centred
    points, which minimises the orthogonal distances (noise sits in all coordinates).

    Returns:
        tuple[np.ndarray, np.ndarray]: Unit normal (3,) and a point on the plane, the centroid (3,).
    """
    X = np.asarray(X, np.float64)
    for _ in range(rounds):
        c0 = X.mean(axis=0)
        n = np.linalg.svd(X - c0, full_matrices=False)[2][-1]
        r = (X - c0) @ n
        X = X[np.abs(r) < k_sigma * r.std() + 1e-12]
    c0 = X.mean(axis=0)
    n = np.linalg.svd(X - c0, full_matrices=False)[2][-1]
    return n, c0


def pose_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """4x4 homogeneous matrix of a pose."""
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, np.asarray(t).reshape(3)
    return T
