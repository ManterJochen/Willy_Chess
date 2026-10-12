"""Scene generator with exact ground truth: board image, piece masks, depth image, games.

Conventions (apply to all adapters):
- Image 640 x 480, camera matrix K with f = 600 px, no distortion.
- Board frame: origin at the board centre, x towards the h-file, y towards the 8th rank, z up, metres.
- Pose: X_cam = R @ X_board + t.
- Squares in python-chess numbering (a1 = 0, h8 = 63), square centre ((f - 3.5) * s, (r - 3.5) * s).
- Depth: metres along the optical axis (z-depth), 0 = invalid.

The pieces are waisted solids of revolution (the "truth"); estimators do not know this profile.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import chess
import cv2 as cv
import numpy as np

W, H = 640, 480
F = 600.0
K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1.0]])
SS = 4                       # supersampling when rendering
PX = 80                      # texture pixels per square
FRAME = 1.0                  # frame width in squares
SQ_DEFAULT = 0.057           # tournament square in metres
MIN_VISIBLE_PX = 25          # fewer visible pixels = piece counts as not detected

# (height, foot radius) in metres for 57 mm squares, roughly Staunton
DIMS = {chess.PAWN: (0.050, 0.0145), chess.KNIGHT: (0.060, 0.016), chess.BISHOP: (0.070, 0.016),
        chess.ROOK: (0.055, 0.016), chess.QUEEN: (0.085, 0.018), chess.KING: (0.095, 0.020)}
PROFILE = [(0, 1.0), (0.12, 1.0), (0.18, 0.75), (0.55, 0.45), (0.8, 0.6), (0.9, 0.5), (1.0, 0.3)]
_PZ = np.array([p[0] for p in PROFILE])
_PR = np.array([p[1] for p in PROFILE])
_ANG = np.linspace(0, 2 * np.pi, 28, endpoint=False)


@dataclass
class Frame:
    """What an adapter gets per image."""
    bgr: np.ndarray                                   # (H, W, 3) uint8
    K: np.ndarray                                     # (3, 3)
    depth: np.ndarray | None = None                   # (H, W) float32, z-depth in metres, 0 = invalid
    labels: np.ndarray | None = None                  # (H, W) int32, 0 = no piece, k = piece k
    classes: dict[int, str] = field(default_factory=dict)   # k -> FEN symbol, e.g. 'P' or 'n'


@dataclass
class Calibration:
    """Result of the calibration: pose, square size, optionally piece dimensions."""
    R: np.ndarray                                     # (3, 3), X_cam = R @ X_board + t
    t: np.ndarray                                     # (3,), metres
    square: float                                     # square size in metres
    dims: dict[str, tuple[float, float]] | None = None   # 'p','n','b','r','q','k' -> (height, foot radius)


@dataclass
class Setup:
    """Camera, board and sensor of a scene or of a whole game."""
    R: np.ndarray
    t: np.ndarray
    C: np.ndarray
    square: float = SQ_DEFAULT
    scale: float = 1.0                    # piece set relative to DIMS * square / 0.057
    depth_sigma: float = 0.005            # depth noise at 0.75 m, grows with z^2
    fly: float = 0.5                      # share of flying pixels at the mask edge
    tof_bias: float = 0.003               # extra depth on dark squares (ToF reflection)
    warp: float = 0.0                     # board centre bulged (+) or sagged (-), metres
    mask_px: int = 0                      # masks given to the adapter dilated (+) / eroded (-) by n px
    depth_shift: int = 0                  # depth image shifted against the colour image, px

    def dims(self, piece_type: int) -> tuple[float, float]:
        h, r = DIMS[piece_type]
        k = self.scale * self.square / SQ_DEFAULT
        return h * k, r * k

    def calibration(self) -> Calibration:
        """True pose and square size, piece dimensions nominal (without the random set factor)."""
        k = self.square / SQ_DEFAULT
        dims = {chess.piece_symbol(pt): (h * k, r * k) for pt, (h, r) in DIMS.items()}
        return Calibration(self.R.copy(), self.t.copy(), self.square, dims)


def look_at(tilt_deg: float, azim_deg: float, dist: float = 0.75) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Camera on a sphere around the board centre; tilt = angle from the vertical."""
    t, a = np.deg2rad(tilt_deg), np.deg2rad(azim_deg)
    C = dist * np.array([np.sin(t) * np.cos(a), np.sin(t) * np.sin(a), np.cos(t)])
    fwd = -C / np.linalg.norm(C)
    hint = np.array([-np.cos(a), -np.sin(a), 0.0])
    x = np.cross(fwd, hint); x /= np.linalg.norm(x)
    y = np.cross(fwd, x)
    R = np.stack([x, y, fwd])
    return R, -R @ C, C


def project(R, t, X):
    p = (K @ (R @ X.T + t[:, None])).T
    return p[:, :2] / p[:, 2:]


def homography_sq2img(R, t, square):
    """Board plane in square units -> image."""
    Hm = K @ np.column_stack((R[:, 0], R[:, 1], t)) @ np.diag([square, square, 1.0])
    return Hm / Hm[2, 2]


def sq_center(s: int, square: float) -> np.ndarray:
    return np.array([(chess.square_file(s) - 3.5) * square, (chess.square_rank(s) - 3.5) * square])


# --------------------------------------------------------------------------- Board image
def _texture(rng):
    n = int((8 + 2 * FRAME) * PX)
    tex = np.full((n, n), 120, np.float32)
    off = int(FRAME * PX)
    for r in range(8):
        for c in range(8):                                   # row = rank, column = file, a1 dark
            tex[off + r * PX: off + (r + 1) * PX, off + c * PX: off + (c + 1) * PX] = 60 if (r + c) % 2 == 0 else 200
    return tex + rng.normal(0, 3, tex.shape).astype(np.float32)


def render_board(rng, H_sq2img, blur=0.9, noise=2.0, clutter=0, blobs=0, paint=None) -> np.ndarray:
    """Board with frame on a calm background. clutter = chessboard patches beside the board,
    blobs = flat discs on random squares (hide corners), paint = (labels, grey values) for pieces."""
    h, w = H, W
    tex = _texture(rng)
    o = 0.5 / PX - (4 + FRAME)
    T = np.array([[1 / PX, 0, o], [0, 1 / PX, o], [0, 0, 1.0]])
    c = (SS - 1) / 2.0
    S = np.array([[SS, 0, c], [0, SS, c], [0, 0, 1.0]])
    hi = cv.warpPerspective(tex, S @ H_sq2img @ T, (w * SS, h * SS), flags=cv.INTER_LINEAR,
                            borderMode=cv.BORDER_CONSTANT, borderValue=-1)
    bg = cv.GaussianBlur(rng.normal(140, 12, (h, w)).astype(np.float32), (0, 0), 6)
    bg = cv.resize(bg, (w * SS, h * SS), interpolation=cv.INTER_LINEAR)
    hi = np.where(hi >= 0, hi, bg)
    for _ in range(clutter):
        cx, cy = rng.uniform(0, w * SS), rng.uniform(0, h * SS)
        sz, ang = rng.uniform(6, 14) * SS, rng.uniform(0, np.pi)
        x0, x1 = int(max(cx - 1.5 * sz, 0)), int(min(cx + 1.5 * sz, w * SS))
        y0, x1y = int(max(cy - 1.5 * sz, 0)), int(min(cy + 1.5 * sz, h * SS))
        yy, xx = np.mgrid[y0:x1y, x0:x1]
        u = (xx - cx) * np.cos(ang) + (yy - cy) * np.sin(ang)
        v = -(xx - cx) * np.sin(ang) + (yy - cy) * np.cos(ang)
        patch = (np.abs(u) < sz) & (np.abs(v) < sz)
        hi[y0:x1y, x0:x1] = np.where(patch, np.where(u * v > 0, 210.0, 50.0), hi[y0:x1y, x0:x1])
    for _ in range(blobs):
        r_, c_ = rng.integers(0, 8, 2)
        bx, by = c_ - 3.5, r_ - 3.5
        p = H_sq2img @ np.array([bx, by, 1.0]); p = p[:2] / p[2]
        q = H_sq2img @ np.array([bx + 1, by, 1.0]); q = q[:2] / q[2]
        rad = rng.uniform(0.38, 0.55) * np.linalg.norm(q - p)
        col = float(rng.choice([30.0, 230.0]))
        cv.circle(hi, (int(p[0] * SS), int((p[1] - 0.6 * rad) * SS)), int(rad * SS), col, -1, cv.LINE_AA)
        cv.ellipse(hi, (int(p[0] * SS), int(p[1] * SS)), (int(rad * SS), int(rad * 0.6 * SS)), 0, 0, 360, col, -1)
    img = cv.resize(hi, (w, h), interpolation=cv.INTER_AREA)
    if paint is not None:
        lab, grey = paint
        for k, g in grey.items():
            m = lab == k
            img[m] = g + rng.normal(0, 6, m.sum())
    if blur > 0:
        img = cv.GaussianBlur(img, (0, 0), blur)
    img = np.clip(img + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    return cv.cvtColor(img, cv.COLOR_GRAY2BGR)


def inner_corners(H_sq2img) -> np.ndarray:
    """True image position of the 49 inner corners, order (i, j) with i, j in -3..3."""
    ij = np.array([(i, j) for j in range(-3, 4) for i in range(-3, 4)], np.float64)
    p = (H_sq2img @ np.column_stack((ij, np.ones(len(ij)))).T).T
    return p[:, :2] / p[:, 2:]


def camera_squares(tilt, yaw, dist_sq, shift=(0.0, 0.0)) -> np.ndarray:
    """Camera for plain board images, board plane in square units (as in the first measurements)."""
    t, y = np.deg2rad(tilt), np.deg2rad(yaw)
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])
    R = Rx @ Rz
    Hm = K @ np.column_stack((R[:, 0], R[:, 1], np.array([shift[0], shift[1], dist_sq])))
    return Hm / Hm[2, 2]


# --------------------------------------------------------------------------- Pieces
def _bulge(setup: Setup, xy) -> np.ndarray:
    if setup.warp == 0.0:
        return np.zeros(np.shape(xy)[:-1]) if np.ndim(xy) > 1 else 0.0
    r2 = (np.asarray(xy)[..., 0] ** 2 + np.asarray(xy)[..., 1] ** 2) / (4 * setup.square) ** 2
    return setup.warp * np.maximum(0.0, 1 - r2)


def _silhouette(setup: Setup, xy, h, r):
    e = float(_bulge(setup, np.asarray(xy)))
    polys = []
    for (z0, r0), (z1, r1) in zip(PROFILE[:-1], PROFILE[1:]):
        P = [np.column_stack((xy[0] + rr * np.cos(_ANG), xy[1] + rr * np.sin(_ANG), np.full(len(_ANG), e + z)))
             for z, rr in ((z0 * h, r0 * r), (z1 * h, r1 * r))]
        hull = cv.convexHull(project(setup.R, setup.t, np.vstack(P)).astype(np.float32))
        polys.append(np.round(hull * 16).astype(np.int32))
    return polys


def jitter_offsets(rng, board: chess.Board, prev: dict | None = None, prev_board: chess.Board | None = None,
                   jitter=0.18) -> dict[int, np.ndarray]:
    """Offset of each piece from the square centre, in squares. Unmoved pieces keep their offset."""
    out = {}
    for s, p in board.piece_map().items():
        if prev is not None and prev_board is not None and prev_board.piece_at(s) == p and s in prev:
            out[s] = prev[s]
        else:
            rr, ang = jitter * np.sqrt(rng.uniform()), rng.uniform(0, 2 * np.pi)
            out[s] = rr * np.array([np.cos(ang), np.sin(ang)])
    return out


@dataclass
class Truth:
    square_of: dict[int, int]             # label -> true square (detected pieces only)
    n_pieces: int                         # pieces on the board, hidden ones too
    calibration: Calibration


def render_pieces(rng, setup: Setup, board: chess.Board, offsets: dict[int, np.ndarray]) -> tuple[Frame, Truth]:
    """Board image with pieces, labels, classes and depth image; occlusion by the painter's algorithm."""
    pieces = []
    for s, p in board.piece_map().items():
        xy = sq_center(s, setup.square) + offsets[s] * setup.square
        h, r = setup.dims(p.piece_type)
        pieces.append((s, p, xy, h, r))
    lab = np.zeros((H, W), np.int32)
    order = sorted(range(len(pieces)), key=lambda i: -np.linalg.norm(setup.C - np.r_[pieces[i][2], 0.03]))
    for i in order:
        for poly in _silhouette(setup, pieces[i][2], pieces[i][3], pieces[i][4]):
            cv.fillPoly(lab, [poly], i + 1, lineType=cv.LINE_8, shift=4)
    grey = {i + 1: (215.0 if pieces[i][1].color == chess.WHITE else 40.0) for i in range(len(pieces))}
    bgr = render_board(rng, homography_sq2img(setup.R, setup.t, setup.square), paint=(lab, grey))
    depth = _render_depth(rng, setup, lab, pieces)

    visible = {i + 1 for i in range(len(pieces)) if np.count_nonzero(lab == i + 1) >= MIN_VISIBLE_PX}
    given = np.where(np.isin(lab, list(visible)), lab, 0).astype(np.int32)
    if setup.mask_px > 0:
        grown = cv.dilate(given.astype(np.float32), np.ones((2 * setup.mask_px + 1,) * 2, np.uint8)).astype(np.int32)
        given = np.where(given == 0, grown, given)
    elif setup.mask_px < 0:
        kern = np.ones((2 * -setup.mask_px + 1,) * 2, np.uint8)
        for k in visible:
            m = given == k
            er = cv.erode(m.astype(np.uint8), kern).astype(bool)
            if er.any():
                given[m & ~er] = 0
    frame = Frame(bgr=bgr, K=K.copy(), depth=depth, labels=given,
                  classes={k: pieces[k - 1][1].symbol() for k in sorted(visible)})
    truth = Truth({k: pieces[k - 1][0] for k in visible}, len(pieces), setup.calibration())
    return frame, truth


def _render_depth(rng, setup: Setup, lab, pieces) -> np.ndarray:
    """z-depth: board surface everywhere, ray intersection with the true solid on piece pixels."""
    vs, us = np.mgrid[0:H, 0:W]
    pix = np.column_stack((us.ravel(), vs.ravel(), np.ones(H * W)))
    D = (setup.R.T @ (np.linalg.inv(K) @ pix.T)).T                 # direction, parameter = z-depth
    C = setup.C
    z = -C[2] / D[:, 2]
    for _ in range(3):                                             # intersection with the bulged surface
        X = C + z[:, None] * D
        z = (_bulge(setup, X[:, :2]) - C[2]) / D[:, 2]
    z_board = z.copy()
    X = C + z[:, None] * D
    f = np.floor(X[:, 0] / setup.square + 4); r = np.floor(X[:, 1] / setup.square + 4)
    dark = (f >= 0) & (f < 8) & (r >= 0) & (r < 8) & ((f + r) % 2 == 0) & (lab.ravel() == 0)
    for i, (s, p, xy, h, rad) in enumerate(pieces):
        idx = np.flatnonzero(lab.ravel() == i + 1)
        if len(idx) == 0:
            continue
        e = float(_bulge(setup, np.asarray(xy)))
        d = D[idx]
        s_lo = (e + h - C[2]) / d[:, 2]
        s_hi = z_board[idx]
        ss = s_lo[:, None] + (s_hi - s_lo)[:, None] * np.linspace(0, 1, 160)[None]
        P = C[None, None] + ss[..., None] * d[:, None, :]
        zz = P[..., 2] - e
        inside = (np.hypot(P[..., 0] - xy[0], P[..., 1] - xy[1]) <= rad * np.interp(zz / h, _PZ, _PR)) & (zz >= 0) & (zz <= h)
        hit = inside.any(1)
        z[idx[hit]] = ss[hit, inside[hit].argmax(1)]
    z = z + rng.normal(0, 1, z.shape) * setup.depth_sigma * (z / 0.75) ** 2 + setup.tof_bias * dark
    fg = (lab > 0).astype(np.uint8)
    border = (fg.ravel() > 0) & (cv.erode(fg, np.ones((5, 5), np.uint8)).ravel() == 0)
    fly = border & (rng.uniform(size=z.shape) < setup.fly)
    z[fly] = z[fly] + rng.uniform(size=fly.sum()) * (z_board[fly] - z[fly])
    depth = z.reshape(H, W).astype(np.float32)
    if setup.depth_shift:
        depth = np.roll(depth, -setup.depth_shift, axis=1)
        depth[:, -setup.depth_shift:] = 0
    return depth


def random_position(rng, plies: int) -> chess.Board:
    b = chess.Board()
    for _ in range(plies):
        moves = list(b.legal_moves)
        if not moves:
            break
        b.push(moves[rng.integers(len(moves))])
    return b
