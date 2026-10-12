"""Reference stages 3-5: calibration, piece to square, move tracking (as in the document)."""
from __future__ import annotations

import chess
import cv2 as cv
import numpy as np
from scipy.optimize import linear_sum_assignment

from bench.reference.board import detect_grid
from bench.sim import Calibration, Frame

NOMINAL = {'p': (0.050, 0.0145), 'n': (0.060, 0.016), 'b': (0.070, 0.016),
           'r': (0.055, 0.016), 'q': (0.085, 0.018), 'k': (0.095, 0.020)}   # for 57 mm squares
ROT90 = np.array([[0, -1], [1, 0]])


# --------------------------------------------------------------------------- Geometry
class Cam:
    """Camera relative to the board; unit = unit of t (metres or squares)."""
    def __init__(self, K, R, t):
        self.K, self.R, self.t = K, R, np.asarray(t, float)
        self.C = -R.T @ self.t
        Hb = K @ np.column_stack((R[:, 0], R[:, 1], self.t))
        self.Hinv = np.linalg.inv(Hb / Hb[2, 2])

    def to_plane(self, uv):
        p = np.column_stack((uv, np.ones(len(uv)))) @ self.Hinv.T
        return p[:, :2] / p[:, 2:]

    def project(self, X):
        p = (self.K @ (self.R @ X.T + self.t[:, None])).T
        return p[:, :2] / p[:, 2:]


def _dims(calib: Calibration, sym: str):
    if calib.dims and sym.lower() in calib.dims:
        return calib.dims[sym.lower()]
    h, r = NOMINAL[sym.lower()]
    return h * calib.square / 0.057, r * calib.square / 0.057


def foot_xy(cam: Cam, mask, r, q=0.02):
    """Front foot edge = mask point with the smallest distance to the camera foot point F (2 % quantile),
    then one foot radius further in the viewing direction."""
    ys, xs = np.nonzero(mask)
    g = cam.to_plane(np.column_stack((xs, ys)).astype(float))
    F = cam.C[:2]
    d = np.linalg.norm(g - F, axis=1)
    k = np.argsort(d)[int(q * len(d))]
    return g[k] + r * (g[k] - F) / max(d[k], 1e-12)


def board_points(frame: Frame, cam: Cam, mask) -> np.ndarray:
    """Depth pixels of a mask as 3D points in the board frame."""
    ys, xs = np.nonzero(mask & (frame.depth > 0))
    z = frame.depth[ys, xs].astype(float)
    X = (np.linalg.inv(frame.K) @ np.vstack((xs, ys, np.ones(len(xs))))).T * z[:, None]
    return (cam.R.T @ (X - cam.t).T).T


def depth_xy(P, F, h, r, n_min=30):
    """Centre from depth: median across, median along plus the distance from the surface to the axis."""
    P = P[(P[:, 2] > 0.004) & (P[:, 2] < 1.3 * h)]
    if len(P) < n_min:
        return None
    u = np.median(P[:, :2], 0) - F; u /= np.linalg.norm(u); v = np.array([-u[1], u[0]])
    lat, lon = (P[:, :2] - F) @ v, (P[:, :2] - F) @ u
    lc = np.median(lat)
    rz = r * (1 - 0.45 * np.clip(P[:, 2] / h, 0, 1))
    return F + np.median(lon + np.sqrt(np.maximum(rz ** 2 - (lat - lc) ** 2, 0))) * u + lc * v


def fuse(x_foot, x_depth, F, tau=0.006):
    """Censored fusion: a hidden foot only biases the foot point away from the camera."""
    if x_depth is None:
        return x_foot
    u = (x_depth - F) / np.linalg.norm(x_depth - F)
    return x_depth if (x_foot - x_depth) @ u > tau else x_foot


def piece_positions(frame: Frame, calib: Calibration) -> dict[int, np.ndarray]:
    cam = Cam(frame.K, calib.R, calib.t)
    out = {}
    for k, sym in frame.classes.items():
        m = frame.labels == k
        if not m.any():
            continue
        h, r = _dims(calib, sym)
        xf = foot_xy(cam, m, r)
        xd = depth_xy(board_points(frame, cam, m), cam.C[:2], h, r) if frame.depth is not None else None
        out[k] = fuse(xf, xd, cam.C[:2])
    return out


def _centres(square):
    return np.array([[(chess.square_file(s) - 3.5) * square, (chess.square_rank(s) - 3.5) * square] for s in range(64)])


def assign(frame: Frame, calib: Calibration) -> dict[int, int]:
    """Free mode: every piece on exactly one square, Hungarian on squared distances."""
    pos = piece_positions(frame, calib)
    if not pos:
        return {}
    keys = list(pos)
    X = np.array([pos[k] for k in keys]) / calib.square
    Cs = _centres(calib.square) / calib.square
    r, c = linear_sum_assignment(((X[:, None] - Cs[None]) ** 2).sum(-1))
    return {keys[i]: int(j) for i, j in zip(r, c)}


# --------------------------------------------------------------------------- Calibration
def _pnp(K, idx, pts):
    obj = np.column_stack((idx, np.zeros(len(idx)))).astype(np.float64)
    _, rv, tv = cv.solvePnP(obj, pts.astype(np.float64), K, None, flags=cv.SOLVEPNP_IPPE)
    rv, tv = cv.solvePnPRefineLM(obj, pts.astype(np.float64), K, None, rv, tv)
    return cv.Rodrigues(rv)[0], tv.ravel()


def _feet_sq(frame, R, t_sq):
    """Foot points of all pieces in square units (radius 0.25 squares) and their symbols."""
    cam = Cam(frame.K, R, t_sq)
    keys = [k for k in frame.classes if (frame.labels == k).any()]
    feet = np.array([foot_xy(cam, frame.labels == k, 0.25) for k in keys]).reshape(-1, 2)
    return feet, [frame.classes[k] for k in keys]


def _parity(frame, R, t_sq):
    """> 0 if squares with even f + r are light, i.e. a1 is not dark."""
    cam = Cam(frame.K, R, t_sq)
    gray = cv.cvtColor(frame.bgr, cv.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    off = np.array([(a, b) for a in np.linspace(-0.3, 0.3, 5) for b in np.linspace(-0.3, 0.3, 5)])
    med, sign = [], []
    for f in range(8):
        for r in range(8):
            c = np.array([f - 3.5, r - 3.5]) + off
            uv = np.rint(cam.project(np.column_stack((c, np.zeros(len(c)))))).astype(int)
            uv = uv[(uv[:, 0] >= 0) & (uv[:, 1] >= 0) & (uv[:, 0] < w) & (uv[:, 1] < h)]
            free = frame.labels[uv[:, 1], uv[:, 0]] == 0
            if free.sum() >= 3:
                med.append(np.median(gray[uv[free, 1], uv[free, 0]])); sign.append(1 if (f + r) % 2 == 0 else -1)
    med = np.array(med)
    return float(np.sum(np.array(sign) * (med - np.median(med)))) if len(med) else 0.0


def _scale_from_depth(frame, R, t_sq):
    """Square size = camera height in metres (plane through free board pixels) / camera height in squares."""
    cam = Cam(frame.K, R, t_sq)
    border = cam.project(np.array([[-4, -4, 0], [4, -4, 0], [4, 4, 0], [-4, 4, 0]], float))
    area = np.zeros(frame.labels.shape, np.uint8)
    cv.fillPoly(area, [np.round(border).astype(np.int32)], 1)
    free = cv.erode(((area > 0) & (frame.labels == 0)).astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    ys, xs = np.nonzero(free & (frame.depth > 0))
    if len(xs) < 500:
        return None, 90.0
    step = max(1, len(xs) // 6000)
    ys, xs = ys[::step], xs[::step]
    z = frame.depth[ys, xs].astype(float)
    X = (np.linalg.inv(frame.K) @ np.vstack((xs, ys, np.ones(len(xs))))).T * z[:, None]
    for _ in range(3):
        c0 = X.mean(0); n = np.linalg.svd(X - c0, full_matrices=False)[2][-1]
        r = (X - c0) @ n
        X = X[np.abs(r) < 3 * r.std() + 1e-12]
    c0 = X.mean(0); n = np.linalg.svd(X - c0, full_matrices=False)[2][-1]
    angle = np.degrees(np.arccos(min(1.0, abs(n @ R[:, 2]))))     # depth normal against the board z from the pose
    return abs(c0 @ n) / abs(cam.C[2]), angle


def _measure_dims(frame, cam: Cam):
    """Height = 95 % quantile of z, foot radius = half the lateral extent in the bottom eighth."""
    hs, rs = {}, {}
    F = cam.C[:2]
    for k, sym in frame.classes.items():
        P = board_points(frame, cam, frame.labels == k)
        P = P[P[:, 2] > 0.004]
        if len(P) < 30:
            continue
        h = np.quantile(P[:, 2], 0.95)
        hs.setdefault(sym.lower(), []).append(h)
        u = np.median(P[:, :2], 0) - F; u /= np.linalg.norm(u); v = np.array([-u[1], u[0]])
        band = P[P[:, 2] < 0.12 * h]
        if len(band) >= 15:
            lat = band[:, :2] @ v
            rs.setdefault(sym.lower(), []).append((np.quantile(lat, 0.98) - np.quantile(lat, 0.02)) / 2)
    return {s: (float(np.median(hs[s])), float(np.median(rs[s]))) for s in hs if s in rs}


def calibrate(frame: Frame, min_corners=12, max_normal_deg=3.0, slack=3) -> Calibration | None:
    """Calibration in the starting position, fail-closed.

    The labelling of the grid is open up to reflection, rotation and translation. The reflection is
    settled by the camera height (C_z > 0). Rotation and window are chosen by a joint search over 4 rotations and
    25 translations: the labelling wins under which the most visible pieces stand on their square of the
    starting position (foot points, up to `slack` outliers from hidden feet).
    Three independent quantities then check: square colour (a1 dark), depth normal against the pose and the
    full assignment with depth, in which every visible piece must stand exactly on its starting square.
    None instead of guessing if one of them contradicts or the choice is not unique.
    """
    det = detect_grid(frame.bgr)
    if det is None:
        return None
    _, pts, idx = det
    if len(idx) < min_corners:
        return None
    R, t = _pnp(frame.K, idx, pts)
    if (-R.T @ t)[2] < 0:                                      # camera below the board => mirrored
        idx = idx * np.array([-1, 1]); R, t = _pnp(frame.K, idx, pts)
    feet, syms = _feet_sq(frame, R, t)
    if len(syms) == 0:
        return None
    start = chess.Board()
    cands = []
    for rot in range(4):
        M = np.linalg.matrix_power(ROT90, rot)
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                sh = np.array([dx, dy])
                f = np.floor(feet @ M.T - sh + 4).astype(int)
                match = sum(1 for (fi, ri), sym in zip(f, syms)
                            if 0 <= fi < 8 and 0 <= ri < 8 and (p := start.piece_at(chess.square(fi, ri))) is not None
                            and p.symbol() == sym)
                inside = int(np.sum(np.abs(idx @ M.T - sh).max(1) <= 3))
                cands.append((match, inside, rot, dx, dy))
    cands.sort(reverse=True)
    best, second = cands[0], cands[1]
    if best[0] < len(syms) - slack or best[0] - second[0] < 3:  # too many off or not unique
        return None
    M = np.linalg.matrix_power(ROT90, best[2])
    idx = idx @ M.T - np.array(best[3:5])
    R, t = _pnp(frame.K, idx, pts)
    if _parity(frame, R, t) > 0:                                 # cross-check: a1 must be dark
        return None
    s, angle = _scale_from_depth(frame, R, t)
    if s is None or angle > max_normal_deg:
        return None
    t = s * t
    calib = Calibration(R, t, float(s), _measure_dims(frame, Cam(frame.K, R, t)))
    for k, sq in assign(frame, calib).items():                  # cross-check: all on their starting square
        p = start.piece_at(sq)
        if p is None or p.symbol() != frame.classes[k]:
            return None
    return calib


# --------------------------------------------------------------------------- Tracking
class Tracker:
    """Move tracking: legal moves + 'no move' as candidates, smallest NLL wins, gate instead of guessing.
    An unseen piece is cheap if its model at a plausible spot in the square would be less than
    hidden_frac visible, i.e. not covered by other masks. The model (truncated cone) is
    wider than real pieces, hence a fraction instead of the 25 px below which the detector loses pieces."""

    def __init__(self, calib: Calibration, sigma=0.3, gate=10.0, hidden_ok=1.0, hidden_bad=50.0, hidden_frac=0.15):
        self.calib, self.sigma, self.gate, self.hidden_frac = calib, sigma, gate, hidden_frac
        self.hidden_ok, self.hidden_bad = hidden_ok, hidden_bad
        self.board = chess.Board()
        self.centres = _centres(calib.square)

    def reset(self, fen: str) -> None:
        self.board = chess.Board(fen)

    def _hidden(self, cam, s, sym, fg, cache):
        """Visible fraction of the piece model on square s, minimised over its possible position in the square
        (centre and 8 points at a distance of 0.18 squares): the piece rarely stands exactly in the centre."""
        key = (s, sym.lower())
        if key not in cache:
            h, r = _dims(self.calib, sym)
            ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)
            best = np.inf
            for dx, dy in [(0, 0)] + [(0.18 * np.cos(a), 0.18 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)]:
                c = self.centres[s] + np.array([dx, dy]) * self.calib.square
                ring = lambda z, rr: np.column_stack((c[0] + rr * np.cos(ang), c[1] + rr * np.sin(ang), np.full(len(ang), z)))
                hull = cv.convexHull(cam.project(np.vstack((ring(0, r), ring(h, 0.55 * r)))).astype(np.float32))
                sil = np.zeros(fg.shape, np.uint8)
                cv.fillPoly(sil, [np.round(hull).astype(np.int32)], 1)
                best = min(best, np.count_nonzero(sil.astype(bool) & ~fg) / max(np.count_nonzero(sil), 1))
            cache[key] = best
        return cache[key]

    def _nll(self, cand: chess.Board, obs, cam, fg, cache):
        groups = {}
        for s, p in cand.piece_map().items():
            groups.setdefault(p.symbol(), ([], []))[0].append(s)
        for xy, sym in obs:
            groups.setdefault(sym, ([], []))[1].append(xy)
        total = 0.0
        for sym, (sqs, xys) in groups.items():
            if len(xys) > len(sqs):
                return np.inf
            used = set()
            if xys:
                Cm = np.array([[np.sum(((xy - self.centres[s]) / self.calib.square) ** 2) / (2 * self.sigma ** 2)
                                for s in sqs] for xy in xys])
                r, c = linear_sum_assignment(Cm)
                total += Cm[r, c].sum(); used = set(c)
            for j, s in enumerate(sqs):
                if j not in used:
                    total += self.hidden_ok if self._hidden(cam, s, sym, fg, cache) < self.hidden_frac else self.hidden_bad
        return total

    def step(self, frame: Frame) -> str | None:
        cam = Cam(frame.K, self.calib.R, self.calib.t)
        pos = piece_positions(frame, self.calib)
        obs = [(pos[k], frame.classes[k]) for k in pos]
        fg, cache = frame.labels > 0, {}
        best, best_m = np.inf, None
        for m in [None] + list(self.board.legal_moves):
            cand = self.board.copy(stack=False)
            if m is not None:
                cand.push(m)
            nll = self._nll(cand, obs, cam, fg, cache)
            if nll < best:
                best, best_m = nll, m
        if best > self.gate:
            return None
        if best_m is not None:
            self.board.push(best_m)
        return self.board.fen()
