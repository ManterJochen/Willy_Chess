"""Referenz Stufe 1-2: Ecken, Gitter, Homographie, 8x8-Fenster (wie im Dokument)."""
from __future__ import annotations

import cv2 as cv
import numpy as np

from bench.reference.frozen.saddle import SaddlePointDetector
from bench.reference.frozen.symetric_solver import ChessboardSolver

CRIT = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 50, 1e-3)


def _gray(img):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY) if img.ndim == 3 else img


def _h(Hm, x):
    return cv.perspectiveTransform(np.asarray(x, np.float64).reshape(-1, 1, 2), Hm).reshape(-1, 2)


def corners(gray: np.ndarray) -> np.ndarray:
    """Sattelpunkte (-det H) + Förstner-Subpixel; Fenster passt sich der Feldgröße im Bild an."""
    p = SaddlePointDetector(gray, 0, True).detect()
    if len(p) < 2:
        return p
    nn = np.sort(np.linalg.norm(p[:, None] - p[None], axis=2), axis=1)[:, 1]
    w = int(max(2, round(0.12 * np.median(nn))))
    q = p.astype(np.float32).reshape(-1, 1, 2).copy()
    cv.cornerSubPix(gray, q, (w, w), (-1, -1), CRIT)
    return q.reshape(-1, 2).astype(np.float64)


def _axes(gray, pts, L, B):
    """Seite oder Diagonale: Mitte zweier Nachbarecken liegt auf einer Kante (Achse) oder im Feld."""
    g = cv.GaussianBlur(gray.astype(np.float32), (0, 0), 1.0)
    gx, gy = cv.Sobel(g, cv.CV_32F, 1, 0), cv.Sobel(g, cv.CV_32F, 0, 1)
    h, w = g.shape
    idx = np.round(L).astype(int)
    ok = np.linalg.norm(L - idx, axis=1) < 0.25
    lut = {tuple(k): i for i, k in zip(np.nonzero(ok)[0], idx[ok])}
    E = [np.array(e) for e in ((1, 0), (0, 1), (1, 1), (1, -1))]
    score = []
    for e in E:
        vals = []
        for k, i in lut.items():
            j = lut.get((k[0] + e[0], k[1] + e[1]))
            if j is None:
                continue
            d = pts[j] - pts[i]; n = np.linalg.norm(d)
            m = 0.5 * (pts[i] + pts[j]); x, y = int(round(m[0])), int(round(m[1]))
            if n < 1e-6 or not (0 <= x < w and 0 <= y < h):
                continue
            gv = np.array([gx[y, x], gy[y, x]]); t = d / n
            vals.append(abs(gv[0] * t[1] - gv[1] * t[0]) - abs(gv @ t))
        score.append(np.median(vals) if len(vals) >= 3 else -np.inf)
    best, M = -np.inf, np.eye(2)
    for a in range(4):
        for b in range(a + 1, 4):
            Mc = np.array([E[a], E[b]], float)
            if abs(round(np.linalg.det(Mc))) != 1:
                continue
            if min(score[a], score[b]) > best:
                best, M = min(score[a], score[b]), Mc
    return L @ np.linalg.inv(M)


def _edge_energy(grad, Hm, cx, cy):
    h, w = grad.shape
    t = np.arange(-3.5, 4.0, 1.0)
    tot = []
    for k in range(-4, 5):
        for line in (np.column_stack((np.full(8, cx + k), cy + t)), np.column_stack((cx + t, np.full(8, cy + k)))):
            p = _h(Hm, line)
            x, y = np.rint(p[:, 0]).astype(int), np.rint(p[:, 1]).astype(int)
            ok = (x >= 0) & (y >= 0) & (x < w) & (y < h)
            tot.append(np.median(grad[y[ok], x[ok]]) if ok.sum() >= 4 else 0.0)
    return float(np.sum(tot))


def _window(gray, Hm, idx, ok, band=3):
    """7x7-Fenster: meiste Inlier, bei Gleichstand (bis band darunter) Kantenenergie der 18 Linien."""
    g = cv.GaussianBlur(gray.astype(np.float32), (0, 0), 1.0)
    grad = cv.magnitude(cv.Sobel(g, cv.CV_32F, 1, 0), cv.Sobel(g, cv.CV_32F, 0, 1))
    I = idx[ok].astype(int)
    lo, hi = I.min(0), I.max(0)
    cands = [(np.sum((np.abs(I[:, 0] - cx) <= 3) & (np.abs(I[:, 1] - cy) <= 3)), cx, cy)
             for cx in range(lo[0] - 3, hi[0] + 4) for cy in range(lo[1] - 3, hi[1] + 4)]
    best_n = max(c[0] for c in cands)
    _, cx, cy = max((_edge_energy(grad, Hm, cx, cy), cx, cy) for n, cx, cy in cands if n >= best_n - band)
    return Hm @ np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1.0]])


def detect_grid(img, thr_px=2.0):
    """Gitter -> Bild-Homographie (innere Ecken bei -3..3) plus Inlier-Ecken und ihre Indizes."""
    gray = _gray(img)
    pts = corners(gray)
    if len(pts) < 8:
        return None
    B, L, _, _ = ChessboardSolver(pts)._get_lattice_and_reproject(pts, 2)
    if abs(np.linalg.det(B)) < 1e-6:
        return None
    L = _axes(gray, pts, L, B)
    Hm, _ = cv.findHomography(np.round(L).astype(np.float32), pts.astype(np.float32), cv.RANSAC, thr_px,
                              maxIters=5000, confidence=0.999)
    if Hm is None:
        return None
    idx = np.round(_h(np.linalg.inv(Hm), pts))
    ok = np.linalg.norm(_h(Hm, idx) - pts, axis=1) < thr_px
    if ok.sum() < 6:
        return None
    Hm = _window(gray, Hm, idx, ok)
    idx = np.round(_h(np.linalg.inv(Hm), pts))
    ok = (np.linalg.norm(_h(Hm, idx) - pts, axis=1) < thr_px) & (np.abs(idx).max(1) <= 3)
    return Hm, pts[ok], idx[ok]


def board_homography(bgr: np.ndarray) -> np.ndarray | None:
    det = detect_grid(bgr)
    return None if det is None else np.linalg.inv(det[0])
