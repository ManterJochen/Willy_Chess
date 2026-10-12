"""The six test suites. Each scene has its own seed [suite, row, scene] and is thus
independent of what the adapter does or which subset (--quick) runs."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import chess
import cv2 as cv
import numpy as np

from bench import sim

FULL = {(i, j) for i in range(-3, 4) for j in range(-3, 4)}


@dataclass
class Row:
    name: str
    counts: dict[str, list[int]] = field(default_factory=dict)    # metric -> [k, n]
    values: dict[str, list[float]] = field(default_factory=dict)  # metric -> single values
    failures: list[int] = field(default_factory=list)             # indices of failed scenes

    def count(self, metric, ok, n=1):
        c = self.counts.setdefault(metric, [0, 0]); c[0] += int(ok); c[1] += n

    def value(self, metric, v):
        self.values.setdefault(metric, []).append(float(v))


class Hasher:
    """Fingerprint of the inputs: same inputs = same numbers with the same adapter."""
    def __init__(self):
        self.h = hashlib.sha256()

    def add(self, frame: sim.Frame):
        for a in (frame.bgr, frame.depth, frame.labels):
            if a is not None:
                self.h.update(np.ascontiguousarray(a).tobytes())
        self.h.update(repr(sorted(frame.classes.items())).encode())

    def hex(self):
        return self.h.hexdigest()[:16]


def _rng(suite_id, row, scene):
    return np.random.default_rng([suite_id, row, scene])


def _dump(dump: Path | None, suite, row, i, frame: sim.Frame, truth: dict):
    if dump is None:
        return
    dump.mkdir(parents=True, exist_ok=True)
    stem = dump / f"{suite}_{row}_{i:03d}"
    cv.imwrite(str(stem) + ".png", frame.bgr)
    np.savez_compressed(str(stem) + ".npz", bgr=frame.bgr, K=frame.K,
                        depth=frame.depth if frame.depth is not None else np.zeros(0),
                        labels=frame.labels if frame.labels is not None else np.zeros(0),
                        classes=np.array(repr(frame.classes)), truth=np.array(repr(truth)))


# --------------------------------------------------------------------------- 1 Corners
def suite_corners(ad, n, dump=None):
    rows, hs = [], Hasher()
    row = Row("0-45°")
    errs = []
    for i in range(n):
        rng = _rng(1, 0, i)
        Hm = sim.camera_squares(rng.uniform(0, 45), rng.uniform(0, 90), rng.uniform(12, 16), rng.uniform(-1, 1, 2))
        bgr = sim.render_board(rng, Hm, blur=rng.uniform(0.6, 1.2), noise=2.0)
        frame = sim.Frame(bgr, sim.K.copy()); hs.add(frame)
        gt = sim.inner_corners(Hm)
        p = np.asarray(ad.corners(cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)), float).reshape(-1, 2)
        found = 0
        if len(p):
            d = np.linalg.norm(gt[:, None] - p[None], axis=2)
            j = d.argmin(1); m = d[np.arange(len(gt)), j] < 3.0
            found = int(m.sum())
            errs += list(d[np.arange(len(gt)), j][m])
        row.count("corners found", found, len(gt))
        if found < len(gt):
            row.failures.append(i); _dump(dump, "corners", 0, i, frame, {"corners": gt.tolist()})
    e = np.array(errs)
    if len(e):
        row.value("RMSE px", np.sqrt(np.mean(e ** 2))); row.value("95% quantile px", np.percentile(e, 95))
    rows.append(row)
    return rows, hs.hex()


# --------------------------------------------------------------------------- 2 Board
def _eval_board(Hm, gt):
    if Hm is None or not np.all(np.isfinite(Hm)) or abs(np.linalg.det(Hm)) < 1e-12:
        return "wrong"
    u = cv.perspectiveTransform(gt.reshape(-1, 1, 2), np.asarray(Hm, np.float64)).reshape(-1, 2)
    if not np.all(np.isfinite(u)):
        return "wrong"
    r = np.round(u)
    if np.abs(u - r).max() > 0.25 or len({tuple(x) for x in r.astype(int)}) != 49:
        return "wrong"
    if {tuple(x) for x in (r - r.min(0) - 3).astype(int)} != FULL:
        return "wrong"
    return "correct" if {tuple(x) for x in r.astype(int)} == FULL else "shifted"


BOARD_ROWS = [("pieces 30-45°", dict(blobs=20), 30, 45), ("pieces 45-60°", dict(blobs=20), 45, 60),
              ("clutter 30-45°", dict(clutter=5), 30, 45), ("clutter 45-60°", dict(clutter=5), 45, 60)]


def suite_board(ad, n, dump=None):
    rows, hs = [], Hasher()
    for ri, (name, kw, lo, hi) in enumerate(BOARD_ROWS):
        row = Row(name)
        for i in range(n):
            rng = _rng(2, ri, i)
            Hm = sim.camera_squares(rng.uniform(lo, hi), rng.uniform(0, 90), rng.uniform(12, 16), rng.uniform(-1, 1, 2))
            bgr = sim.render_board(rng, Hm, blur=rng.uniform(0.6, 1.2), noise=2.0, **kw)
            frame = sim.Frame(bgr, sim.K.copy()); hs.add(frame)
            out = _eval_board(ad.board_homography(bgr), sim.inner_corners(Hm))
            row.count("board correct", out == "correct")
            row.count("shifted", out == "shifted")
            if out != "correct":
                row.failures.append(i); _dump(dump, "board", ri, i, frame, {"outcome": out})
        rows.append(row)
    return rows, hs.hex()


# --------------------------------------------------------------------------- 3 Calibration
def _calib_ok(truth: sim.Calibration, est: sim.Calibration) -> bool:
    """All 64 square centres land in the same square via the estimated pose (tolerance 0.25 squares)."""
    if est is None:
        return False
    c = np.array([sim.sq_center(s, truth.square) for s in range(64)])
    uv = sim.project(truth.R, truth.t, np.column_stack((c, np.zeros(64))))
    Hb = sim.K @ np.column_stack((est.R[:, 0], est.R[:, 1], est.t))
    p = np.column_stack((uv, np.ones(64))) @ np.linalg.inv(Hb).T
    xy = p[:, :2] / p[:, 2:] / est.square
    want = c / truth.square
    return bool(np.all(np.abs(xy - want).max(1) < 0.25))


def suite_calib(ad, n, dump=None):
    rows, hs = [], Hasher()
    for ri, tilt in enumerate((35, 50, 60, 70)):
        row = Row(f"{tilt}°")
        for i in range(n):
            rng = _rng(3, ri, i)
            setup = sim.Setup(*sim.look_at(tilt + rng.uniform(-3, 3), rng.uniform(0, 360)),
                              square=rng.uniform(0.045, 0.065), scale=rng.uniform(0.85, 1.15))
            board = chess.Board()
            frame, truth = sim.render_pieces(rng, setup, board, sim.jitter_offsets(rng, board))
            hs.add(frame)
            est = ad.calibrate(frame)
            ok = _calib_ok(truth.calibration, est)
            row.count("calibrated", ok)
            row.count("miscalibrated", est is not None and not ok)
            if est is not None and ok:
                row.value("square size error %", 100 * abs(est.square / setup.square - 1))
            full = False
            if est is not None:
                a = ad.assign(frame, est)
                full = len(truth.square_of) == truth.n_pieces and all(a.get(k) == s for k, s in truth.square_of.items())
            row.count("board complete", full)
            if not (ok and full):
                row.failures.append(i); _dump(dump, "calibration", ri, i, frame, {"square_of": truth.square_of, "square": setup.square})
        rows.append(row)
    return rows, hs.hex()


# --------------------------------------------------------------------------- 4 Mapping, 6 Stress
def _mapping_rows(ad, n, suite_id, label, specs, dump, spread):
    rows, hs = [], Hasher()
    for ri, (name, tilt, kw) in enumerate(specs):
        row = Row(name)
        for i in range(n):
            rng = _rng(suite_id, ri, i)
            kw2 = dict(kw); centre = kw2.pop("scale", 1.0)
            setup = sim.Setup(*sim.look_at(tilt + rng.uniform(-3, 3), rng.uniform(0, 360)),
                              scale=centre * rng.uniform(1 - spread, 1 + spread), **kw2)
            board = sim.random_position(rng, int(rng.integers(6, 70)))
            frame, truth = sim.render_pieces(rng, setup, board, sim.jitter_offsets(rng, board))
            hs.add(frame)
            a = ad.assign(frame, truth.calibration)
            good = [a.get(k) == s for k, s in truth.square_of.items()]
            row.count("piece correct", sum(good), len(good))
            full = len(truth.square_of) == truth.n_pieces and all(good)
            row.count("board complete", full)
            if not all(good):
                row.failures.append(i); _dump(dump, label, ri, i, frame, {"square_of": truth.square_of})
        rows.append(row)
    return rows, hs.hex()


def suite_mapping(ad, n, dump=None):
    specs = [(f"{t}°", t, {}) for t in (0, 20, 30, 35, 50, 60, 70)]
    return _mapping_rows(ad, n, 4, "mapping", specs, dump, spread=0.08)


def suite_stress(ad, n, dump=None):
    specs = [("none", 70, {}), ("mask +1 px", 70, dict(mask_px=1)), ("mask -1 px", 70, dict(mask_px=-1)),
             ("piece set +15 %", 70, dict(scale=1.15)), ("piece set -15 %", 70, dict(scale=0.85)),
             ("depth shifted 3 px", 70, dict(depth_shift=3)), ("board bulged 8 mm", 70, dict(warp=0.008)),
             ("board sagged 8 mm", 70, dict(warp=-0.008))]
    return _mapping_rows(ad, n, 6, "stress", specs, dump, spread=0.03)


# --------------------------------------------------------------------------- 5 Tracking
def _fen4(fen):
    return " ".join(chess.Board(fen).fen(en_passant="legal").split()[:4])


def suite_tracking(ad, n, dump=None, plies=50):
    """n = games per tilt. After a flagged or wrong move the harness resets to the truth
    (like a confirmation by the human) and counts the case."""
    rows, hs = [], Hasher()
    for ri, tilt in enumerate((50, 60, 70)):
        row = Row(f"{tilt}°")
        for g in range(n):
            rng = _rng(5, ri, g)
            setup = sim.Setup(*sim.look_at(tilt + rng.uniform(-3, 3), rng.uniform(0, 360)), scale=rng.uniform(0.92, 1.08))
            board = chess.Board()
            offs = sim.jitter_offsets(rng, board)
            tracker = ad.Tracker(setup.calibration())
            tracker.reset(board.fen())
            for ply in range(plies):
                moves = list(board.legal_moves)
                if not moves:
                    break
                prev, prev_offs = board.copy(), offs
                board.push(moves[rng.integers(len(moves))])
                offs = sim.jitter_offsets(rng, board, prev_offs, prev)
                frame, _ = sim.render_pieces(rng, setup, board, offs)
                hs.add(frame)
                out = tracker.step(frame)
                ok = out is not None and _fen4(out) == _fen4(board.fen())
                row.count("move correct", ok)
                row.count("flagged", out is None)
                row.count("silently wrong", out is not None and not ok)
                if not ok:
                    row.failures.append(g * 1000 + ply)
                    _dump(dump, "tracking", ri, g * 1000 + ply, frame, {"fen": board.fen(), "prev": prev.fen(), "out": out})
                    tracker.reset(board.fen())
        rows.append(row)
    return rows, hs.hex()


SUITES: dict[str, tuple[Callable, int, str]] = {
    # name: (function, scenes per row in the full run, required adapter function)
    "corners": (suite_corners, 40, "corners"),
    "board": (suite_board, 30, "board_homography"),
    "calibration": (suite_calib, 30, "calibrate"),
    "mapping": (suite_mapping, 60, "assign"),
    "tracking": (suite_tracking, 8, "Tracker"),
    "stress": (suite_stress, 20, "assign"),
}
