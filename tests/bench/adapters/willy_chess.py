"""Adapter: die Produktiv-Implementierung aus src/ gegen den Benchmark.

    cd tests
    python -m bench.run --adapter bench/adapters/willy_chess.py --baseline <baseline_local.json>
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench.sim import Calibration, Frame  # noqa: E402
from src.board.grid import board_homography as _board_homography  # noqa: E402
from src.board.grid import detect_corners  # noqa: E402
from src.game.tracker import MoveTracker  # noqa: E402
from src.mapping.calibration import calibrate as _calibrate  # noqa: E402
from src.mapping.geometry import BoardCalibration  # noqa: E402
from src.mapping.pieces import assign as _assign  # noqa: E402

NAME = "willy_chess"
# Kosten einer falsch erkannten Figurenklasse im Tracking (float("inf") = harter Ausschluss wie die Referenz).
CLASS_COST = float(os.environ.get("WILLY_CHESS_CLASS_COST", "4.0"))


def _calib(calib: Calibration | BoardCalibration) -> BoardCalibration:
    if isinstance(calib, BoardCalibration):
        return calib
    return BoardCalibration(np.asarray(calib.R), np.asarray(calib.t), float(calib.square), calib.dims)


def corners(gray: np.ndarray) -> np.ndarray:
    return detect_corners(gray)


def board_homography(bgr: np.ndarray) -> np.ndarray | None:
    return _board_homography(bgr)


def calibrate(frame: Frame) -> BoardCalibration | None:
    return _calibrate(frame)


def assign(frame: Frame, calib: Calibration) -> dict[int, int]:
    return _assign(frame, _calib(calib))


class Tracker:
    """Der Simulator rendert nach einer Umwandlung die neue Figur, also promotion_mode='piece_swapped'."""

    def __init__(self, calib: Calibration):
        self.tracker = MoveTracker(_calib(calib), class_cost=CLASS_COST, promotion_mode="piece_swapped")

    def reset(self, fen: str) -> None:
        self.tracker.reset(fen)

    def step(self, frame: Frame) -> str | None:
        return self.tracker.step(frame).fen
