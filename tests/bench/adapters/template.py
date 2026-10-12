"""Template for your adapter. Copy it, e.g. to bench/adapters/mine.py, and connect the functions
to your code. Anything missing or raising NotImplementedError is skipped.

Conventions (see bench/sim.py):
- Board frame: origin board centre, x towards the h-file, y towards the 8th rank, z up, metres.
- Pose: X_cam = R @ X_board + t. Squares: python-chess (a1 = 0, h8 = 63).
- Frame.depth: z-depth in metres (0 = invalid). Frame.labels: 0 = no piece, k = piece k.
- Frame.classes: k -> FEN symbol ('P' white pawn, 'n' black knight, ...).
"""
from __future__ import annotations

import numpy as np

from bench.sim import Calibration, Frame

NAME = "mine"


def corners(gray: np.ndarray) -> np.ndarray:
    """Suite corners: inner board corners as an (N, 2) array in pixels."""
    raise NotImplementedError


def board_homography(bgr: np.ndarray) -> np.ndarray | None:
    """Suite board: 3x3 homography image -> grid. The 49 inner corners lie on integer
    grid points (i, j) with -3 <= i, j <= 3; rotation and reflection do not matter here. None = not found."""
    raise NotImplementedError


def calibrate(frame: Frame) -> Calibration | None:
    """Suite calibration: starting position, square size unknown (45-65 mm). Full pose with orientation
    (a1 bottom left as seen by White), square size in metres, optionally piece dimensions."""
    raise NotImplementedError


def assign(frame: Frame, calib: Calibration) -> dict[int, int]:
    """Suites mapping, stress, calibration: label k -> square 0..63 for every piece in frame.classes."""
    raise NotImplementedError


class Tracker:
    """Suite tracking: game from the starting position, one move per image."""

    def __init__(self, calib: Calibration):
        raise NotImplementedError

    def reset(self, fen: str) -> None:
        """Set the state to this position (start and after a flagged or wrong move)."""

    def step(self, frame: Frame) -> str | None:
        """Full FEN after the move, or None if unsure (counts as flagged, not as an error)."""
        return None
