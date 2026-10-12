"""Your code before the changes (src/board, commit f3d51b2), stages 1-2 only."""
import numpy as np

from bench.reference.frozen.saddle import SaddlePointDetector
from bench.reference.frozen.symetric_solver import ChessboardSolver

NAME = "original"


def corners(gray):
    return SaddlePointDetector(gray, 0, True).detect()


def board_homography(bgr):
    pts = SaddlePointDetector(bgr, 50, True).detect()
    solver = ChessboardSolver(lattice_points=pts)
    grid, _, _ = solver.estimate_chess_grid(lattice_points=pts)
    return solver.estimate_homography(pts, grid)
