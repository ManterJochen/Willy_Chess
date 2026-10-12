"""
Live view of the board detection the game uses (grid.py): saddle points, lattice and the projected
chessboard (visualize.py).

    python -m src.board.demo [--camera 0]
    python -m src.board.demo --record data/demo/empty     # also save every frame the detection sees
    python -m src.board.demo --replay data/demo/empty     # run on a recording instead of the camera
"""
import argparse
from pathlib import Path
from typing import Iterator

import cv2 as cv
import matplotlib
import numpy as np

matplotlib.use("Agg")       # the figure is rendered into the OpenCV window, no matplotlib window of its own

from src.board.grid import BASIS_CORNERS, detect_corners, detect_grid, transform  # noqa: E402
from src.board.symetric_solver import ChessboardSolver  # noqa: E402
from src.board.visualize import visualize_reconstruction  # noqa: E402


def frames(camera_id: int = 0, replay: Path | None = None) -> Iterator[np.ndarray]:
    """The camera's frames at 640 x 480, or the images of a recording in order."""
    if replay is not None:
        for path in sorted(Path(replay).glob("*.png")):
            yield cv.imread(str(path))
        return
    cap = cv.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Kamera mit ID {camera_id} konnte nicht geöffnet werden.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Kein Kamerabild empfangen.")
                return
            yield cv.resize(frame, (640, 480), interpolation=cv.INTER_AREA)
    finally:
        cap.release()


def run(camera_id: int = 0, record: Path | None = None, replay: Path | None = None) -> None:
    """
    Show the reconstruction of every frame. Press 'q' in the window to exit.

    Args:
        camera_id: Camera index (0 = default camera).
        record: Folder to save every frame to, exactly as the detection sees it.
        replay: Folder of a recording to run on instead of the camera.
    """
    if record is not None:
        Path(record).mkdir(parents=True, exist_ok=True)
    try:
        for i, image in enumerate(frames(camera_id, replay)):
            if record is not None:
                cv.imwrite(str(Path(record) / f"{i:05d}.png"), image)
            points = detect_corners(image)
            grid = detect_grid(image)
            H = grid_points = basis = None
            if grid is not None:
                H = grid.H_image_to_grid
                grid_points = np.round(transform(H, points))
                basis, _ = ChessboardSolver(points[:BASIS_CORNERS]).estimate_lattice(points[:BASIS_CORNERS])

            visualize_reconstruction(
                cv.cvtColor(image, cv.COLOR_BGR2RGB),      # matplotlib draws RGB, the camera gives BGR
                points,
                grid_points,
                H,
                basis)

            if cv.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cv.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live view of the board detection.")
    parser.add_argument("--camera", type=int, default=0, help="camera index (default: 0)")
    parser.add_argument("--record", type=Path, default=None, help="save every frame to this folder")
    parser.add_argument("--replay", type=Path, default=None, help="run on a recorded folder instead of the camera")
    args = parser.parse_args()
    run(args.camera, args.record, args.replay)
