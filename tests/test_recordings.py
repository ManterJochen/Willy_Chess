"""The board detection on real webcam recordings (data/demo, made with `python -m src.board.demo --record`).

No ground truth is needed: a grid is right when at least `need` of its 49 predicted inner corners lie on a
detected saddle point; a grid shifted by one square puts 7 of them onto the board edge, where there are
none. Skipped where the recordings are missing (data/ is not in git).

    cd tests
    python -m unittest test_recordings -v
"""
import sys
import unittest
from pathlib import Path

import cv2 as cv
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.board.grid import detect_grid, transform  # noqa: E402
from src.board.saddle import SaddlePointDetector  # noqa: E402

# recording -> (corners a right grid supports, frames that must be right: measured 2026-10-12 minus a margin)
EXPECTED = {"leer": (44, 54), "grundstellung": (40, 57), "bewegt": (36, 114),
            "test": (28, 156)}         # mid-game, down to close-ups where tall pieces hide many corners: 164 of 210
INNER = np.array([(i, j) for j in range(-3, 4) for i in range(-3, 4)], float)


def right_frames(folder: Path, need: int) -> int:
    right = 0
    for path in sorted(folder.glob("*.png")):
        image = cv.imread(str(path))
        grid = detect_grid(image)
        if grid is None:
            continue
        saddles = SaddlePointDetector(image, 0, False).detect()
        inner = transform(grid.H_grid_to_image, INNER)
        right += int((np.linalg.norm(inner[:, None] - saddles[None], axis=2).min(axis=1) < 3.0).sum() >= need)
    return right


@unittest.skipUnless((REPO / "data" / "demo").is_dir(), "no recordings in data/demo")
class RecordingsTest(unittest.TestCase):
    def test_the_board_is_found_in_the_recordings(self):
        for name, (need, at_least) in EXPECTED.items():
            folder = REPO / "data" / "demo" / name
            if folder.is_dir():
                with self.subTest(recording=name):
                    self.assertGreaterEqual(right_frames(folder, need), at_least)


if __name__ == "__main__":
    unittest.main()
