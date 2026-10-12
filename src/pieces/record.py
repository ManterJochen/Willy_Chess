"""
Training data from real games: every confirmed view as a YOLO image and label file.

Each box is the detector's mask, its class comes from the tracked position, so a piece the detector
named wrongly is labelled right. A piece the detector missed gets no box: look through the images
before training. data.yaml names no validation split on purpose: Willy's DetectorTraining then holds
out a fixed share of the images by a hash of their names, so a later recording does not move them.
"""
from pathlib import Path

import cv2 as cv
import numpy as np

from src.mapping.frame import CLASSES, SYMBOL_TO_CLASS, BoardFrame


class Recorder:
    """Writes views in the layout DetectorTraining reads (data.yaml, images/, labels/)."""

    def __init__(self, out_dir: str | Path):
        self.out = Path(out_dir)
        (self.out / "images").mkdir(parents=True, exist_ok=True)
        (self.out / "labels").mkdir(parents=True, exist_ok=True)
        names = "\n".join(f'  {i}: "{name}"' for i, name in enumerate(CLASSES))
        (self.out / "data.yaml").write_text(
            f"path: {self.out.resolve()}\ntrain: images\nnc: {len(CLASSES)}\nnames:\n{names}\n")
        self.count = len(list((self.out / "images").glob("*.png")))

    def save(self, frame: BoardFrame, truth: dict[int, str]) -> Path:
        """
        Write one view.

        Args:
            frame (BoardFrame): The view with the detector's masks.
            truth (dict[int, str]): Piece label -> FEN symbol of the piece it really is.

        Returns:
            Path: The image written.
        """
        h, w = frame.labels.shape
        lines = []
        for k, symbol in truth.items():
            ys, xs = np.nonzero(frame.labels == k)
            if len(xs):
                x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
                cls = CLASSES.index(SYMBOL_TO_CLASS[symbol])
                lines.append(f"{cls} {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")
        image = self.out / "images" / f"{self.count:06d}.png"
        cv.imwrite(str(image), frame.bgr)
        (self.out / "labels" / f"{self.count:06d}.txt").write_text("".join(f"{line}\n" for line in lines))
        self.count += 1
        return image
