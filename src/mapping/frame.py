"""
One observation of the board: colour image, depth and the piece masks with their classes.

The same fields as tests/bench/sim.Frame, so the benchmark scenes run through the production
code unchanged. ChessBoardMapping3D builds a BoardFrame from a Willy RGBDFrame and the
detector's Detections.
"""
from dataclasses import dataclass, field

import numpy as np

# Detector class names (data.yaml of the piece dataset) -> FEN symbols.
CLASS_TO_SYMBOL = {
    "white pawn": "P", "white rook": "R", "white knight": "N", "white bishop": "B", "white queen": "Q", "white king": "K",
    "black pawn": "p", "black rook": "r", "black knight": "n", "black bishop": "b", "black queen": "q", "black king": "k",
}
SYMBOL_TO_CLASS = {symbol: name for name, symbol in CLASS_TO_SYMBOL.items()}
# The dataset's class order (data.yaml): index i of a YOLO label is CLASSES[i].
CLASSES = tuple(CLASS_TO_SYMBOL)


@dataclass
class BoardFrame:
    """
    What the pipeline sees in one view.

    Attributes:
        bgr (np.ndarray): Colour image, (H, W, 3) uint8, already undistorted.
        K (np.ndarray): Camera matrix of the undistorted image, (3, 3).
        depth (np.ndarray | None): z-depth in metres aligned to bgr, (H, W) float32, 0 = invalid.
        labels (np.ndarray | None): Piece masks, (H, W) int32, 0 = no piece, k = piece k.
        classes (dict[int, str]): Piece k -> FEN symbol ('P' white pawn, 'n' black knight).
        scores (dict[int, float]): Piece k -> detector confidence, if known.
    """
    bgr: np.ndarray
    K: np.ndarray
    depth: np.ndarray | None = None
    labels: np.ndarray | None = None
    classes: dict[int, str] = field(default_factory=dict)
    scores: dict[int, float] = field(default_factory=dict)


def labels_from_masks(shape: tuple[int, int], masks: list[np.ndarray]) -> np.ndarray:
    """
    Merge boolean masks into one label image, piece k = masks[k - 1].

    Where masks overlap, the smaller one wins: it is usually the piece in front, whose
    silhouette cuts into the one behind it.
    """
    labels = np.zeros(shape, np.int32)
    order = sorted(range(len(masks)), key=lambda i: -int(np.count_nonzero(masks[i])))
    for i in order:
        labels[masks[i]] = i + 1
    return labels
