"""
Detector test: does the camera see the pieces? No calibration, no robot motion besides the look pose.

The arm goes to its look pose, grabs one view, runs the detector (with segmentation) on the board crop
and on the whole image, prints what it found per piece type and saves an image with every mask and label.

    python -m src.game.detect_test [--willy-data DIR] [--threshold 0.3] [--no-move]
"""
import argparse
from collections import Counter
from pathlib import Path

import cv2 as cv
import numpy as np

from src.game.config import REPO_ROOT, load_config
from src.game.play import load_cell_tree, open_cell

START = Counter({"P": 8, "N": 2, "B": 2, "R": 2, "Q": 1, "K": 1,
                 "p": 8, "n": 2, "b": 2, "r": 2, "q": 1, "k": 1})


def draw(frame) -> np.ndarray:
    """The colour image with every piece mask outlined and labelled with symbol and score."""
    out = frame.bgr.copy()
    if frame.labels is None:
        return out
    for k, symbol in frame.classes.items():
        mask = (frame.labels == k).astype(np.uint8)
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        colour = (0, 0, 255) if symbol.isupper() else (255, 128, 0)
        cv.drawContours(out, contours, -1, colour, 2)
        ys, xs = np.nonzero(mask)
        if len(xs):
            text = f"{symbol} {frame.scores.get(k, 0):.2f}"
            org = (int(xs.min()), max(int(ys.min()) - 4, 12))
            cv.putText(out, text, org, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv.putText(out, text, org, cv.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
    return out


def report(name: str, frame) -> Counter:
    found = Counter(frame.classes.values())
    print(f"\n{name}: {sum(found.values())} pieces")
    for symbol in "KQRBNPkqrbnp":
        n, want = found.get(symbol, 0), START[symbol]
        flag = "" if n == want else f"   <- starting position has {want}"
        print(f"  {symbol}: {n}{flag}")
    if frame.scores:
        s = np.array(list(frame.scores.values()))
        print(f"  scores: min {s.min():.2f}, median {np.median(s):.2f}, max {s.max():.2f}")
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the piece detector on one camera view.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--willy-data", default=None, help="Willy's config directory of the cell (default: willy.data_dir)")
    parser.add_argument("--threshold", type=float, default=None, help="score threshold (default: detector.threshold)")
    parser.add_argument("--no-move", action="store_true", help="grab from where the arm stands now")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "detect_test")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.threshold is not None:
        config.detector.threshold = args.threshold

    tree = load_cell_tree(args.willy_data or config.willy.data_dir)
    if not tree.ok:
        print(tree)
        return tree.exit_code
    args.out.mkdir(parents=True, exist_ok=True)
    with open_cell(tree, config) as cell:
        if not args.no_move:
            cell.go_look()
        mapping = cell.mapping
        grabbed, _, _ = mapping._grab()
        cv.imwrite(str(args.out / "raw.png"), grabbed.bgr)
        if grabbed.depth is None or not np.any(grabbed.depth):
            print("WARNING: no depth in this frame (calibration and grasp widths need it)")
        else:
            valid = np.count_nonzero(grabbed.depth) / grabbed.depth.size
            print(f"depth: {valid:.0%} of the pixels valid")

        roi = mapping._grid_roi(grabbed)
        print(f"board crop: {roi if roi is not None else 'board grid not found, whole image only'}")
        if roi is not None:
            cropped = mapping._with_pieces(grabbed, roi=roi)
            report("board crop", cropped)
            cv.imwrite(str(args.out / "pieces_crop.png"), draw(cropped))
        whole = mapping._with_pieces(grabbed, roi=None)
        report("whole image", whole)
        cv.imwrite(str(args.out / "pieces_whole.png"), draw(whole))

    print(f"\nimages: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
