"""
From the logs of real games to a training set: images from the robot's own point of view, labelled by the game.

The game knows which piece stands where from the rules, so a view gets its labels whatever the detector made of it. That
is the point: a bishop taken for a knight is labelled a bishop, a piece the detector missed gets a box from the depth,
and a move only a click could settle comes along too. Those are the views the detector needs most, and they count
--hard-repeat times in the training split.

A label is only as good as the position behind it, so the depth has to agree, which owes the detector nothing: whatever
stands on the board must be explained by a piece of the position, and a piece the detector did not see must show in the
depth. A view that fails (a hand, a piece on the wrong square before it was taken back) goes to review/, not to training.

Whole games go to validation, every --val-every-th: views of one game are nearly the same, and split by image they would
stand on both sides and flatter the score. Views are cropped as the game crops them for the detector.

    python -m src.pieces.export [--logs data/game_logs] [--out data/cell_dataset] [--val-every 5] [--hard-repeat 3]
    python -m src.pieces.train --data data/cell_dataset --base-model docs/weights/chess_pieces_rtdetr/last --out ...
"""
import argparse
import json
import shutil
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import chess
import cv2 as cv
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.game.config import REPO_ROOT
from src.game.game import INTRUDER_PX
from src.mapping.frame import CLASSES, SYMBOL_TO_CLASS, BoardFrame
from src.mapping.geometry import BoardCalibration, BoardCamera, square_centres
from src.mapping.pieces import OFF_BOARD_MARGIN, piece_positions
from src.pieces.record import LOG_NAME, LoggedGame, load_view, read_game

REPORT = "export_report.json"
SPLITS = ("train", "val", "review", "preview")
PIECE_LIMIT = 0.7           # squares: a detection further from a piece of the position is not that piece
PIECE_ROOM = 0.2            # squares: how far off its centre a piece not seen may stand
ABOVE_BOARD = 0.015         # metres: what stands higher than this has to be a piece (BoardObservation.intruders)
ON_THE_BOARD = 0.008        # metres: the depth of a piece the detector missed
HIDDEN_BELOW = 0.15         # visible share of a piece below which it needs no label: nobody could see it
MIN_DEPTH_PX = 30           # the biggest connected part of a piece not seen. Simulated: at least 93 px and 17 % of its
MIN_DEPTH_SHARE = 0.1       # visible silhouette, against at most 7 px and 1 % on a square that is really empty
HARD_ANSWERS = ("clicked", "forced")


@dataclass
class ViewLabels:
    """
    What the position says about one view.

    Attributes:
        boxes (list[tuple[str, int, int, int, int]]): FEN symbol and x0, y0, x1, y1 in image pixels.
        off_board (list[tuple[int, int, int, int]]): Detections beside the board: no piece of the position, but maybe a
            captured one, so neither piece nor background.
        problems (list[str]): Why the position and the depth disagree; a view with any goes to review.
        wrong_type (int): Pieces the detector named wrongly.
        missed (int): Visible pieces the detector did not find.
        false (int): Detections on the board where the position has no piece.
    """
    boxes: list[tuple[str, int, int, int, int]] = field(default_factory=list)
    off_board: list[tuple[int, int, int, int]] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    wrong_type: int = 0
    missed: int = 0
    false: int = 0

    @property
    def hard(self) -> bool:
        return bool(self.wrong_type or self.missed or self.false)


def _box(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _largest_part(mask: np.ndarray) -> np.ndarray:
    count, parts, stats, _ = cv.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return parts == 1 + int(np.argmax(stats[1:, cv.CC_STAT_AREA])) if count > 1 else mask


def _heights(frame: BoardFrame, cam: BoardCamera, square: float) -> tuple[np.ndarray, np.ndarray]:
    """Per pixel: height above the board in metres (NaN in a depth hole), and whether it lies over the 8x8 squares."""
    height, over = np.full(frame.depth.shape, np.nan), np.zeros(frame.depth.shape, bool)
    ys, xs = np.nonzero(frame.depth > 0)
    P = cam.board_points_at(frame.depth, xs, ys)
    height[ys, xs] = P[:, 2]
    over[ys, xs] = np.abs(P[:, :2]).max(axis=1) <= 4 * square
    return height, over


def _silhouette(cam: BoardCamera, shape: tuple[int, int], xy: np.ndarray, h: float, r: float) -> np.ndarray:
    """Where a piece standing at xy can show in the image, its centre known to a fifth of a square: a cone frustum."""
    angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    rings = [np.column_stack((xy[0] + rr * np.cos(angles), xy[1] + rr * np.sin(angles), np.full(len(angles), z)))
             for z, rr in ((0.0, r), (1.05 * h, 0.55 * r))]
    hull = cv.convexHull(cam.project(np.vstack(rings)).astype(np.float32))
    mask = np.zeros(shape, np.uint8)
    cv.fillPoly(mask, [np.round(hull).astype(np.int32)], 1)
    return mask.astype(bool)


def label_view(frame: BoardFrame, calib: BoardCalibration, physical: dict[int, str]) -> ViewLabels:
    """
    The labels of a view from the position it showed.

    Args:
        frame (BoardFrame): The view with the detector's masks.
        calib (BoardCalibration): Its board pose.
        physical (dict[int, str]): Square -> piece that really stood there.

    Returns:
        ViewLabels: A box per visible piece, or the reason this view must not be trained on.
    """
    out = ViewLabels()
    if frame.depth is None or frame.labels is None:
        out.problems.append("no depth to check the position against")
        return out
    square = calib.square
    cam = BoardCamera(frame.K, calib.R, calib.t)
    centres = square_centres(square)
    positions = piece_positions(frame, calib)
    on_board = [k for k, xy in positions.items() if np.abs(xy / square).max() <= 4 + OFF_BOARD_MARGIN]
    out.off_board = [_box(frame.labels == k) for k in positions if k not in on_board]

    # Each detection to the piece it is, by place alone: the type and the colour are what the position corrects.
    squares = sorted(physical)
    matched: dict[int, int] = {}
    if on_board and squares:
        X = np.array([positions[k] for k in on_board]) / square
        distance = ((X[:, None] - centres[squares][None] / square) ** 2).sum(axis=-1)
        other_colour = np.array([[frame.classes[k].isupper() != physical[sq].isupper() for sq in squares] for k in on_board])
        for row, col in zip(*linear_sum_assignment(distance + 0.25 * other_colour)):
            if distance[row, col] <= PIECE_LIMIT ** 2:
                matched[squares[col]] = on_board[row]
    out.false = len(on_board) - len(matched)

    seen = np.isin(frame.labels, list(matched.values()))
    height, over = _heights(frame, cam, square)
    explained = cv.dilate(seen.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    for sq, symbol in physical.items():
        h, r = calib.piece_dims(symbol)
        hull = _silhouette(cam, frame.labels.shape, centres[sq], h, r + PIECE_ROOM * square)
        explained |= hull
        if sq in matched:
            k = matched[sq]
            out.boxes.append((symbol, *_box(frame.labels == k)))
            out.wrong_type += frame.classes[k] != symbol
            continue
        visible = hull & ~seen
        if np.count_nonzero(visible) < HIDDEN_BELOW * max(np.count_nonzero(hull), 1):
            continue
        with np.errstate(invalid="ignore"):
            piece = visible & (height > ON_THE_BOARD) & (height < 1.3 * h)
        part = _largest_part(piece)         # the edges of its neighbours are many small parts, a piece is one
        if np.count_nonzero(part) < max(MIN_DEPTH_PX, MIN_DEPTH_SHARE * np.count_nonzero(visible)):
            out.problems.append(f"the position has a {symbol} on {chess.square_name(sq)}, nothing shows there")
            continue
        out.boxes.append((symbol, *_box(part)))
        out.missed += 1

    with np.errstate(invalid="ignore"):
        unexplained = ((height > ABOVE_BOARD) & over & ~explained).astype(np.uint8)
    pixels = int(cv.morphologyEx(unexplained, cv.MORPH_OPEN, np.ones((5, 5), np.uint8)).sum())
    if pixels > INTRUDER_PX:
        out.problems.append(f"{pixels} pixels stand on the board where the position has no piece")
    return out


def calibration_of(game: LoggedGame, number: int, calib: BoardCalibration | None,
                   camera_to_base: np.ndarray | None) -> BoardCalibration | None:
    """A view's board pose; for a refused calibration from the board the game locked next, as seen from its camera pose."""
    if calib is not None or camera_to_base is None:
        return calib
    for later in sorted(n for n in game.views if n > number and game.views[n].get("square")):
        _, found, _ = load_view(game, later)
        with np.load(game.folder / "views" / f"{later:06d}.npz") as arrays:
            if found is None or "board_to_base" not in arrays.files:
                continue
            board_to_camera = np.linalg.inv(camera_to_base) @ arrays["board_to_base"]
        return BoardCalibration(board_to_camera[:3, :3], board_to_camera[:3, 3], found.square, found.dims)
    return None


def crop_view(frame: BoardFrame, labels: ViewLabels) -> tuple[np.ndarray, list[str]]:
    """
    The view as the detector gets it in a game, and its YOLO label lines. Whatever is neither piece nor background (a
    detection beside the board, a piece mostly cut off by the crop) is painted over, so it teaches nothing.
    """
    h, w = frame.bgr.shape[:2]
    x0, y0, x1, y1 = frame.roi or (0, 0, w, h)
    crop = frame.bgr[y0:y1, x0:x1].copy()
    ch, cw = crop.shape[:2]

    def paint(box: tuple[int, int, int, int]) -> None:
        a, b, c, d = max(box[0] - x0, 0), max(box[1] - y0, 0), min(box[2] - x0, cw), min(box[3] - y0, ch)
        if c > a and d > b:
            crop[b:d, a:c] = crop[b:d, a:c].mean(axis=(0, 1))

    for box in labels.off_board:
        paint(box)
    lines = []
    for symbol, a, b, c, d in labels.boxes:
        left, top, right, bottom = max(a - x0, 0), max(b - y0, 0), min(c - x0, cw), min(d - y0, ch)
        if right <= left or bottom <= top or (right - left) * (bottom - top) < 0.5 * (c - a) * (d - b):
            paint((a, b, c, d))
            continue
        lines.append(f"{CLASSES.index(SYMBOL_TO_CLASS[symbol])} {(left + right) / 2 / cw:.6f} {(top + bottom) / 2 / ch:.6f} "
                     f"{(right - left) / cw:.6f} {(bottom - top) / ch:.6f}")
    return crop, lines


def _drawn(image: np.ndarray, lines: list[str]) -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]
    for line in lines:
        cls, cx, cy, bw, bh = line.split()
        a, b = int((float(cx) - float(bw) / 2) * w), int((float(cy) - float(bh) / 2) * h)
        c, d = int((float(cx) + float(bw) / 2) * w), int((float(cy) + float(bh) / 2) * h)
        cv.rectangle(out, (a, b), (c, d), (0, 255, 255), 1)
        cv.putText(out, CLASSES[int(cls)].replace("white ", "w ").replace("black ", "b "), (a, max(b - 3, 10)),
                   cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    return out


def games_in(logs: Path) -> list[Path]:
    return sorted(p.parent for p in logs.glob(f"*/{LOG_NAME}"))


def validation_games(games: list[Path], every: int) -> set[Path]:
    """Every n-th game, so a game keeps its side as more are played; with fewer than n the newest one, if not the only one."""
    return set(games[every - 1::every]) or ({games[-1]} if len(games) > 1 else set())


def _prepare(out: Path) -> None:
    """Refuse a folder this program did not write: it may hold the training set of a whole project."""
    if out.exists() and any(out.iterdir()) and not (out / REPORT).exists():
        raise ValueError(f"{out} holds something else; name a new folder with --out")
    for name in SPLITS:
        shutil.rmtree(out / name, ignore_errors=True)
    for split in ("train", "val"):
        (out / split / "images").mkdir(parents=True)
        (out / split / "labels").mkdir(parents=True)
    (out / "review").mkdir()
    (out / "preview").mkdir()


def _write(out: Path, split: str, stem: str, image: np.ndarray, lines: list[str], repeat: int) -> None:
    text = "".join(f"{line}\n" for line in lines)
    for copy in range(repeat):
        name = stem if copy == 0 else f"{stem}_dup{copy}"
        if copy == 0:
            cv.imwrite(str(out / split / "images" / f"{name}.png"), image)
        else:
            shutil.copyfile(out / split / "images" / f"{stem}.png", out / split / "images" / f"{name}.png")
        (out / split / "labels" / f"{name}.txt").write_text(text, encoding="utf-8")


def export(logs: str | Path, out: str | Path, val_every: int = 5, hard_repeat: int = 3) -> dict:
    """
    Write the training set.

    Args:
        logs (str | Path): The folder with one folder per game (record.out_dir).
        out (str | Path): The dataset folder, in the layout DetectorTraining reads.
        val_every (int): Every n-th game goes to validation.
        hard_repeat (int): How often a view the detector got wrong counts in the training split.

    Returns:
        dict: What was written, as in export_report.json.
    """
    logs, out = Path(logs), Path(out)
    games = games_in(logs)
    _prepare(out)
    val = validation_games(games, val_every)
    report = {"games": [g.name for g in games], "validation_games": sorted(g.name for g in val),
              "train": 0, "val": 0, "hard": 0, "near_duplicates": 0, "not_settled": 0, "review": 0,
              "wrong_type": 0, "missed": 0, "false": 0, "boxes": Counter(), "review_reasons": []}
    for folder in games:
        game = read_game(folder)
        split = "val" if folder in val else "train"
        kept = set()
        for number in sorted(game.views):
            if number not in game.truth:
                report["not_settled"] += 1
                continue
            physical, how = game.truth[number]
            stem = f"{game.name}_{number:06d}"
            try:
                frame, calib, camera_to_base = load_view(game, number)
            except (FileNotFoundError, OSError) as err:
                report["review_reasons"].append(f"{stem}: {err}")
                report["review"] += 1
                continue
            calib = calibration_of(game, number, calib, camera_to_base)
            labels = label_view(frame, calib, physical) if calib is not None else ViewLabels(problems=["no board pose"])
            if labels.problems:
                report["review"] += 1
                reason = "; ".join(labels.problems)
                report["review_reasons"].append(f"{stem} ({how}): {reason}")
                cv.imwrite(str(out / "review" / f"{stem}.jpg"), frame.bgr)
                continue
            refused = not game.views[number].get("square")     # a calibration that failed on this frame
            hard = labels.hard or how in HARD_ANSWERS or number in game.doubted or refused
            key = (frozenset(physical.items()), game.views[number].get("look", 0))
            if not hard and key in kept:
                report["near_duplicates"] += 1
                continue
            kept.add(key)
            image, lines = crop_view(frame, labels)
            _write(out, split, stem, image, lines, hard_repeat if hard and split == "train" else 1)
            report[split] += 1
            report["hard"] += hard
            report["wrong_type"] += labels.wrong_type
            report["missed"] += labels.missed
            report["false"] += labels.false
            report["boxes"].update(CLASSES[int(line.split()[0])] for line in lines)
            if hard:
                cv.imwrite(str(out / "preview" / f"{stem}.jpg"), _drawn(image, lines))
    names = "\n".join(f'  {i}: "{name}"' for i, name in enumerate(CLASSES))
    val_line = "val: val/images\n" if report["val"] else ""
    (out / "data.yaml").write_text(f"train: train/images\n{val_line}nc: {len(CLASSES)}\nnames:\n{names}\n", encoding="utf-8")
    report["boxes"] = dict(report["boxes"])
    (out / REPORT).write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "review" / "reasons.txt").write_text("\n".join(report["review_reasons"]) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Turn the logs of real games into a training set.")
    parser.add_argument("--logs", type=Path, default=REPO_ROOT / "data" / "game_logs", help="record.out_dir")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "cell_dataset")
    parser.add_argument("--val-every", type=int, default=5, help="every n-th game goes to validation (default: 5)")
    parser.add_argument("--hard-repeat", type=int, default=3,
                        help="how often a view the detector got wrong counts in training (default: 3)")
    args = parser.parse_args(argv)
    report = export(args.logs, args.out, args.val_every, args.hard_repeat)
    print(f"{len(report['games'])} game(s): {report['train']} training and {report['val']} validation views in {args.out}")
    print(f"  {report['hard']} views the detector got wrong ({report['wrong_type']} wrong types, {report['missed']} missed "
          f"pieces, {report['false']} false detections), drawn in preview/")
    print(f"  left out: {report['near_duplicates']} nearly the same as another, {report['not_settled']} never settled, "
          f"{report['review']} where the depth disagrees (review/reasons.txt)")
    if not report["val"]:
        print("  no game for validation yet: with one game the training splits by image, and its score flatters")
    return 0 if report["train"] or report["val"] else 1


if __name__ == "__main__":
    sys.exit(main())
