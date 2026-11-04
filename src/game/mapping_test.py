"""
Mapping test: which square does every piece land on, and how far off is it?

Set up the starting position. The arm goes to its look pose, calibrates, then takes a few views
(observe, exactly as in the game). For every piece it prints the square it was assigned, the square it
stands on, and the error of the position estimates (foot point from the mask, axis from the depth, and
the grasp position the game uses), split into "along" (away from the camera) and "lateral" (sideways),
in millimetres. A type the detector gets wrong is counted apart: the square does not depend on it.

The pattern of the errors names the cause:
  along >> lateral, all the same sign  -> foot radius / piece height wrong, or depth biased
  lateral and along in one direction for every piece  -> board pose / square size off
  only the far ranks wrong  -> feet hidden behind the pieces in front (look pose too flat)
  depth "-" for many pieces  -> too little depth on the pieces (shiny), the foot point alone is used
  depth along rising over the files, free board height too  -> depth tilted against the camera pose

--all-looks takes one view from every look pose instead: is the whole board in the picture, does the board stay
where it is from every side (a jump is the hand-eye calibration), and how far does each estimate of a piece move
between the poses (that is its error, wherever the piece really stands). --save keeps every view as an .npz.

    python -m src.game.mapping_test [--views 3] [--no-move] [--all-looks] [--save]
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import chess
import cv2 as cv
import numpy as np

from src.game.config import REPO_ROOT, load_config
from src.game.play import load_cell_tree, open_cell
from src.mapping.frame import BoardFrame
from src.mapping.geometry import BoardCalibration, BoardCamera, square_centres
from src.mapping.pieces import depth_xy, foot_xy

START = {sq: p.symbol() for sq, p in chess.Board().piece_map().items()}
GRIP_ROOM_MM = 8.0      # robot.finger_clearance_mm: what the open jaws forgive along their closing direction


def nearest_start_square(xy: np.ndarray, centres: np.ndarray) -> int:
    """The start square a piece stands on, whatever type the detector takes it for."""
    return min(START, key=lambda sq: np.linalg.norm(xy - centres[sq]))


def split(err: np.ndarray, xy: np.ndarray, F: np.ndarray) -> tuple[float, float]:
    """Error (x, y) split into along the viewing direction and lateral, millimetres."""
    u = (xy - F) / max(np.linalg.norm(xy - F), 1e-12)
    v = np.array([-u[1], u[0]])
    return float(err @ u * 1000), float(err @ v * 1000)


def fmt(e: tuple[float, float] | None) -> str:
    return "      -      " if e is None else f"{e[0]:+6.1f}/{e[1]:+5.1f}"


def board_depth_profile(frame: BoardFrame, calib: BoardCalibration) -> tuple[list[float], list[float]]:
    """Median height of the free board per file a..h and per rank 1..8, millimetres: 0 where depth and pose agree."""
    ys, xs = np.nonzero((frame.depth > 0) & (frame.labels == 0))
    P = BoardCamera(frame.K, calib.R, calib.t).board_points_at(frame.depth, xs, ys)
    index = np.floor(P[:, :2] / calib.square + 4).astype(int)
    keep = (index >= 0).all(axis=1) & (index < 8).all(axis=1) & (np.abs(P[:, 2]) < 0.02)
    P, index = P[keep], index[keep]

    def medians(axis: int) -> list[float]:
        return [float(np.median(P[index[:, axis] == i, 2])) * 1000 if (index[:, axis] == i).any() else float("nan")
                for i in range(8)]

    return medians(0), medians(1)


def squares_out_of_view(frame: BoardFrame, calib: BoardCalibration) -> int:
    """Square centres outside the image. Every piece out of the picture costs the tracker 1 of its gate of 10."""
    centres = square_centres(calib.square)
    uv = BoardCamera(frame.K, calib.R, calib.t).project(np.column_stack((centres, np.zeros(64))))
    h, w = frame.bgr.shape[:2]
    return int(np.sum((uv[:, 0] < 0) | (uv[:, 1] < 0) | (uv[:, 0] >= w) | (uv[:, 1] >= h)))


def save_view(path: Path, obs) -> None:
    """One view with everything an estimator needs, to look at it again without the cell."""
    frame, calib = obs.frame, obs.calib
    np.savez_compressed(path, bgr=frame.bgr, depth=frame.depth, labels=frame.labels, K=frame.K, R=calib.R, t=calib.t,
                        square=calib.square, camera_to_base=obs.camera_to_base, board_to_base=obs.board_to_base,
                        classes=json.dumps(frame.classes), scores=json.dumps(frame.scores), dims=json.dumps(calib.dims))


def report_view(obs, name: str, totals: dict, by_square: dict, out: Path) -> None:
    """Print one view, piece by piece, add it to the totals and save the picture of it."""
    calib, frame = obs.calib, obs.frame
    cam = BoardCamera(frame.K, calib.R, calib.t)
    F = cam.foot
    centres = square_centres(calib.square)
    print(f"\n{name}: pose {obs.pose_status}, {obs.corners_used} corners, "
          f"residual {obs.corner_residual_px:.2f} px, {len(frame.classes)} pieces, "
          f"{squares_out_of_view(frame, calib)} of 64 squares out of the picture")
    print("  piece  got   true   ok  type   foot along/lat   depth along/lat   grasp along/lat")
    image = frame.bgr.copy()
    rows = []
    for k, symbol in frame.classes.items():
        mask = frame.labels == k
        h, r = calib.piece_dims(symbol)
        xf = foot_xy(cam, mask, r)
        xd = depth_xy(cam.board_points(frame.depth, mask), F, h, r) if frame.depth is not None else None
        xg = obs.grasp_positions[k]
        true = nearest_start_square(obs.positions[k], centres)
        got = obs.squares.get(k)
        ok, type_ok = got == true, START[true] == symbol
        ef = split(xf - centres[true], centres[true], F)
        ed = None if xd is None else split(xd - centres[true], centres[true], F)
        eg = split(xg - centres[true], centres[true], F)
        rank, file = chess.square_rank(true), chess.square_file(true)
        totals["ok"].append(ok)
        totals[f"rank{rank + 1}"].append(ok)
        totals["foot"].append(ef)
        totals["grasp"].append(eg)
        if ed is not None:
            totals["depth"].append(ed)
            totals["pawn_along" if symbol.lower() == "p" else "other_along"].append((file, ed[0]))
        if not type_ok:
            totals["confused"].append(f"{START[true]} seen as {symbol}")
        totals["type_ok"].append(type_ok)
        by_square[true].append(("foot", xf))
        by_square[true].append(("grasp", xg))
        if xd is not None:
            by_square[true].append(("depth", xd))
        rows.append((rank, file, f"  {symbol:>5}  {chess.square_name(got) if got is not None else '--':>3}   "
                                 f"{chess.square_name(true):>4}   {'yes' if ok else 'NO ':>3}  "
                                 f"{'ok ' if type_ok else START[true] + '? '} {fmt(ef)}    {fmt(ed)}    {fmt(eg)}"))
        ys, xs = np.nonzero(mask)
        colour = (0, 200, 0) if ok else (0, 0, 255)
        contours, _ = cv.findContours(mask.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(image, contours, -1, colour, 2)
        label = chess.square_name(got) if got is not None else "--"
        cv.putText(image, label, (int(xs.min()), max(int(ys.min()) - 4, 12)), cv.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
    for *_, line in sorted(rows):
        print(line)
    cv.imwrite(str(out / f"{name.replace(' ', '')}.png"), image)
    if frame.depth is not None:
        files, ranks = board_depth_profile(frame, calib)
        print("  free board height, mm, files a..h: " + " ".join(f"{v:+5.1f}" for v in files))
        print("                         ranks 1..8: " + " ".join(f"{v:+5.1f}" for v in ranks))


def report_totals(totals: dict, by_square: dict, square_mm: float) -> None:
    n = len(totals["ok"])
    print(f"\nright square: {sum(totals['ok'])} of {n} ({sum(totals['ok']) / n:.0%})")
    for rank in range(1, 9):
        hits = totals.get(f"rank{rank}")
        if hits:
            print(f"  rank {rank}: {sum(hits)}/{len(hits)}")
    print(f"right type:   {sum(totals['type_ok'])} of {n}"
          + "".join(f", {what} {times}x" for what, times in Counter(totals["confused"]).most_common()))
    for name in ("foot", "depth", "grasp"):
        if not totals[name]:
            print(f"{name}: no estimates")
            continue
        e = np.array(totals[name])
        size = np.hypot(e[:, 0], e[:, 1])
        print(f"{name:>5} error mm (median along / lateral): {np.median(e[:, 0]):+.1f} / {np.median(e[:, 1]):+.1f}, "
              f"spread {np.std(e[:, 0]):.1f} / {np.std(e[:, 1]):.1f}, "
              f"size median {np.median(size):.1f} max {size.max():.1f}, "
              f"over {GRIP_ROOM_MM:.0f} mm: {int(np.sum(size > GRIP_ROOM_MM))} of {len(e)}")
    for name, label in (("pawn_along", "pawns"), ("other_along", "others")):
        if totals[name]:
            files, along = np.array(totals[name]).T
            means = " ".join(f"{along[files == i].mean():+5.1f}" if (files == i).any() else "    -" for i in range(8))
            print(f"depth along per file a..h, {label:>6}: {means}")
    moved = defaultdict(list)
    for estimates in by_square.values():
        for name in ("foot", "depth", "grasp"):
            xy = np.array([p for kind, p in estimates if kind == name])
            if len(xy) > 1:
                moved[name] += list(np.linalg.norm(xy - xy.mean(axis=0), axis=1) * 1000)
    for name, distances in moved.items():
        print(f"{name:>5} between the views, mm from its own mean: "
              f"median {np.median(distances):.1f}, max {max(distances):.1f}")
    print(f"half a square is {square_mm / 2:.1f} mm: a median error near that is the culprit")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check the piece-to-square mapping in the starting position.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--willy-data", default=None)
    parser.add_argument("--views", type=int, default=3, help="views to observe after the calibration")
    parser.add_argument("--no-move", action="store_true", help="stay where the arm stands")
    parser.add_argument("--all-looks", action="store_true", help="one view from every look pose, game.look_views or not")
    parser.add_argument("--save", action="store_true", help="keep every view as an .npz (image, depth, masks, pose)")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "mapping_test")
    args = parser.parse_args(argv)
    config = load_config(args.config)

    tree = load_cell_tree(args.willy_data or config.willy.data_dir)
    if not tree.ok:
        print(tree)
        return tree.exit_code
    args.out.mkdir(parents=True, exist_ok=True)
    with open_cell(tree, config) as cell:
        mapping = cell.mapping
        result = mapping.calibrate() if args.no_move else cell.calibrate()
        if result.calibration is None:
            print(f"Calibration refused: {result.reason}")
            return 1
        print(f"Calibrated: square {mapping.square * 1000:.1f} mm, {result.corners} corners, "
              f"reprojection {result.reprojection_px:.2f} px, plane normal {result.normal_deg:.1f} deg, "
              f"type mismatches {result.class_mismatches}")
        print("dims (height / foot radius, mm):",
              ", ".join(f"{s} {h * 1000:.0f}/{r * 1000:.1f}" for s, (h, r) in sorted((mapping.dims or {}).items())))

        totals, by_square = defaultdict(list), defaultdict(list)
        first_lock = None
        for number in range(cell.looks if args.all_looks else args.views):
            if args.all_looks:
                cell.go_look(number) if number else cell.go_look()
            obs = mapping.observe()
            name = f"look {number}" if args.all_looks else f"view {number + 1}"
            report_view(obs, name, totals, by_square, args.out)
            if first_lock is None:
                first_lock = obs.board_to_base
            elif args.all_looks:
                shift = np.linalg.norm(obs.board_to_base[:3, 3] - first_lock[:3, 3]) * 1000
                turn = np.trace(first_lock[:3, :3].T @ obs.board_to_base[:3, :3])
                print(f"  the board as seen from here against look 0: {shift:.1f} mm, "
                      f"{np.degrees(np.arccos(np.clip((turn - 1) / 2, -1, 1))):.2f} deg")
            if args.save:
                save_view(args.out / f"{name.replace(' ', '')}.npz", obs)
        square_mm = mapping.square * 1000

    if not totals["ok"]:
        print("\nno pieces observed")
        return 1
    report_totals(totals, by_square, square_mm)
    print(f"images: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
