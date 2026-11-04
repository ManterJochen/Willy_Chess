"""
Mapping test: which square does every piece land on, and how far off is it?

Set up the starting position. The arm goes to its look pose, calibrates, then takes a few views
(observe, exactly as in the game). For every piece it prints the square it was assigned, the square it
stands on, and the error of the two position estimates (foot point from the mask, axis from the depth),
split into "along" (away from the camera) and "lateral" (sideways), in millimetres.

The pattern of the errors names the cause:
  along >> lateral, all the same sign  -> foot radius / piece height wrong, or depth biased
  lateral and along in one direction for every piece  -> board pose / square size off
  only the far ranks wrong  -> feet hidden behind the pieces in front (look pose too flat)
  depth "-" for many pieces  -> too little depth on the pieces (shiny), the foot point alone is used

    python -m src.game.mapping_test [--views 3] [--no-move]
"""
import argparse
from collections import defaultdict
from pathlib import Path

import chess
import cv2 as cv
import numpy as np

from src.game.config import REPO_ROOT, load_config
from src.game.play import load_cell_tree, open_cell
from src.mapping.geometry import BoardCamera, square_centres
from src.mapping.pieces import depth_xy, foot_xy, fuse

START = {sq: p.symbol() for sq, p in chess.Board().piece_map().items()}


def split(err: np.ndarray, xy: np.ndarray, F: np.ndarray) -> tuple[float, float]:
    """Error (x, y) split into along the viewing direction and lateral, millimetres."""
    u = (xy - F) / max(np.linalg.norm(xy - F), 1e-12)
    v = np.array([-u[1], u[0]])
    return float(err @ u * 1000), float(err @ v * 1000)


def fmt(e: tuple[float, float] | None) -> str:
    return "      -      " if e is None else f"{e[0]:+6.1f}/{e[1]:+5.1f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check the piece-to-square mapping in the starting position.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--willy-data", default=None)
    parser.add_argument("--views", type=int, default=3, help="views to observe after the calibration")
    parser.add_argument("--no-move", action="store_true", help="stay where the arm stands")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "mapping_test")
    args = parser.parse_args(argv)
    config = load_config(args.config)

    tree = load_cell_tree(args.willy_data or config.willy.data_dir)
    if not tree.ok:
        print(tree)
        return tree.exit_code
    args.out.mkdir(parents=True, exist_ok=True)
    with open_cell(tree, config) as cell:
        if not args.no_move:
            cell.go_look()
        mapping = cell.mapping
        result = mapping.calibrate()
        if result.calibration is None:
            print(f"Calibration refused: {result.reason}")
            return 1
        print(f"Calibrated: square {mapping.square * 1000:.1f} mm, {result.corners} corners, "
              f"reprojection {result.reprojection_px:.2f} px, plane normal {result.normal_deg:.1f} deg, "
              f"type mismatches {result.class_mismatches}")
        print("dims (height / foot radius, mm):",
              ", ".join(f"{s} {h * 1000:.0f}/{r * 1000:.1f}" for s, (h, r) in sorted((mapping.dims or {}).items())))

        totals = defaultdict(list)
        for view in range(args.views):
            obs = mapping.observe()
            calib, frame = obs.calib, obs.frame
            cam = BoardCamera(frame.K, calib.R, calib.t)
            F = cam.foot
            centres = square_centres(calib.square)
            print(f"\nview {view + 1}: pose {obs.pose_status}, {obs.corners_used} corners, "
                  f"residual {obs.corner_residual_px:.2f} px, {len(frame.classes)} pieces")
            print("  piece  got   true   ok   foot along/lat   depth along/lat   used")
            image = frame.bgr.copy()
            rows = []
            for k, symbol in frame.classes.items():
                mask = frame.labels == k
                h, r = calib.piece_dims(symbol)
                xf = foot_xy(cam, mask, r)
                xd = depth_xy(cam.board_points(frame.depth, mask), F, h, r) if frame.depth is not None else None
                xs = fuse(xf, xd, F)
                # the true square: the nearest start square of this piece type
                cands = [sq for sq, s in START.items() if s == symbol] or list(START)
                true = min(cands, key=lambda sq: np.linalg.norm(xs - centres[sq]))
                got = obs.squares.get(k)
                ok = got == true
                ef = split(xf - centres[true], centres[true], F)
                ed = None if xd is None else split(xd - centres[true], centres[true], F)
                used = "foot" if xd is None or np.allclose(xs, xf) else "depth"
                totals["ok"].append(ok)
                totals["foot"].append(ef)
                if ed is not None:
                    totals["depth"].append(ed)
                totals["used_" + used].append(1)
                rank = chess.square_rank(true)
                totals[f"rank{rank + 1}"].append(ok)
                rows.append((rank, chess.square_file(true),
                             f"  {symbol:>5}  {chess.square_name(got) if got is not None else '--':>3}   "
                             f"{chess.square_name(true):>4}   {'yes' if ok else 'NO ':>3}  {fmt(ef)}    {fmt(ed)}    {used}"))
                ys, xs_ = np.nonzero(mask)
                colour = (0, 200, 0) if ok else (0, 0, 255)
                contours, _ = cv.findContours(mask.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                cv.drawContours(image, contours, -1, colour, 2)
                label = chess.square_name(got) if got is not None else "--"
                cv.putText(image, label, (int(xs_.min()), max(int(ys.min()) - 4, 12)),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
            for *_, line in sorted(rows):
                print(line)
            cv.imwrite(str(args.out / f"view{view + 1}.png"), image)

    n = len(totals["ok"])
    if not n:
        print("\nno pieces observed")
        return 1
    print(f"\nright square: {sum(totals['ok'])} of {n} ({sum(totals['ok']) / n:.0%})")
    for rank in range(1, 9):
        hits = totals.get(f"rank{rank}")
        if hits:
            print(f"  rank {rank}: {sum(hits)}/{len(hits)}")
    for name in ("foot", "depth"):
        e = np.array(totals[name]) if totals[name] else None
        if e is None:
            print(f"{name}: no estimates")
            continue
        print(f"{name:>5} error mm (median along / lateral): {np.median(e[:, 0]):+.1f} / {np.median(e[:, 1]):+.1f}, "
              f"spread {np.std(e[:, 0]):.1f} / {np.std(e[:, 1]):.1f}  ({len(e)} pieces)")
    print(f"square assigned from: depth {len(totals['used_depth'])}x, foot point {len(totals['used_foot'])}x")
    print(f"half a square is {mapping.square * 500:.1f} mm: a median error near that is the culprit")
    print(f"images: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
