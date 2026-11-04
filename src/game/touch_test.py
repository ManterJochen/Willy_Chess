"""
Touch test: does the board in the robot's frame lie where the real one does, and can the arm reach all of it?

Set up the starting position. Willy calibrates, then holds its grasp centre above the centre of free
squares (ranks 3-6, the pieces stand on the others) one after the other: check that it stands over the
middle of each square. An offset that is the same everywhere is a hand-eye or camera calibration error.
Then the same above the corners of the graveyard: a slot the arm cannot reach stops a capture in the game
with the piece in the hand, here it only stops the test. With --corners also down to a1, h1, a8 and h8, where
the arm reaches furthest: the test asks you to take the rook off first.

Between two points the arm goes up first and comes down again, as it does in the game.

    python -m src.game.touch_test --willy-data <Willy config directory of the cell> [--height-mm 30] [--corners]
"""
import argparse
import sys
from pathlib import Path

import chess

from src.game.config import load_config
from src.game.play import load_cell_tree, open_cell
from src.game.robot_moves import graveyard_slot

SQUARES = (chess.A3, chess.H3, chess.H6, chess.A6, chess.E5)
CORNERS = (chess.A1, chess.H1, chess.H8, chess.A8)
GRAVEYARD_SLOTS = (0, 7, 15, 8)     # the corners of two columns of eight, the most a game can fill


def touch_points(mapping, graveyard, height_m: float,
                 corners: bool = False) -> list[tuple[str, tuple[float, float, float]]]:
    """(name, x, y, z in BASE, millimetres) of everything the test holds the grasp centre above."""
    points = [(chess.square_name(sq), mapping.square_center_base(sq, height_m)) for sq in SQUARES]
    if corners:
        points += [(f"{chess.square_name(sq)} (take the rook off first)", mapping.square_center_base(sq, height_m))
                   for sq in CORNERS]
    for slot in GRAVEYARD_SLOTS:
        xy = graveyard_slot(slot, mapping.square, graveyard)
        points.append((f"graveyard slot {slot}", mapping.board_to_base_mm(xy, graveyard.table_offset_m + height_m)))
    return [(name, tuple(float(v) for v in xyz)) for name, xyz in points]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hold Willy's grasp centre above board squares and graveyard slots.")
    parser.add_argument("--height-mm", type=float, default=30.0,
                        help="grasp centre above the board; the fingertips reach ~10 mm further down (default: 30)")
    parser.add_argument("--corners", action="store_true", help="also down to a1, h1, h8 and a8, the rooks taken off")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--willy-data", default=None, help="Willy's config directory of the cell (default: willy.data_dir)")
    args = parser.parse_args(argv)
    config = load_config(args.config)

    tree = load_cell_tree(args.willy_data or config.willy.data_dir)
    if not tree.ok:
        print(tree)
        return tree.exit_code
    with open_cell(tree, config) as cell:
        result = cell.calibrate()
        if result.calibration is None:
            print(f"Calibration refused: {result.reason}")
            return 1
        print(f"Calibrated: square {cell.mapping.square * 1000:.1f} mm, {result.corners} corners, "
              f"reprojection {result.reprojection_px:.2f} px")
        # over the tallest piece: the same height a carried piece flies at in the game
        above = cell.mover.standoff_mm(result.calibration)
        for name, (x, y, z) in touch_points(cell.mapping, config.robot.graveyard, args.height_mm / 1000.0, args.corners):
            input(f"Enter: {args.height_mm:.0f} mm above {name} ({x:.1f}, {y:.1f}, {z:.1f} mm) ")
            for height in (z + above, z):
                report = cell.robot.move(cell.robot.tool_down(x, y, height), **cell.decline)
                if not report.ok:
                    print(report)
                    return 1
            report = cell.robot.move(cell.robot.tool_down(x, y, z + above), **cell.decline)
            if not report.ok:
                print(report)
                return 1
        cell.go_look()
    return 0


if __name__ == "__main__":
    sys.exit(main())
