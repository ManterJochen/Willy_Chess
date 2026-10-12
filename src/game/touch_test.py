"""
Touch test: does the board in the robot's frame lie where the real one does?

Set up the starting position. Willy calibrates, then holds its grasp centre above the centre of free
squares (ranks 3-6, the pieces stand on the others) one after the other: check that it stands over the
middle of each square. An offset that is the same everywhere is a hand-eye or camera calibration error.

    python -m src.game.touch_test --willy-data <Willy config directory of the cell> [--height-mm 30]
"""
import argparse
import sys
from pathlib import Path

import chess

from src.game.config import load_config
from src.game.play import load_cell_tree, open_cell

SQUARES = (chess.A3, chess.H3, chess.H6, chess.A6, chess.E5)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hold Willy's grasp centre above board squares.")
    parser.add_argument("--height-mm", type=float, default=30.0,
                        help="grasp centre above the board; the fingertips reach ~10 mm further down (default: 30)")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--willy-data", default=None, help="Willy's config directory of the cell (default: willy.data_dir)")
    args = parser.parse_args(argv)
    config = load_config(args.config)

    tree = load_cell_tree(args.willy_data or config.willy.data_dir)
    if not tree.ok:
        print(tree)
        return tree.exit_code
    with open_cell(tree, config) as cell:
        cell.go_look()
        result = cell.mapping.calibrate()
        if result.calibration is None:
            print(f"Einmessen abgelehnt: {result.reason}")
            return 1
        print(f"Eingemessen: Feld {cell.mapping.square * 1000:.1f} mm, {result.corners} Ecken, "
              f"Reprojektion {result.reprojection_px:.2f} px")
        for square in SQUARES:
            x, y, z = cell.mapping.square_center_base(square, args.height_mm / 1000.0)
            input(f"Enter: {args.height_mm:.0f} mm über {chess.square_name(square)} ({x:.1f}, {y:.1f}, {z:.1f} mm) ")
            report = cell.robot.move(cell.robot.tool_down(x, y, z), **cell.decline)
            if not report.ok:
                print(report)
                return 1
        cell.go_look()
    return 0


if __name__ == "__main__":
    sys.exit(main())
