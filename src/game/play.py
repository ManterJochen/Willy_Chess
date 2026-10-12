"""
Play chess against Willy on the real board.

    python main.py --willy-data <Willy config directory of the cell>
    python main.py                      # with willy.data_dir set in config/willy_chess.yaml

Willy's config directory names the cameras, their hand-eye calibration, the robot, its hand and the
look poses; config/willy_chess.yaml the game. Before the first game: set up the starting position,
Willy calibrates the board from it.
"""
import argparse
import sys
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator

import chess
import chess.pgn

from src.game.config import REPO_ROOT, ChessConfig, load_config
from src.game.engine import Engine
from src.game.game import ChessGame
from src.game.robot_moves import DECLINE, JawGeometry, RobotMover
from src.mapping.chessboard_mapping_3d import ChessBoardMapping3D, WillyCameraPose
from src.pieces.record import Recorder


@dataclass
class Cell:
    """The cell as the game uses it: a connected robot, the board mapping, the mover and the look pose."""
    robot: Any
    mapping: ChessBoardMapping3D
    mover: RobotMover
    go_look: Callable[[], None]
    decline: dict


def load_cell_tree(data_dir: str | Path | None) -> Any:
    """
    Willy's config tree of the cell from its config directory (no profile, no WILLY_PROFILE).

    Args:
        data_dir (str | Path | None): The directory, relative to the repository or absolute.

    Returns:
        LoadedTree: The tree; one that does not load carries its refusal (tree.ok false).
    """
    from willy import load_tree

    if not data_dir:
        raise SystemExit("no Willy data directory: set willy.data_dir in config/willy_chess.yaml "
                         "or pass --willy-data DIR")
    root = Path(data_dir) if Path(data_dir).is_absolute() else REPO_ROOT / data_dir
    if not root.is_dir():
        raise SystemExit(f"no Willy data directory at {root}")
    return load_tree(None, root=root)


def check_weights(folder: Path) -> None:
    """Refuse weights that are missing or still a Git LFS pointer, with the command that fixes it."""
    model = folder / "model.safetensors"
    if not model.is_file():
        raise SystemExit(f"no detector weights at {model}: train with `python -m src.pieces.train` or pull them")
    if model.stat().st_size < 1024 and model.read_bytes().startswith(b"version https://git-lfs"):
        raise SystemExit(f"{model} is a Git LFS pointer, not the weights: run `git lfs install && git lfs pull`")


def save_pgn(board: chess.Board, result: str, human: bool, folder: Path) -> Path:
    """Write a finished game as PGN, named by its start time."""
    game = chess.pgn.Game.from_board(board)
    now = datetime.now()
    game.headers.update(Event="Willy_Chess", Date=now.strftime("%Y.%m.%d"), Result=result,
                        White="Human" if human == chess.WHITE else "Willy",
                        Black="Willy" if human == chess.WHITE else "Human")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{now:%Y-%m-%d_%H%M%S}.pgn"
    path.write_text(str(game) + "\n", encoding="utf-8")
    return path


@contextmanager
def open_cell(tree: Any, config: ChessConfig) -> Iterator[Cell]:
    """
    Open the cameras, build and connect the robot, load the detector, everything released however it ends.

    The arm plans with a camera world exactly when Willy's CameraWorldPlan says the cell has one; then
    every camera of that world is opened (the primary first, the one the board is mapped with).
    """
    from willy import Camera, CameraWorldPlan, JointPositions, ObjectDetector, Robot

    weights = REPO_ROOT / config.detector.weights
    check_weights(weights)
    section = tree.app_config.camera.cameras
    plan = CameraWorldPlan.from_config(tree.robot, list(section.rigs), primary_rig_id=section.primary_rig_id)
    camera_world = bool(plan.rig_ids)
    if not camera_world:
        print(f"Ohne Kamerawelt: {plan.reason}")
    decline = {} if camera_world else {"decline": DECLINE}

    with ExitStack() as stack:
        cameras = [stack.enter_context(Camera.from_tree(tree, rig_id=rig))
                   for rig in (plan.rig_ids or (section.primary_rig_id,))]
        robot = Robot.from_tree(tree, cameras=cameras) if camera_world else Robot.from_tree(tree)
        detector = ObjectDetector.from_weights(str(weights), segmenter=config.detector.segmenter)
        camera_pose = WillyCameraPose(cameras[0], tcp_reader=robot.arm.get_tcp_pose)
        mapping = ChessBoardMapping3D(cameras[0], detector, camera_pose, detector_threshold=config.detector.threshold,
                                      grasp_height_fraction=config.robot.grasp_height_fraction)
        mover = RobotMover(robot, mapping, config.robot, JawGeometry.from_tree(tree), camera_world)
        looks = tree.robot.look_joint_positions_deg or ()

        def go_look() -> None:
            """Wrist camera: to the first look pose. Fixed camera: home, out of its view."""
            if camera_pose.stamp is not None and looks:
                report = robot.move_joints(JointPositions.deg(*looks[0]), **decline)
            else:
                report = robot.home(**decline)
            if not report.ok:
                raise RuntimeError(f"the arm did not reach the look pose:\n{report}")

        stack.enter_context(robot.connected())
        yield Cell(robot, mapping, mover, go_look, decline)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Play chess against Willy.")
    parser.add_argument("--config", type=Path, default=None, help="game settings (default: config/willy_chess.yaml)")
    parser.add_argument("--willy-data", default=None, help="Willy's config directory of the cell (default: willy.data_dir)")
    args = parser.parse_args(argv)
    config = load_config(args.config)

    from src.ui.pygame_view import PygameView

    tree = load_cell_tree(args.willy_data or config.willy.data_dir)
    if not tree.ok:
        print(tree)
        return tree.exit_code
    recorder = Recorder(REPO_ROOT / config.record.out_dir) if config.record.enabled else None
    with open_cell(tree, config) as cell, Engine(config.engine) as engine, PygameView() as view:
        while True:
            game = ChessGame(cell.mapping, engine, cell.mover, view, config, cell.go_look, recorder)
            result = game.run()
            if game.tracker is not None and game.tracker.board.move_stack:
                print(f"{result}: {save_pgn(game.tracker.board, result, game.human, REPO_ROOT / config.game.pgn_dir)}")
            if result == "*" or not game.ask_new_game():
                return 0


if __name__ == "__main__":
    sys.exit(main())
