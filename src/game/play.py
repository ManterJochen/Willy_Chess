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
from typing import Any, Callable, Iterator, Sequence

import chess
import chess.pgn

from src.game.config import REPO_ROOT, ChessConfig, load_config
from src.game.engine import Engine
from src.game.game import ChessGame, TakenOff
from src.game.robot_moves import DECLINE, JawGeometry, RobotMover
from src.mapping.calibration import CalibrationResult
from src.mapping.chessboard_mapping_3d import ChessBoardMapping3D, WillyCameraPose
from src.pieces.record import GameLog


@dataclass
class Cell:
    """
    The cell as the game uses it: a connected robot, the board mapping, the mover and the look poses.

    Attributes:
        go_look (Callable[..., None]): Brings the arm to a look pose, go_look(view), by default the first.
        looks (int): Look poses the cell has (1 for a fixed camera, which does not move to look).
        views (int): How many of them a game may use (game.look_views).
    """
    robot: Any
    mapping: ChessBoardMapping3D
    mover: RobotMover
    go_look: Callable[..., None]
    decline: dict
    looks: int = 1
    views: int = 1

    def calibrate(self, expected: dict[int, str] | None = None, attempts: int = 3) -> CalibrationResult:
        """
        Calibrate as a game does: a refusal is fail-closed and every grab has new noise, so try again, from the next
        look pose where there is one.

        Args:
            expected (dict[int, str] | None): Square -> FEN symbol of the pieces; None is the starting position.
            attempts (int): The fewest tries; every usable look pose gets one.

        Returns:
            CalibrationResult: The first calibration that succeeded, or the last refusal.
        """
        for attempt in range(max(attempts, self.views)):
            view = attempt % self.views
            self.go_look(view) if view else self.go_look()
            result = self.mapping.calibrate(expected)
            if result.calibration is not None:
                break
            print(f"Calibration attempt {attempt + 1}, look {view}: {result.reason}")
        return result


def look_stops(looks: tuple, view: int) -> tuple:
    """The joint poses on the way to look pose `view`: any other than the first by way of the first."""
    return (looks[0], looks[view]) if view else (looks[0],)


def go_to_look(robot: Any, looks: tuple, view: int, decline: dict) -> None:
    """
    Bring the arm to where the camera sees the board: a wrist camera to that look pose, stop by stop, and no further
    than the first stop it did not reach; a fixed camera (no looks) to home, out of its view.

    Raises:
        RuntimeError: The arm did not get there.
    """
    from willy import JointPositions

    if not looks:
        reports = (robot.home(**decline) for _ in range(1))
    else:
        reports = (robot.move_joints(JointPositions.deg(*stop), **decline) for stop in look_stops(looks, view))
    for report in reports:          # one at a time: a generator, the next motion only after this one was reached
        if not report.ok:
            raise RuntimeError(f"the arm did not reach the look pose:\n{report}")


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


def hand_detector(tree: Any) -> Any:
    """
    Willy's palm detector, its bundle from the tree's weights folder. Refused with the command that fixes it:
    a check that silently does not run looks exactly like a cell with nobody in it.
    """
    from src.models.handdetection import build_palm_detector  # Willy, imported only when it is used

    try:
        return build_palm_detector(weights_root=str(tree.app_config.models.weights_root))
    except (ImportError, FileNotFoundError) as err:
        raise SystemExit(f"{err}\nThe hand check before every move needs it: `python -m src.pieces.fetch` "
                         f"(or game.hand_detection: false, at your own risk)") from err


def save_pgn(board: chess.Board, result: str, human: bool, folder: Path, earlier: Sequence[TakenOff] = ()) -> Path:
    """
    Write a finished game as PGN, named by its start time. A piece Willy took off the board is no move a PGN can
    hold: such a game is written in parts, one PGN game each, every later one from its own position (FEN) and
    saying which piece Willy took.
    """
    now = datetime.now()
    parts = [taken.before for taken in earlier] + [board]
    games = []
    for i, part in enumerate(parts):
        game = chess.pgn.Game.from_board(part)
        game.headers.update(Event="Willy_Chess", Date=now.strftime("%Y.%m.%d"),
                            Result=result if i == len(parts) - 1 else "*",
                            White="Human" if human == chess.WHITE else "Willy",
                            Black="Willy" if human == chess.WHITE else "Human")
        if len(parts) > 1:
            game.headers["Round"] = f"part {i + 1} of {len(parts)}"
        if i > 0:
            game.comment = earlier[i - 1].describe()
        games.append(str(game))
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{now:%Y-%m-%d_%H%M%S}.pgn"
    path.write_text("\n\n".join(games) + "\n", encoding="utf-8")
    return path


@contextmanager
def open_cell(tree: Any, config: ChessConfig) -> Iterator[Cell]:
    """
    Open the cameras, build and connect the robot, load the detector, everything released however it ends.

    The arm plans with a camera world exactly when Willy's CameraWorldPlan says the cell has one; then
    every camera of that world is opened (the primary first, the one the board is mapped with).
    """
    from willy import Camera, CameraWorldPlan, ObjectDetector, Robot

    weights = REPO_ROOT / config.detector.weights
    check_weights(weights)
    section = tree.app_config.camera.cameras
    plan = CameraWorldPlan.from_config(tree.robot, list(section.rigs), primary_rig_id=section.primary_rig_id)
    camera_world = bool(plan.rig_ids)
    if not camera_world:
        print(f"No camera world: {plan.reason}")
    decline = {} if camera_world else {"decline": DECLINE}

    with ExitStack() as stack:
        cameras = [stack.enter_context(Camera.from_tree(tree, rig_id=rig))
                   for rig in (plan.rig_ids or (section.primary_rig_id,))]
        robot = Robot.from_tree(tree, cameras=cameras) if camera_world else Robot.from_tree(tree)
        detector = ObjectDetector.from_weights(str(weights), segmenter=config.detector.segmenter)
        camera_pose = WillyCameraPose(cameras[0], robot)
        mapping = ChessBoardMapping3D(cameras[0], detector, camera_pose, detector_threshold=config.detector.threshold,
                                      grasp_height_fraction=config.robot.grasp_height_fraction)
        mapping.surface_mm = config.robot.board_surface_mm
        if config.game.hand_detection:
            mapping.hand_detector = stack.enter_context(hand_detector(tree))
        mover = RobotMover(robot, mapping, config.robot, JawGeometry.from_tree(tree), camera_world)
        looks = tuple(tree.robot.motion.look_deg or ()) if camera_pose.on_wrist else ()

        def go_look(view: int = 0) -> None:
            go_to_look(robot, looks, view, decline)

        stack.enter_context(robot.connected())
        count = len(looks) or 1
        yield Cell(robot, mapping, mover, go_look, decline, count, min(config.game.look_views, count))


def play_games(cell: Cell, engine: Any, view: Any, config: ChessConfig, log_root: Path | None = None) -> ChessGame:
    """
    Game after game until the human stops: every game with moves saved as PGN, a crashed one too.

    Args:
        log_root (Path | None): Where each game keeps its log (src/pieces/record.py); None keeps none.

    Returns:
        ChessGame: The last game.
    """
    while True:
        log = GameLog(log_root) if log_root is not None else None
        game = ChessGame(cell.mapping, engine, cell.mover, view, config, cell.go_look, log, cell.views)
        result = "*"
        try:
            result = game.run()
        except BaseException as err:
            if log is not None:
                log.event("crash", error=repr(err))
            raise
        finally:
            if game.tracker is not None and (game.tracker.board.move_stack or game.earlier):
                print(f"{result}: {save_pgn(game.tracker.board, result, game.human, REPO_ROOT / config.game.pgn_dir, game.earlier)}")
            if log is not None:
                log.close()
        if result == "*" or not game.ask_new_game():
            return game


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
    log_root = REPO_ROOT / config.record.out_dir if config.record.enabled else None
    # Stockfish first: a missing binary is found before the arm connects and the models load
    with Engine(config.engine) as engine, open_cell(tree, config) as cell, PygameView() as view:
        play_games(cell, engine, view, config, log_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
