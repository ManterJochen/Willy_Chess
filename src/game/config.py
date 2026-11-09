"""
Settings of the chess game, loaded from config/willy_chess.yaml.

Willy's own config directory (willy.data_dir) keeps what belongs to the cell: camera, hand-eye
calibration, robot, hand and look poses. This file keeps what belongs to the game.
"""
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "willy_chess.yaml"


@dataclass
class WillyConfig:
    data_dir: str | None = None


@dataclass
class DetectorConfig:
    weights: str = "docs/weights/chess_pieces_rtdetr"
    threshold: float | None = None
    segmenter: str | None = None


@dataclass
class GameConfig:
    human_color: str = "white"
    promotion_mode: str = "pawn_stays"
    pgn_dir: str = "data/games"
    look_views: int = 1
    # Before the arm moves, Willy's MediaPipe palm detector looks for a hand in the whole picture, not only
    # over the 8x8 squares where the depth looks; it needs the bundle `python -m src.pieces.fetch` downloads
    hand_detection: bool = True
    # Willy thinks at least this long on screen: he weighs his candidates once more after Stockfish has answered
    think_show_s: float = 2.5
    # A move against the rules, clearly on the board: the first ones you take back yourself. From the put_back_from-th
    # of a game Willy carries the piece back himself (0: never).
    put_back_from: int = 2
    # Whether Willy loses his patience: from the take_off_from-th he takes the piece off the board, out of the game.
    take_off: bool = True
    take_off_from: int = 4


@dataclass
class EngineConfig:
    path: str | None = None
    think_time_s: float = 1.0
    elo: int | None = None


@dataclass
class TrackingConfig:
    sigma_squares: float = 0.3
    gate: float = 10.0
    class_cost: float = 4.0


@dataclass
class GraveyardConfig:
    side: str = "+x"
    distance_squares: float = 1.5
    columns: int = 2
    table_offset_m: float = -0.012
    # From one slot to the next, in squares. The Hand-E's housing reaches 37.5 mm to each side 36.5 mm above the
    # grasp: one square (45 mm) puts it into a queen set down beside, where no camera sees and nothing refuses
    spacing_squares: float = 1.3


@dataclass
class RobotConfig:
    standoff_mm: float = 80.0
    finger_clearance_mm: float = 8.0
    place_air_mm: float = 2.0
    # The jaws close this much narrower than the measured width. A Robotiq stops on the piece at its force anyway;
    # Willy's 1 mm let them close on nothing where the depth read a knight's neck a little wide (2026-11-05)
    squeeze_mm: float = 4.0
    # Back home (over the board) before a captured piece goes out to the graveyard: a clean start for the long reach
    home_before_graveyard: bool = True
    # The board surface above the robot base, millimetres, measured by hand; None takes it from the camera. Only
    # where the camera's height is known to be off: Willy's camera world keeps its own reading, and a finger sent
    # to the true height meets the board where that reading lies higher (a hand-eye calibration is the real fix).
    board_surface_mm: float | None = None
    grasp_height_fraction: dict[str, float] = field(
        default_factory=lambda: {s: 0.45 for s in "pnbrqk"})
    graveyard: GraveyardConfig = field(default_factory=GraveyardConfig)


@dataclass
class RecordConfig:
    enabled: bool = True
    out_dir: str = "data/game_logs"


@dataclass
class ChessConfig:
    willy: WillyConfig = field(default_factory=WillyConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    game: GameConfig = field(default_factory=GameConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    record: RecordConfig = field(default_factory=RecordConfig)

    def validate(self) -> "ChessConfig":
        """Refuse values no game can run with, naming the key."""
        if self.game.human_color not in ("white", "black"):
            raise ValueError(f"game.human_color must be white or black, not {self.game.human_color!r}")
        if self.game.promotion_mode not in ("pawn_stays", "piece_swapped"):
            raise ValueError(f"game.promotion_mode must be pawn_stays or piece_swapped, not {self.game.promotion_mode!r}")
        for key, least in (("put_back_from", 0), ("take_off_from", 1)):
            value = getattr(self.game, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < least:
                raise ValueError(f"game.{key} must be a whole number, {least} or more, not {value!r}")
        if not isinstance(self.game.take_off, bool):
            raise ValueError(f"game.take_off must be true or false, not {self.game.take_off!r}")
        if self.game.look_views < 1:
            raise ValueError(f"game.look_views must be at least 1, not {self.game.look_views!r}")
        if self.robot.graveyard.side not in ("+x", "-x", "+y", "-y"):
            raise ValueError(f"robot.graveyard.side must be +x, -x, +y or -y, not {self.robot.graveyard.side!r}")
        if self.robot.graveyard.spacing_squares < 1.0:
            raise ValueError(f"robot.graveyard.spacing_squares must be at least 1, not {self.robot.graveyard.spacing_squares!r}")
        missing = set("pnbrqk") - set(self.robot.grasp_height_fraction)
        if missing:
            raise ValueError(f"robot.grasp_height_fraction misses {', '.join(sorted(missing))}")
        return self


def load_config(path: str | Path | None = None) -> ChessConfig:
    """
    Load the game settings; keys missing from the file keep their defaults.

    Args:
        path (str | Path | None): The YAML file (default: config/willy_chess.yaml).

    Returns:
        ChessConfig: The validated settings.

    Raises:
        ValueError: A file named but not there, an unknown key (with its dotted name) or a value no game can run with.
    """
    if path is not None and not Path(path).is_file():
        raise ValueError(f"no chess config at {path}")
    path = Path(path) if path is not None else DEFAULT_CONFIG
    data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else None
    return _merge(ChessConfig(), data or {}, "").validate()


def _merge(target: Any, data: dict, prefix: str) -> Any:
    known = {f.name: f for f in fields(target)}
    for key, value in data.items():
        if key not in known:
            raise ValueError(f"unknown setting {prefix}{key} in the chess config")
        current = getattr(target, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise ValueError(f"{prefix}{key} must be a section, not {value!r}")
            _merge(current, value, f"{prefix}{key}.")
        elif isinstance(current, dict):
            setattr(target, key, {**current, **(value or {})})
        else:
            setattr(target, key, value)
    return target
