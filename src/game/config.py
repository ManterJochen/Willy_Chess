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


@dataclass
class RobotConfig:
    standoff_mm: float = 80.0
    finger_clearance_mm: float = 8.0
    place_air_mm: float = 2.0
    grasp_height_fraction: dict[str, float] = field(
        default_factory=lambda: {s: 0.45 for s in "pnbrqk"})
    graveyard: GraveyardConfig = field(default_factory=GraveyardConfig)


@dataclass
class RecordConfig:
    enabled: bool = False
    out_dir: str = "data/recorded"


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
        if self.robot.graveyard.side not in ("+x", "-x", "+y", "-y"):
            raise ValueError(f"robot.graveyard.side must be +x, -x, +y or -y, not {self.robot.graveyard.side!r}")
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
        ValueError: An unknown key (with its dotted name) or a value no game can run with.
    """
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
