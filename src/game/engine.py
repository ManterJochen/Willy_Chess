"""
Stockfish through python-chess.
"""
import os
import shutil
import sys
from pathlib import Path

import chess
import chess.engine

from src.game.config import REPO_ROOT, EngineConfig

BUNDLED = {"win32": REPO_ROOT / "external" / "chess" / "windows" / "stockfish" / "stockfish-windows-x86-64-universal.exe"}


def find_stockfish(path: str | None = None) -> Path:
    """
    The Stockfish binary: the configured path, $STOCKFISH_PATH, stockfish on PATH, the bundled one.

    Raises:
        FileNotFoundError: None of them exists.
    """
    for candidate in (path, os.environ.get("STOCKFISH_PATH"), shutil.which("stockfish"), BUNDLED.get(sys.platform)):
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    raise FileNotFoundError("Stockfish not found: set engine.path in config/willy_chess.yaml or $STOCKFISH_PATH, "
                            "or install it (macOS: brew install stockfish)")


class Engine:
    """
    Stockfish as Willy's opponent brain. Use as a context manager so the process always ends.

    Args:
        config (EngineConfig): Path, think time per move and playing strength (Elo; None = full strength).
    """

    def __init__(self, config: EngineConfig):
        self.limit = chess.engine.Limit(time=config.think_time_s)
        self.engine = chess.engine.SimpleEngine.popen_uci(str(find_stockfish(config.path)))
        if config.elo is not None:
            option = self.engine.options["UCI_Elo"]
            if not option.min <= config.elo <= option.max:
                self.engine.quit()
                raise ValueError(f"engine.elo: this Stockfish plays {option.min}..{option.max}, not {config.elo}")
            self.engine.configure({"UCI_LimitStrength": True, "UCI_Elo": config.elo})

    def __enter__(self) -> "Engine":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def best_move(self, board: chess.Board) -> chess.Move:
        """Stockfish's move in this position."""
        return self.engine.play(board, self.limit).move

    def evaluation(self, board: chess.Board) -> str:
        """Short evaluation from White's view for the display: '+0.35' or 'M3'."""
        info = self.engine.analyse(board, chess.engine.Limit(time=0.1))
        score = info["score"].white()
        if score.is_mate():
            return f"M{score.mate()}"
        return f"{score.score() / 100:+.2f}"

    def score(self, board: chess.Board, color: bool) -> int:
        """Centipawns from the view of `color` (a mate counts as +-100000), for a draw offer."""
        info = self.engine.analyse(board, chess.engine.Limit(time=0.3))
        return info["score"].pov(color).score(mate_score=100000)

    def close(self) -> None:
        self.engine.quit()
