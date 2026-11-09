"""
Stockfish through python-chess.
"""
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import chess
import chess.engine

from src.game.config import REPO_ROOT, EngineConfig

BUNDLED = {"win32": REPO_ROOT / "external" / "chess" / "windows" / "stockfish" / "stockfish-windows-x86-64-universal.exe"}
MATE_CP = 10000                 # a mate as centipawns, for bars and moods


@dataclass(frozen=True)
class Candidate:
    """
    One move Stockfish is weighing.

    Attributes:
        move (chess.Move): The move.
        san (str): The move as a person writes it.
        cp (int): How good it is for the side to move, centipawns (a mate is +-MATE_CP).
        mate (int | None): Moves to mate, negative where the side to move gets mated.
    """
    move: chess.Move
    san: str
    cp: int
    mate: int | None = None

    @property
    def label(self) -> str:
        """'+0.35', or 'M3' / 'M-3' for a mate."""
        return f"M{self.mate}" if self.mate is not None else f"{self.cp / 100:+.2f}"


@dataclass(frozen=True)
class Thought:
    """How far Stockfish has got: its search depth and its best moves so far, the best first."""
    depth: int
    candidates: tuple[Candidate, ...] = ()


def thought_of(board: chess.Board, lines: list[dict]) -> Thought:
    """A Thought from python-chess's analysis lines (``analysis.multipv``); lines without a move yet are left out."""
    candidates = []
    for info in lines:
        pv, score = info.get("pv"), info.get("score")
        if not pv or score is None or not board.is_legal(pv[0]):
            continue
        pov = score.relative
        candidates.append(Candidate(pv[0], board.san(pv[0]), pov.score(mate_score=MATE_CP), pov.mate()))
    depth = max((int(info.get("depth", 0)) for info in lines), default=0)
    return Thought(depth, tuple(candidates))


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

    def think(self, board: chess.Board, on_thought: Callable[[Thought], None], lines: int = 3,
              fps: float = 30.0) -> chess.Move:
        """
        Stockfish's move, watched while it searches: on_thought gets its best `lines` moves so far about `fps`
        times a second, so the window can show Willy thinking. The move is the one Stockfish answers with (its
        UCI_Elo included), not simply the first line.
        """
        with self.engine.analysis(board, self.limit, multipv=lines) as analysis:
            while True:
                try:
                    while not analysis.would_block():
                        analysis.get()
                except chess.engine.AnalysisComplete:
                    break
                on_thought(thought_of(board, analysis.multipv))
                time.sleep(1.0 / fps)
            on_thought(thought_of(board, analysis.multipv))
            return analysis.wait().move

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
