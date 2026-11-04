"""
Everything a game saw and decided, written as it happens: the flight recorder of the cell.

One folder per game under record.out_dir:

    log.jsonl           one line per event: every view, what the tracker made of it, every question and its answer,
                        Stockfish's moves, what the arm did, and for each view which position it really showed
    views/000012.png    the colour image of view 12, undistorted and lossless
    views/000012.npz    its depth, piece masks and poses

A view's truth is often known only later: a move the camera did not recognise is settled by the click that follows.
So nothing here labels anything, src/pieces/export.py does, from the finished log. Images are written beside the game,
and a problem with the disk is printed once and never stops a game.
"""
import json
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import chess
import cv2 as cv
import numpy as np

from src.mapping.frame import BoardFrame
from src.mapping.geometry import BoardCalibration

LOG_NAME = "log.jsonl"
VIEWS = "views"
_QUEUE = 16             # views waiting for the disk before the game waits too


def _plain(value: Any) -> Any:
    """What json cannot write by itself."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


class GameLog:
    """
    The log of one game. Use as a context manager, or close it: the views still on their way are written first.

    Args:
        root (str | Path): The folder the game's own folder is made in, named by its start time.
    """

    def __init__(self, root: str | Path):
        stamp = f"{datetime.now():%Y-%m-%d_%H%M%S}"
        self.folder = Path(root) / stamp
        for attempt in range(1, 100):
            if not self.folder.exists():
                break
            self.folder = Path(root) / f"{stamp}_{attempt}"
        (self.folder / VIEWS).mkdir(parents=True)
        self._file = (self.folder / LOG_NAME).open("a", encoding="utf-8")
        self._views = 0
        self._told = False
        self._queue: queue.Queue = queue.Queue(maxsize=_QUEUE)
        self._writer = threading.Thread(target=self._write_views, name="game-log", daemon=True)
        self._writer.start()

    def __enter__(self) -> "GameLog":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def event(self, kind: str, **fields: Any) -> None:
        """One line of the log, on disk at once: what a crash has to explain must not sit in a buffer."""
        if self._file.closed:               # "another game?" comes after the game's log was closed
            return
        try:
            self._file.write(json.dumps({"event": kind, "time": time.time(), **fields}, default=_plain) + "\n")
            self._file.flush()
        except Exception as err:  # noqa: BLE001 (the game goes on without its log)
            self._tell(err)

    def view(self, frame: BoardFrame, *, moment: str, look: int = 0, ply: int = 0, calib: BoardCalibration | None = None,
             camera_to_base: np.ndarray | None = None, board_to_base: np.ndarray | None = None,
             status: str = "") -> int:
        """
        Keep one view with what the detector made of it.

        Args:
            frame (BoardFrame): The image, depth and masks.
            moment (str): When it was taken: "calibration", "your_move", "before_arm", "after_arm", ...
            look (int): The look pose it was taken from.
            ply (int): Half moves the tracker had committed at the time.
            calib (BoardCalibration | None): Its board pose; None for a refused calibration.
            camera_to_base (np.ndarray | None): CAMERA -> BASE, 4x4, metres.
            board_to_base (np.ndarray | None): BOARD -> BASE as seen in this view, 4x4, metres.
            status (str): The pose status of the view.

        Returns:
            int: The number of the view, for what the log says about it later.
        """
        number = self._views
        self._views += 1
        self.event("view", view=number, moment=moment, look=look, ply=ply, status=status, roi=frame.roi,
                   classes=frame.classes, scores=frame.scores, detections=frame.detections,
                   square=calib.square if calib is not None else None, dims=calib.dims if calib is not None else None)
        arrays = {"K": frame.K, "depth": frame.depth, "labels": frame.labels, "camera_to_base": camera_to_base,
                  "board_to_base": board_to_base, "R": calib.R if calib is not None else None,
                  "t": calib.t if calib is not None else None}
        self._queue.put((number, frame.bgr, {name: value for name, value in arrays.items() if value is not None}))
        return number

    def settled(self, views: list[int], physical: dict[int, str], how: str) -> None:
        """Which position these views showed, and how that came to be known."""
        self.event("settled", views=list(views), how=how,
                   physical={chess.square_name(sq): symbol for sq, symbol in physical.items()})

    def close(self) -> None:
        """Write what is still waiting and close the log; closing twice is fine."""
        if self._writer.is_alive():
            self._queue.put(None)
            self._writer.join()
        if not self._file.closed:
            self._file.close()

    def _write_views(self) -> None:
        while (item := self._queue.get()) is not None:
            number, bgr, arrays = item
            try:
                stem = self.folder / VIEWS / f"{number:06d}"
                if not cv.imwrite(str(stem.with_suffix(".png")), bgr):
                    raise OSError(f"could not write {stem}.png")
                np.savez_compressed(stem.with_suffix(".npz"), **arrays)
            except Exception as err:  # noqa: BLE001 (the game goes on without this view)
                self._tell(err)

    def _tell(self, err: Exception) -> None:
        if not self._told:
            self._told = True
            print(f"The game log in {self.folder} has a problem and may miss something: {err}")


@dataclass
class LoggedGame:
    """
    A log read back.

    Attributes:
        folder (Path): The game's folder.
        events (list[dict]): Every line of the log, in order.
        views (dict[int, dict]): View number -> its line.
        truth (dict[int, tuple[dict[int, str], str]]): View number -> the position it showed and how that is known;
            views missing here were never settled.
        doubted (set[int]): Views the tracker could not explain when they were taken.
    """
    folder: Path
    events: list[dict] = field(default_factory=list)
    views: dict[int, dict] = field(default_factory=dict)
    truth: dict[int, tuple[dict[int, str], str]] = field(default_factory=dict)
    doubted: set[int] = field(default_factory=set)

    @property
    def name(self) -> str:
        return self.folder.name


def read_game(folder: str | Path) -> LoggedGame:
    """Read a game's log; a last line a crash cut off is left out."""
    game = LoggedGame(Path(folder))
    for line in (game.folder / LOG_NAME).read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        game.events.append(event)
        kind = event["event"]
        if kind == "view":
            game.views[event["view"]] = event
        elif kind == "settled":
            physical = {chess.parse_square(name): symbol for name, symbol in event["physical"].items()}
            for number in event["views"]:
                game.truth[number] = (physical, event["how"])
        elif (kind == "step" and event.get("flagged")) or (kind == "verify" and not event.get("ok")):
            if event.get("view") is not None:
                game.doubted.add(event["view"])
    return game


def load_view(game: LoggedGame, number: int) -> tuple[BoardFrame, BoardCalibration | None, np.ndarray | None]:
    """
    One view as the game saw it, for the export and for looking at a strange moment again.

    Returns:
        tuple: The frame, its calibration (None for a refused one) and CAMERA -> BASE in metres, if known.
    """
    meta = game.views[number]
    stem = game.folder / VIEWS / f"{number:06d}"
    bgr = cv.imread(str(stem.with_suffix(".png")))
    if bgr is None:
        raise FileNotFoundError(f"no image of view {number} in {game.folder}")
    with np.load(stem.with_suffix(".npz")) as arrays:
        data = {name: arrays[name] for name in arrays.files}
    frame = BoardFrame(bgr, data["K"], data.get("depth"), data.get("labels"),
                       {int(k): v for k, v in meta["classes"].items()}, {int(k): v for k, v in meta["scores"].items()},
                       tuple(meta["roi"]) if meta.get("roi") else None, meta.get("detections", []))
    calib = None
    if "R" in data:
        dims = {k: tuple(v) for k, v in meta["dims"].items()} if meta.get("dims") else None
        calib = BoardCalibration(data["R"], data["t"], float(meta["square"]), dims)
    return frame, calib, data.get("camera_to_base")
