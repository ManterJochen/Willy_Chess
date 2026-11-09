"""
The game: calibrate, then take turns until the game ends.

Your turn: you move, press the space bar, Willy looks and recognises the move from the rules
(src/game/tracker.py). Willy's turn: Stockfish chooses, the robot carries the pieces, Willy looks
again and checks that the board shows what it should. Whatever cannot be confirmed is shown and
asked for, never guessed: a flagged move, a moved board, a grasp that did not finish.

A move against the rules that the whole board clearly shows is no question but a fault: the first ones
you take back yourself, then Willy carries the piece back, and in the end he takes it off the board.
"""
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Protocol

import chess
import numpy as np

from src.game.config import ChessConfig
from src.game.engine import Thought
from src.game.persona import Persona, decided_event, human_move_event, willy_pawns, your_move_event
from src.game.robot_moves import PIECE_NAMES, MoveExecution, RobotMover, Transfer, graveyard_slot, plan_transfers
from src.game.tracker import MoveTracker, TrackResult, board_without, physical_after
from src.mapping.chessboard_mapping_3d import BoardObservation, ChessBoardMapping3D
from src.mapping.geometry import square_centres
from src.pieces.record import GameLog

PROMOTION_KEYS = {"q": chess.QUEEN, "r": chess.ROOK, "b": chess.BISHOP, "n": chess.KNIGHT}
INTRUDER_PX = 800           # a hand or an arm over the board covers thousands of pixels, noise none
DRAW_ACCEPT_CP = 50         # Willy accepts a draw unless it stands better than half a pawn
REACTION_S = 1.2            # Willy's word on the human's move stays this long before his thinking takes over
PONDER_S = 0.6              # each candidate Willy weighs again once Stockfish has answered
REASONS = {"moves": "this piece cannot move like that", "check": "your king would be in check",
           "rules": "the rules do not allow it here"}
ILLEGAL_EVENTS = {"moves": "illegal", "check": "illegal_check", "rules": "illegal_rules"}
OUTCOMES = {chess.Termination.CHECKMATE: "checkmate", chess.Termination.STALEMATE: "stalemate",
            chess.Termination.INSUFFICIENT_MATERIAL: "insufficient material",
            chess.Termination.SEVENTYFIVE_MOVES: "75-move rule",
            chess.Termination.FIVEFOLD_REPETITION: "fivefold repetition"}


@dataclass
class ViewState:
    """
    Everything the display shows.

    Attributes:
        mood (str): Willy's face (src/game/persona.py).
        speech (str): What Willy says.
        thought (Thought | None): The moves Willy weighs while he thinks, kept after he decided.
        focus (chess.Move | None): The move Willy looks at while he thinks.
        intent (chess.Move | None): Willy's move while the arm carries it out; also a piece he carries back.
        taking (int | None): The square of a piece Willy takes off the board, while the arm does.
        strikes (int): Moves against the rules in this game.
        patience (int): Those Willy lets pass before he takes the piece (0: he never does).
        earlier (list[TakenOff]): The game before each piece Willy took off.
    """
    board: chess.Board
    physical: dict[int, str]
    human_color: bool
    status: str = ""
    evaluation: str = ""
    observation: BoardObservation | None = None
    uncertain: set[int] = field(default_factory=set)
    mood: str = "idle"
    speech: str = ""
    thought: Thought | None = None
    focus: chess.Move | None = None
    intent: chess.Move | None = None
    taking: int | None = None
    strikes: int = 0
    patience: int = 0
    earlier: list["TakenOff"] = field(default_factory=list)


@dataclass(frozen=True)
class TakenOff:
    """
    A piece Willy took off the board for moves against the rules.

    Attributes:
        before (chess.Board): The game until then, with its moves (the next part starts from the position without it).
        square (int): Where the piece stood.
        piece (chess.Piece): The piece, as the rules know it.
        count (int): The how-manyth move against the rules it was.
    """
    before: chess.Board
    square: int
    piece: chess.Piece
    count: int

    def describe(self) -> str:
        colour = "white" if self.piece.color == chess.WHITE else "black"
        return (f"Willy took the {colour} {chess.piece_name(self.piece.piece_type)} on "
                f"{chess.square_name(self.square)} off the board, for illegal move {self.count}.")


@dataclass(frozen=True)
class WrongMove:
    """
    One clear move against the rules on the board: one of the human's pieces from one square to another.

    Attributes:
        from_square (int): Where it stood.
        to_square (int): Where it stands now.
        symbol (str): The piece as it physically stands (a pawn that counts as a queen is still a pawn).
        reason (str): "moves" (the piece cannot move like that), "check" (the own king would be in check), "rules".
        captured (str | None): The piece of Willy's that stood on to_square and is off the board now.
    """
    from_square: int
    to_square: int
    symbol: str
    reason: str
    captured: str | None = None

    def shown(self, physical: dict[int, str]) -> dict[int, str]:
        """The pieces the board shows, from those that stood there before."""
        out = {sq: sym for sq, sym in physical.items() if sq != self.from_square}
        out[self.to_square] = self.symbol
        return out

    def describe(self, board: chess.Board) -> str:
        name = chess.piece_name(board.piece_type_at(self.from_square)).capitalize()
        return (f"{name} {chess.square_name(self.from_square)}-{chess.square_name(self.to_square)} is not allowed: "
                f"{REASONS[self.reason]}")


class View(Protocol):
    """The display and the human's input (src/ui/pygame_view.py, or a scripted one in tests)."""

    def show(self, state: ViewState) -> None: ...

    def wait(self, state: ViewState, accept: set[str]) -> str:
        """
        Block until the human gives one of the accepted answers: "done" (space bar), "recalibrate" (R),
        "quit" (Esc), "resign" (G), "draw" (D), "force" (F), a promotion piece "q", "r", "b", "n", a colour
        "white" (W) or "black" (B), or, if "move" is accepted, "move:<uci>" (two squares clicked).
        """
        ...


class GameAborted(Exception):
    """The human ended the game."""


class GameOver(Exception):
    """The game ended before the board says so: resigned or drawn by agreement."""

    def __init__(self, result: str, reason: str):
        super().__init__(reason)
        self.result, self.reason = result, reason


def _changes(physical: dict[int, str], seen: dict[int, str], white: bool) -> tuple[list[int], list[int]]:
    """The squares one side's pieces left and arrived on, from the position before to what the camera sees."""
    left = [sq for sq, sym in physical.items() if sym.isupper() == white
            and not (sq in seen and seen[sq].isupper() == white)]
    arrived = [sq for sq, sym in seen.items() if sym.isupper() == white
               and not (sq in physical and physical[sq].isupper() == white)]
    return left, arrived


def find_wrong_move(board: chess.Board, physical: dict[int, str], seen: dict[int, str],
                    human: bool) -> WrongMove | None:
    """
    The move against the rules the picture shows, if it shows exactly one: one of the human's pieces gone from its
    square and arrived on another, Willy's pieces where they were (but the one it took).

    Args:
        board (chess.Board): The position before the move.
        physical (dict[int, str]): Square -> piece that stood there before the move.
        seen (dict[int, str]): Square -> piece the camera sees there now.
        human (bool): The human's colour.
    """
    left, arrived = _changes(physical, seen, human)
    willy_left, willy_arrived = _changes(physical, seen, not human)
    if len(left) != 1 or len(arrived) != 1 or willy_arrived or not set(willy_left) <= set(arrived):
        return None
    piece = board.piece_at(left[0])
    if piece is None:
        return None
    promotion = chess.QUEEN if piece.piece_type == chess.PAWN and chess.square_rank(arrived[0]) in (0, 7) else None
    move = chess.Move(left[0], arrived[0], promotion=promotion)
    if move in board.legal_moves:
        return None
    reason = "moves" if not board.is_pseudo_legal(move) else "check" if board.is_into_check(move) else "rules"
    return WrongMove(left[0], arrived[0], physical[left[0]], reason, physical[arrived[0]] if willy_left else None)


def describe_wrong_move(board: chess.Board, physical: dict[int, str], seen: dict[int, str],
                        human: bool) -> str | None:
    """
    Why the board shows no legal move, when the picture shows one clear wrong move: a piece of Willy's
    moved, or one of the human's pieces moved from one square to another against the rules.

    Returns:
        str | None: The message for the human; None if the picture shows no single clear move.
    """
    willy_arrived = _changes(physical, seen, not human)[1]
    if willy_arrived:
        return (f"A piece of Willy's now stands on {chess.square_name(willy_arrived[0])}. "
                f"Please put it back, then press space.")
    wrong = find_wrong_move(board, physical, seen, human)
    return None if wrong is None else f"{wrong.describe(board)}. Please take it back, then press space."


class ChessGame:
    """
    Human against Willy on the real board.

    Args:
        mapping (ChessBoardMapping3D): Camera, detector and board geometry.
        engine: Something with best_move(board) and evaluation(board) (src/game/engine.Engine).
        mover (RobotMover): Executes transfers with the robot.
        view (View): Display and input.
        config (ChessConfig): Game settings.
        go_look (Callable[[], Any]): Brings the arm to where the camera sees the board (the look pose
            for a wrist camera, out of the view for a fixed one).
        log (GameLog | None): Keeps every view and every decision of this game.
        views (int): Look poses to use: go_look(view) for view 1 and up, where one view cannot settle something.
    """

    def __init__(self, mapping: ChessBoardMapping3D, engine: Any, mover: RobotMover, view: View,
                 config: ChessConfig, go_look: Callable[..., Any], log: GameLog | None = None, views: int = 1):
        self.mapping, self.engine, self.mover, self.view = mapping, engine, mover, view
        self.config, self.go_look, self.log, self.views = config, go_look, log, views
        self._unsettled: list[int] = []     # views of the log whose position only the next committed move will tell
        self._last_view: int | None = None
        self.human = chess.WHITE if config.game.human_color == "white" else chess.BLACK
        self.tracker: MoveTracker | None = None
        self.graveyard_used = 0
        # Where each piece was last seen (square -> board x, y in metres): a piece hidden in the current
        # view is gripped where it was last seen, not at its square centre (pieces stand up to ~1 cm off).
        self.piece_xy: dict[int, np.ndarray] = {}
        self.state = ViewState(chess.Board(), {}, self.human)
        self.persona = Persona()
        self._willy_eval: float | None = None       # pawns from Willy's view, as the display last showed them
        self._willy_eval_before: float | None = None    # ... and before that
        self._reacted_at = float("-inf")            # when Willy last remarked on the human's move
        self.illegal_moves = 0                      # moves against the rules in this game
        self._last_wrong: tuple[int, int] | None = None     # the one the last look showed, if it did
        self.earlier: list[TakenOff] = []
        self.state.patience = config.game.take_off_from - 1 if config.game.take_off else 0

    # ------------------------------------------------------------------ main loop
    def run(self) -> str:
        """Play one game from the starting position. Returns the result ("1-0", "0-1", "1/2-1/2", "*" aborted)."""
        try:
            self._react("hello")
            self.choose_color()
            self._event("game", human="white" if self.human == chess.WHITE else "black", views=self.views,
                        promotion_mode=self.config.game.promotion_mode)
            self.calibrate()
            while not self.tracker.board.is_game_over():
                if self.tracker.board.turn == self.human:
                    self.human_turn()
                else:
                    self.willy_turn()
            outcome = self.tracker.board.outcome()
            result, reason = outcome.result(), OUTCOMES.get(outcome.termination, "")
        except GameAborted:
            self._react("bye")
            self._status("Game aborted.")
            self._log_result("*", "aborted")
            return "*"
        except GameOver as over:
            result, reason = over.result, over.reason
        self._react(self._ending(result, reason))
        self._status(f"Game over: {result} ({reason}).")
        self._log_result(result, reason)
        return result

    def ask_new_game(self) -> bool:
        """After a game: another one? The human sets up the starting position and clears the graveyard."""
        try:
            return self._ask(f"{self.state.status} New game: set up the starting position, clear the graveyard, "
                             f"press space. Esc: end.", {"done", "quit"}) == "done"
        except GameAborted:                                   # the window was closed
            return False

    def choose_color(self) -> None:
        """The human picks a side; the space bar keeps the one preselected in the config."""
        preset = "White" if self.human == chess.WHITE else "Black"
        answer = self._ask(f"Which colour do you play? W = White, B = Black, space = {preset}.",
                           {"white", "black", "done", "quit"})
        if answer == "quit":
            raise GameAborted
        if answer != "done":
            self.human = chess.WHITE if answer == "white" else chess.BLACK
        self.state.human_color = self.human

    def calibrate(self, shown: dict[int, str] | None = None) -> None:
        """
        Calibrate until it succeeds; in a running game against the position the tracker knows.

        Args:
            shown (dict[int, str] | None): The pieces the board shows instead, when it is ahead of the tracker: Willy's
                move is made but not confirmed yet.
        """
        expected = shown if shown is not None else self.tracker.physical if self.tracker is not None else None
        self._react("calibrating")
        while True:
            self._status("Willy is calibrating the board ...")
            # a refusal is fail-closed and every grab has new noise, so try a few frames before asking, every look pose
            attempts = []
            for attempt in range(max(3, self.views)):
                view = attempt % self.views
                result = self._safely("Calibration", lambda: (self._go_look(view), self.mapping.calibrate(expected))[1])
                attempts.append(self._log_calibration(view, result))
                if result.calibration is not None:
                    break
            if result.calibration is not None:
                break
            # it compares with the position before the human's move: a move already made is a piece out of place
            human_to_move = self.tracker is not None and shown is None and self.tracker.board.turn == self.human
            taken_back = " If you have moved already, take the move back and play it again afterwards." if human_to_move else ""
            answer = self._ask(f"Calibration refused: {result.reason}.{taken_back} Check the board, then press space.",
                               {"done", "quit"})
            if answer == "quit":
                raise GameAborted
        if self.tracker is None:
            t = self.config.tracking
            self.tracker = MoveTracker(result.calibration, sigma=t.sigma_squares, gate=t.gate,
                                       class_cost=t.class_cost, promotion_mode=self.config.game.promotion_mode)
        else:
            self.tracker.recalibrated(result.calibration)
        known = shown if shown is not None else self.tracker.physical
        if self.log is not None:            # the refused tries showed this position too: the board did not change
            self.log.settled([number for number in attempts if number is not None], known, "calibration")
        self.piece_xy = {}
        observation = self._observe(moment="calibrated", truth=known, how="calibration")
        if observation is not None:
            self._remember_positions(observation, changed=set())
        self._sync()

    # ------------------------------------------------------------------ turns
    def human_turn(self) -> None:
        """Wait for the human's move and recognise it; ask whenever it cannot be confirmed."""
        prompt = "Your move. Press space when you are done (or click the move)."
        self._react(your_move_event(self._willy_eval))
        while True:
            answer = self._ask(prompt, {"done", "recalibrate", "quit", "move", "resign", "draw"})
            if answer == "quit":
                raise GameAborted
            if answer == "resign":
                raise GameOver("0-1" if self.human == chess.WHITE else "1-0", "you resigned")
            if answer == "draw":
                score = self._safely("Stockfish", lambda: self.engine.score(self.tracker.board, not self.human))
                if score <= DRAW_ACCEPT_CP:
                    raise GameOver("1/2-1/2", "draw agreed")
                self._react("draw_declined")
                prompt = "Willy declines the draw. Your move."
                continue
            if answer == "recalibrate":
                self.calibrate()
                continue
            if answer.startswith("move:"):
                if self._commit_typed(answer[5:]):
                    self._react_to_human_move()
                    return
                prompt = f"{answer[5:]} is not a legal move here. Try again."
                continue

            mark = len(self._unsettled)
            observation = self._clear_view(self._observe(moment="your_move"), moment="your_move")
            if observation is None:
                self._react("board_moved")
                prompt = "The board was moved. Press R to calibrate again."
                continue
            before = dict(self.tracker.physical)
            result = self._step(observation)
            for other in self._other_views() if result.flagged else ():
                observation, result = other, self._step(other)
                if not result.flagged:
                    break
            self.state.observation = observation
            wrong = self._wrong_move(observation) if result.flagged else None
            if wrong is not None:
                prompt = self._answer_wrong_move(wrong, mark)
                continue
            self._last_wrong = None
            if result.flagged:
                self.state.uncertain = self._disagreeing_squares(observation)
                self._react("unsure")
                prompt = describe_wrong_move(self.tracker.board, self.tracker.physical, observation.occupancy(),
                                             self.human) or (f"Move not recognised for sure (NLL {result.nll:.1f}). "
                                                             f"Check the board and press space, or click the move.")
                continue
            if result.move is None:
                self._settle("no_move")
                self._react("unsure")
                prompt = "I see no move. Press space when you have moved."
                continue
            if result.move.promotion:     # also with the new piece on the board: one wrong class is another game
                self._ask_promotion(result.move)
            self._settle("seen")
            self._remember_positions(observation, self._changed_squares(before))
            self.state.uncertain = set()
            self._sync()
            self._react_to_human_move()
            return

    def willy_turn(self) -> None:
        """Stockfish moves, the robot executes, the camera confirms."""
        board = self.tracker.board
        self._status("Willy is thinking ...")
        move = self._safely("Stockfish", lambda: self._think(board))
        san = board.san(move)
        self._event("engine", move=move.uci(), san=san, fen=board.fen())
        event, facts = decided_event(board, move)
        self._react(event, **facts)
        self.state.focus, self.state.intent = None, move

        def plan(observation: BoardObservation) -> list[Transfer]:
            self._remember_positions(observation, changed=set())
            transfers = plan_transfers(board, self.tracker.physical, move, self.piece_xy, observation.calib.square,
                                       self.graveyard_used, self.config.robot.graveyard)
            self._status(f"Willy plays {san}: " + ", ".join(t.describe() for t in transfers))
            return transfers

        execution = self._carry(plan, self.tracker.physical, san, move=move.uci())
        if execution.ok and move.promotion and self.config.game.promotion_mode == "piece_swapped":
            name = PIECE_NAMES[chess.piece_symbol(move.promotion)]
            self._ask(f"Please replace the pawn on {chess.square_name(move.to_square)} with the new piece ({name}), "
                      f"then press space.", {"done"})
        self._confirm_own_move(move, san)

    def _think(self, board: chess.Board) -> chess.Move:
        """
        Stockfish's move, with Willy visibly thinking: he looks at his best move so far while the engine
        searches, says when he changes his mind, and once it has answered weighs the other candidates
        once more, until game.think_show_s has passed. An engine that cannot be watched just answers.
        """
        think = getattr(self.engine, "think", None)
        if think is None:
            return self.engine.best_move(board)
        started = time.monotonic()
        quiet_until = self._reacted_at + REACTION_S         # Willy's word on the human's move first
        if time.monotonic() >= quiet_until:
            self._react("thinking")
        self.state.thought = self.state.focus = None
        said = {"move": None, "at": float("-inf")}

        def watch(thought: Thought) -> None:
            self.state.thought = thought
            top = thought.candidates[0] if thought.candidates else None
            now = time.monotonic()
            if top is not None and top.move != said["move"] and now >= quiet_until and now - said["at"] >= PONDER_S:
                self._react("considering" if said["move"] is None else "reconsidering", san=top.san)
                said.update(move=top.move, at=now)
            self.state.focus = top.move if top is not None else None
            self.view.show(self.state)

        move = think(board, watch)
        others = [c for c in self.state.thought.candidates if c.move != move] if self.state.thought else []
        for candidate in others:
            if time.monotonic() - started >= self.config.game.think_show_s - PONDER_S:
                break
            self.state.focus = candidate.move
            self._react("pondering", san=candidate.san)
            self._hold(PONDER_S)
        self.state.focus = move
        self._hold(self.config.game.think_show_s - (time.monotonic() - started))
        return move

    def _hold(self, seconds: float) -> None:
        """Keep the window alive for a moment while nothing else happens: Willy's face goes on moving."""
        until = time.monotonic() + seconds
        while time.monotonic() < until:
            self.view.show(self.state)
            time.sleep(1 / 30)

    def _react(self, event: str, **facts: str) -> None:
        """Willy's mood and line for what just happened (src/game/persona.py); shown with the next update."""
        self.state.mood, self.state.speech = self.persona.react(event, **facts)

    def _react_to_human_move(self) -> None:
        """Willy's word on the move just committed: a piece of his taken, a check, a big swing either way."""
        board = self.tracker.board.copy()
        move = board.pop()
        event, facts = human_move_event(board, move, self._willy_eval_before, self._willy_eval)
        self._react(event, **facts)
        self._reacted_at = time.monotonic()
        self.view.show(self.state)

    def _ending(self, result: str, reason: str) -> str:
        """The persona's event for how the game ended."""
        if reason == "you resigned":
            return "resigned"
        if result == "1/2-1/2":
            return "draw_agreed" if reason == "draw agreed" else "draw"
        return "won" if (result == "1-0") == (self.human == chess.BLACK) else "lost"

    def _after_a_stop(self, execution: MoveExecution) -> None:
        """
        The controller stopped the arm: it met something (a piece, the board, a person), which no refusal
        does. Nothing moves until a person has looked at the cell and the controller stands ready again.
        """
        what = execution.failed.describe() if execution.failed else "Willy's move"
        self._event("stopped", stop=execution.stopped, transfer=what)
        stop = execution.stopped
        while stop:
            answer = self._ask(f"The arm stopped ({stop}) during {what}: it touched something. Check the gripper, the "
                               f"pieces and the board, clear the stop on the pendant, then press space.",
                               {"done", "quit"})
            if answer == "quit":
                raise GameAborted
            stop = self.mover.controller_stop()

    def _carry(self, plan: Callable[[BoardObservation], list[Transfer]], shown: dict[int, str], task: str,
               **record: Any) -> MoveExecution:
        """
        Carry pieces with the arm. Before it moves, nothing but pieces may be over the board, and the transfers are
        planned from that view. A transfer that does not finish is handed to the person at the board: a stop needs
        somebody to look at the cell, a piece still held a hand under the gripper, the rest is done by hand.

        Args:
            plan (Callable): The transfers, from the view taken just before.
            shown (dict[int, str]): The pieces the board shows before the arm moves.
            task (str): What the arm carries out, for the person to finish ("Nf3", "putting the knight back on g1").
            record: For the log's execution event.
        """
        said = self.state.mood, self.state.speech
        look = {"moment": "before_arm", "truth": shown, "how": "before_arm"}
        observation = self._clear_view(self._observe(**look), **look)
        while observation is None:
            self._react("board_moved")
            if self._ask("The board was moved. Press R to calibrate again.", {"recalibrate", "quit"}) == "quit":
                raise GameAborted
            self.calibrate(shown)
            observation = self._clear_view(self._observe(**look), **look)
        self.state.mood, self.state.speech = said

        transfers = plan(observation)
        try:
            execution = self.mover.execute(transfers, observation)
        except Exception as err:  # noqa: BLE001 (never repeated: a piece may have moved already)
            execution = MoveExecution(ok=False, holding=True, message=f"Willy's move aborted ({err})",
                                      stopped=self.mover.controller_stop())
        self._event("execution", **record, ok=execution.ok, holding=execution.holding, message=execution.message,
                    done=[t.describe() for t in execution.done],
                    failed=execution.failed.describe() if execution.failed else None,
                    reports=[str(report) for report in execution.reports])
        # every planned slot, whatever the arm finished: the piece may lie there, by the arm's hand or a person's
        self.graveyard_used += sum(1 for t in transfers if t.to_square is None)

        if not execution.ok:
            self._react("stopped" if execution.stopped else "refused")
            if execution.stopped:
                self._after_a_stop(execution)
            if execution.holding:
                # never open the hand by itself: the piece would fall onto the board
                self._ask(f"{execution.message}. If Willy still holds a piece, please put a hand under the gripper; "
                          f"space opens it.", {"done"})
                self._safely("Opening the gripper", self.mover.release)
            self._ask(f"{execution.message}. Please finish {task} by hand, then press space.", {"done"})
        return execution

    def _confirm_own_move(self, move: chess.Move, san: str) -> None:
        """Willy's move is known, so the camera only has to confirm it."""
        shown = physical_after(self.tracker.board, self.tracker.physical, move, self.config.game.promotion_mode)
        observation, forced = self._confirm(shown, f"After {san} the board looks different than expected", move)
        self._commit_own_move(move, observation, "forced" if forced else "robot")

    def _confirm(self, shown: dict[int, str], differs: str,
                 move: chess.Move | None = None) -> tuple[BoardObservation, bool]:
        """
        Look until the camera confirms that the board shows these pieces. From the second time it does not, the
        person at the board may overrule the camera: a piece the detector keeps missing must not end the game.

        Args:
            shown (dict[int, str]): The pieces the board should show.
            differs (str): What to tell the person when it does not.
            move (chess.Move | None): The move that led there, for the log.

        Returns:
            tuple[BoardObservation, bool]: The view that confirmed it, and whether the person overruled the camera.
        """
        doubts = 0
        while True:
            observation = self._observe(moment="after_arm")
            accept = {"done", "recalibrate", "quit"}
            if observation is not None:
                ok, nll = self._verify(observation, shown, move)
                for other in self._other_views() if not ok else ():
                    observation = other
                    ok, nll = self._verify(other, shown, move)
                    if ok:
                        break
                self.state.observation = observation
                if ok:
                    self.state.uncertain = set()
                    return observation, False
                doubts += 1
                self.state.uncertain = self._disagreeing_squares(observation, shown)
                prompt = f"{differs} (NLL {nll:.1f}). Please fix it, then press space."
                if doubts >= 2:
                    accept.add("force")
                    prompt += " F: the board is right, play on."
            else:
                self._react("board_moved")
                prompt = "The board was moved. Press R to calibrate again."
            answer = self._ask(prompt, accept)
            if answer == "quit":
                raise GameAborted
            if answer == "recalibrate":
                self.calibrate(shown)
            if answer == "force":
                return observation, True

    # ------------------------------------------------------------------ moves against the rules
    def _wrong_move(self, observation: BoardObservation) -> WrongMove | None:
        """The move against the rules this view shows, if the whole board agrees with it as it would with a move."""
        wrong = find_wrong_move(self.tracker.board, self.tracker.physical, observation.occupancy(), self.human)
        if wrong is None:
            return None
        ok, nll = self.tracker.explains(observation.frame, wrong.shown(self.tracker.physical), observation.calib)
        self._event("wrong_move", view=self._last_view, ok=bool(ok), nll=nll, reason=wrong.reason,
                    move=chess.square_name(wrong.from_square) + chess.square_name(wrong.to_square))
        return wrong if ok else None

    def _answer_wrong_move(self, wrong: WrongMove, mark: int) -> str:
        """
        A move against the rules, clear on the board. The first ones of a game the human takes back. From
        game.put_back_from on Willy carries the piece back himself; with game.take_off, from game.take_off_from on he takes it off the
        board instead, out of the game; but never the king, nor a piece whose loss would end the game or break the
        position (he carries that one back). The same move shown again (not taken back) costs nothing more, but where
        Willy carries pieces back at all, he does so with this one.

        Args:
            wrong (WrongMove): What the board shows.
            mark (int): The views of this look start there in the waiting list: they show this move.

        Returns:
            str: What to ask the human next.
        """
        g, board = self.config.game, self.tracker.board
        a, b = wrong.from_square, wrong.to_square
        again = (a, b) == self._last_wrong
        if not again:
            self.illegal_moves += 1
        self._last_wrong = (a, b)
        self.state.strikes = self.illegal_moves
        shown = wrong.shown(self.tracker.physical)
        self._settle_since(mark, shown, "wrong_move")
        self._event("illegal_move", move=chess.square_name(a) + chess.square_name(b), reason=wrong.reason,
                    count=self.illegal_moves, again=again)
        piece = chess.piece_name(board.piece_type_at(a))
        told = wrong.describe(board)

        take = not again and g.take_off and self.illegal_moves >= g.take_off_from
        put_back = g.put_back_from > 0 and (again or self.illegal_moves >= g.put_back_from)
        if take and not self._may_take_off(a):
            take, put_back = False, True
            self._react("spared", piece=piece)
        elif take:
            self._react("take_off", piece=piece)
        elif put_back:
            warn = g.take_off and self.illegal_moves == g.take_off_from - 1
            self._react("last_warning" if warn else "put_back", piece=piece)
        else:
            self._react(ILLEGAL_EVENTS[wrong.reason], piece=piece)
            return f"{told}. Please take it back, then press space."

        self._status(f"{told}. " + (f"Willy takes your {piece} off the board." if take else "Willy puts it back."))
        self._hold(REACTION_S)                          # his word first, then the arm
        if take:
            self.state.taking = a
            task = f"taking the {piece} on {chess.square_name(b)} off the board"
        else:
            self.state.intent = chess.Move(b, a)
            task = f"putting the {piece} back on {chess.square_name(a)}"

        def plan(observation: BoardObservation) -> list[Transfer]:
            located = self.tracker.locate(observation.frame, observation.calib, shown)
            centres = square_centres(observation.calib.square)
            label = located.get(b)
            xy = observation.grasp_positions[label] if label in observation.grasp_positions else centres[b]
            if take:
                graveyard = self.config.robot.graveyard
                return [Transfer(wrong.symbol, xy, graveyard_slot(self.graveyard_used, observation.calib.square, graveyard),
                                 0.0, graveyard.table_offset_m, b, None)]
            return [Transfer(wrong.symbol, xy, centres[a], 0.0, 0.0, b, a)]

        self._carry(plan, shown, task, wrong_move=chess.square_name(a) + chess.square_name(b), take_off=take)
        self.state.intent = self.state.taking = None
        if take:
            taken = TakenOff(self.tracker.take_off(a), a, board.piece_at(a), self.illegal_moves)
            self.earlier.append(taken)
            self.state.earlier = list(self.earlier)
            self._event("taken_off", square=chess.square_name(a), piece=taken.piece.symbol(),
                        fen=self.tracker.board.fen())
        self._sync()
        if wrong.captured:
            self._ask(f"Please put my {PIECE_NAMES[wrong.captured.lower()]} back on {chess.square_name(b)}, "
                      f"then press space.", {"done"})
        observation, forced = self._confirm(self.tracker.physical, "The board does not show the position Willy "
                                                                   "expects yet")
        self._settle("forced" if forced else "taken_off" if take else "put_back")
        self._remember_positions(observation, {a, b})
        self._last_wrong = None
        self._sync()
        if take:
            return f"Willy took your {piece} off the board. Your move."
        self._react("again")
        return "Your move again, a legal one. Press space when you are done (or click the move)."

    def _may_take_off(self, square: int) -> bool:
        """Whether Willy may take this piece off: not the king, and the game goes on from a sound position."""
        if self.tracker.board.piece_type_at(square) == chess.KING:
            return False
        after = board_without(self.tracker.board, square)
        return after.is_valid() and not after.is_game_over()

    def _commit_own_move(self, move: chess.Move, observation: BoardObservation, how: str) -> None:
        before = dict(self.tracker.physical)
        self.tracker.commit(move)
        self._settle(how)
        self._remember_positions(observation, self._changed_squares(before))
        self.state.uncertain, self.state.intent = set(), None
        self._sync()

    # ------------------------------------------------------------------ helpers
    def _go_look(self, view: int = 0) -> None:
        self.go_look(view) if view else self.go_look()

    def _observe(self, view: int = 0, moment: str = "look", truth: dict[int, str] | None = None,
                 how: str = "") -> BoardObservation | None:
        """
        Look at the board from a look pose; None if it moved since the calibration.

        Args:
            view (int): The look pose.
            moment (str): For the log: when this view was taken.
            truth (dict[int, str] | None): The position this view shows, where that is known already; otherwise the
                next committed move settles it.
            how (str): For the log: how the truth is known.
        """
        observation = self._safely("Looking at the board", lambda: (self._go_look(view), self.mapping.observe())[1])
        self.state.observation = observation
        if self.log is not None:
            ply = self.tracker.board.ply() if self.tracker is not None else 0
            self._last_view = self.log.view(observation.frame, moment=moment, look=view, ply=ply, calib=observation.calib,
                                            camera_to_base=observation.camera_to_base,
                                            board_to_base=observation.board_to_base, status=observation.pose_status,
                                            hands=observation.hands)
            if not observation.board_moved:        # a moved board is a pose nobody can vouch for: no label from it
                if truth is not None:
                    self.log.settled([self._last_view], truth, how)
                else:
                    self._unsettled.append(self._last_view)
        return None if observation.board_moved else observation

    def _other_views(self) -> Iterator[BoardObservation]:
        """The board from the other look poses, one at a time, for what the first view could not settle."""
        for view in range(1, self.views):
            self._status("Willy looks again from another side ...")
            observation = self._observe(view, moment="other_look")
            if observation is not None:
                yield observation

    def _clear_view(self, observation: BoardObservation | None, **look: Any) -> BoardObservation | None:
        """
        Wait until nothing but pieces is over the board (a hand, an arm, in the depth) and no hand is in
        the picture (Willy's palm detector, beside the board too). After the second request the space bar
        goes on anyway: a piece the detector missed looks the same.
        """
        for attempt in range(3):
            seen = 0 if observation is None else observation.intruders()
            hands = 0 if observation is None else observation.hands
            if observation is None or attempt == 2 or (seen <= INTRUDER_PX and not hands):
                return observation
            extra = "" if attempt == 0 else " If there is nothing: space goes on anyway."
            self._react("hand")
            what = ("Willy sees a hand in the picture" if hands
                    else f"Something is over the board, a hand? ({seen} px)")
            if self._ask(f"{what}. Please take it away, then press space.{extra}",
                         {"done", "quit"}) == "quit":
                raise GameAborted
            observation = self._observe(**look)
        return observation

    def _safely(self, what: str, action: Callable[[], Any]) -> Any:
        """Run an action of the camera or the arm; if it fails, show why and let the human try again."""
        while True:
            try:
                return action()
            except (GameAborted, GameOver):
                raise
            except Exception as err:  # noqa: BLE001 (the person at the board decides, the game goes on)
                if self._ask(f"{what} failed: {err}. Space: try again, Esc: end.", {"done", "quit"}) == "quit":
                    raise GameAborted from err

    def _remember_positions(self, observation: BoardObservation, changed: set[int]) -> None:
        """
        Update where each piece of the tracked position stands: as seen now if it was matched, else as last
        seen, else (a piece that just arrived and is hidden) its square centre.
        """
        located = self.tracker.locate(observation.frame, observation.calib)
        seen = {sq: observation.grasp_positions[k] for sq, k in located.items() if k in observation.grasp_positions}
        centres = square_centres(observation.calib.square)
        self.piece_xy = {sq: seen[sq] if sq in seen else
                         (self.piece_xy[sq] if sq in self.piece_xy and sq not in changed else centres[sq])
                         for sq in self.tracker.physical}

    def _changed_squares(self, before: dict[int, str]) -> set[int]:
        """Squares whose physical piece changed with the last move (arrived, left or was replaced)."""
        after = self.tracker.physical
        return {sq for sq in set(before) | set(after) if before.get(sq) != after.get(sq)}

    def _ask_promotion(self, move: chess.Move) -> None:
        answer = self._ask("Promote to? Q = queen, R = rook, B = bishop, N = knight.", set(PROMOTION_KEYS))
        if PROMOTION_KEYS[answer] != move.promotion:
            self.tracker.undo()
            self.tracker.commit(chess.Move(move.from_square, move.to_square, promotion=PROMOTION_KEYS[answer]))

    def _commit_typed(self, uci: str) -> bool:
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            return False
        board = self.tracker.board
        before = dict(self.tracker.physical)
        if move not in board.legal_moves:
            promoted = chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)
            if promoted not in board.legal_moves:
                return False
            self.tracker.commit(promoted)
            self._ask_promotion(promoted)
        else:
            self.tracker.commit(move)
        for square in self._changed_squares(before):          # last seen there: another piece, or none
            self.piece_xy.pop(square, None)
        self._settle("clicked")
        self.state.uncertain = set()
        self._sync()
        return True

    def _disagreeing_squares(self, observation: BoardObservation, shown: dict[int, str] | None = None) -> set[int]:
        """Squares where the free-mode view and the tracked position (or `shown`) disagree on the colour, for the display."""
        seen = {sq: sym.isupper() for sq, sym in observation.occupancy().items()}
        known = {sq: sym.isupper() for sq, sym in (self.tracker.physical if shown is None else shown).items()}
        return {sq for sq in set(seen) | set(known) if seen.get(sq) != known.get(sq)}

    def _sync(self) -> None:
        self.state.board = self.tracker.board.copy() if self.tracker is not None else chess.Board()
        self.state.physical = dict(self.tracker.physical) if self.tracker is not None else {}
        try:
            self.state.evaluation = self.engine.evaluation(self.state.board)
        except Exception:  # noqa: BLE001 (the display can live without an evaluation)
            self.state.evaluation = ""
        self._willy_eval_before, self._willy_eval = self._willy_eval, willy_pawns(self.state.evaluation, not self.human)
        self.view.show(self.state)

    def _status(self, text: str) -> None:
        self.state.status = text
        self._event("status", text=text)
        self.view.show(self.state)

    def _ask(self, text: str, accept: set[str]) -> str:
        self.state.status = text
        answer = self.view.wait(self.state, accept)
        self._event("ask", text=text, accept=sorted(accept), answer=answer)
        return answer

    # ------------------------------------------------------------------ the log
    def _event(self, kind: str, **fields: Any) -> None:
        if self.log is not None:
            self.log.event(kind, **fields)

    def _settle_since(self, mark: int, physical: dict[int, str], how: str) -> None:
        """The views from `mark` on in the waiting list showed these pieces, not what the next committed move will."""
        views, self._unsettled = self._unsettled[mark:], self._unsettled[:mark]
        if self.log is not None and views:
            self.log.settled(views, physical, how)

    def _settle(self, how: str) -> None:
        """The views waiting for it showed the position the tracker has now."""
        if self.log is not None and self._unsettled:
            self.log.settled(self._unsettled, self.tracker.physical, how)
        self._unsettled = []

    def _step(self, observation: BoardObservation) -> TrackResult:
        result = self.tracker.step(observation.frame, observation.calib)
        self._event("step", view=self._last_view, flagged=result.flagged, nll=result.nll,
                    move=result.move.uci() if result.move else None, ambiguous=result.promotion_ambiguous)
        return result

    def _verify(self, observation: BoardObservation, shown: dict[int, str],
                move: chess.Move | None = None) -> tuple[bool, float]:
        ok, nll = self.tracker.explains(observation.frame, shown, observation.calib)
        self._event("verify", view=self._last_view, ok=bool(ok), nll=nll, move=move.uci() if move else None)
        return ok, nll

    def _log_calibration(self, look: int, result: Any) -> int | None:
        """Keep the frame a calibration attempt judged, a refused one above all, and return its number."""
        if self.log is None or self.mapping.calibration_view is None:
            return None
        frame, camera_to_base = self.mapping.calibration_view
        self._event("calibration", reason=result.reason, corners=result.corners, reprojection_px=result.reprojection_px,
                    normal_deg=result.normal_deg, class_mismatches=result.class_mismatches)
        calib = result.calibration
        board_to_base = None if calib is None else self.mapping.board_to_base
        return self.log.view(frame, moment="calibration", look=look, calib=calib, camera_to_base=camera_to_base,
                             board_to_base=board_to_base, status=result.reason)

    def _log_result(self, result: str, reason: str) -> None:
        boards = [taken.before for taken in self.earlier] + ([self.tracker.board] if self.tracker is not None else [])
        moves = [move.uci() for board in boards for move in board.move_stack]
        self._event("result", result=result, reason=reason, moves=moves)
