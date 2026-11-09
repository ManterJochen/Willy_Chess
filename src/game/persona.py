"""
Willy as a character: his mood and what he says, from what happens in a game.

Game logic only, nothing drawn: the window turns the mood into a face (src/ui/face.py). Each event has a mood and a
few lines to choose from, so the same moment does not sound the same twice in a row.
"""
import random

import chess

# event -> (mood, lines); {san} is a move, {piece} a piece name
LINES: dict[str, tuple[str, tuple[str, ...]]] = {
    "hello": ("happy", ("Hello! Shall we play?", "Ready when you are.", "My gripper is itching for a game.")),
    "calibrating": ("focused", ("Let me have a good look at the board.", "Counting squares... 64. Good.",
                                "Hold still, board.")),
    "your_move": ("idle", ("Your move.", "Take your time.", "Go on, I'm watching.")),
    "your_move_winning": ("smug", ("Your move. No pressure.", "Your move. Think carefully!")),
    "your_move_losing": ("worried", ("Your move... be nice to me.", "Your move. I'm a little nervous.")),
    "thinking": ("thinking", ("Hmm...", "Let me think...", "Now then...")),
    "considering": ("thinking", ("What about {san}?", "{san}, maybe?", "Hmm, {san}...")),
    "reconsidering": ("thinking", ("No, wait. {san}!", "Or rather {san}...", "Hmm, {san} looks better.")),
    "pondering": ("thinking", ("Or {san}...?", "{san} is tempting too...", "Then again, {san}...")),
    "decided": ("happy", ("{san}. Let's see what you make of that.", "I'll go with {san}.", "{san}!")),
    "decided_capture": ("smug", ("{san}. I'll take that {piece}, thank you!", "{san}. Your {piece} is mine.")),
    "decided_check": ("smug", ("{san}. Check!", "{san}, and that's check.")),
    "decided_mate": ("happy", ("{san}. Checkmate, I believe!",)),
    "lost_piece": ("surprised", ("Hey, that was my {piece}!", "Ouch, my {piece}.")),
    "in_check": ("worried", ("Check? Oh dear.", "Eek, check!")),
    "good_move": ("surprised", ("Oh! I didn't see that coming.", "Hm. Strong move.")),
    "blunder": ("smug", ("Interesting choice...", "Are you sure about that?")),
    "seen": ("idle", ("I see.", "Okay.", "Noted.")),
    "illegal": ("surprised", ("Hey! A {piece} doesn't move like that.", "Nice try. That's not how a {piece} moves.")),
    "illegal_check": ("worried", ("Careful, that leaves your king in check!", "Psst... your king would be in check.")),
    "illegal_rules": ("surprised", ("Hmm, the rules don't allow that here.", "I don't think that's allowed.")),
    "put_back": ("annoyed", ("Again? Let me put that back for you.", "Tsk. Back where it belongs.",
                             "I'll fix that. Hands off the board, please.")),
    "last_warning": ("annoyed", ("Last warning. Once more and that {piece} is mine.",
                                 "Careful. Next time, your {piece} leaves the game.")),
    "take_off": ("angry", ("That's it! Your {piece} is mine now.", "Enough! Your {piece} leaves the game.",
                           "I warned you. Bye bye, {piece}!")),
    "spared": ("annoyed", ("I'd take your {piece}, but that would end the game. Back it goes.",)),
    "again": ("annoyed", ("Now a legal one, please.", "Your move. By the rules this time.", "Again. Properly.")),
    "unsure": ("embarrassed", ("I couldn't quite see your move.", "Sorry, which move was that?")),
    "refused": ("embarrassed", ("Oops, I can't reach that one. Could you help me out?",
                                "That one is tricky for my gripper. A little help?")),
    "stopped": ("dizzy", ("Ouch! I bumped into something.", "Whoa, something got in my way.")),
    "hand": ("worried", ("Hands away please, I'm about to move!", "Careful, I don't want to grab your fingers.")),
    "board_moved": ("surprised", ("Whoa, did the board move?", "Hey, the board isn't where I left it.")),
    "draw_declined": ("smug", ("A draw? Not today!", "Nice try. Play on!")),
    "draw_agreed": ("idle", ("A draw it is. Fair enough.",)),
    "won": ("happy", ("Good game! That was fun.", "Victory! Shall we play again?")),
    "lost": ("sad", ("You got me. Well played!", "I lost... rematch?")),
    "draw": ("idle", ("A draw. Well fought.",)),
    "resigned": ("happy", ("Thanks for the game!",)),
    "bye": ("sleepy", ("Bye for now.",)),
}
SWING_PAWNS = 1.5               # a move that changes the evaluation this much is worth a remark
CLEAR_PAWNS = 2.0               # an advantage this big shows in Willy's face while he waits


class Persona:
    """Picks Willy's mood and line for an event, never the same line twice in a row."""

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self._last: dict[str, str] = {}

    def react(self, event: str, **facts: str) -> tuple[str, str]:
        """(mood, line) for an event of LINES, its {placeholders} filled from facts."""
        mood, lines = LINES[event]
        line = self.rng.choice([text for text in lines if text != self._last.get(event)] or list(lines))
        self._last[event] = line
        return mood, line.format(**facts)


def willy_pawns(evaluation: str, willy: bool) -> float | None:
    """
    The display's evaluation ('+0.35', 'M3', 'M-3', from White's view) in pawns from Willy's view; a mate is
    +-100. None where there is none.
    """
    if not evaluation:
        return None
    try:
        white = (100.0 if int(evaluation[1:]) > 0 else -100.0) if evaluation.startswith("M") else float(evaluation)
    except ValueError:
        return None
    return white if willy == chess.WHITE else -white


def your_move_event(willy_eval: float | None) -> str:
    """How Willy waits for the human's move: smug when well ahead, worried when well behind."""
    if willy_eval is not None and willy_eval >= CLEAR_PAWNS:
        return "your_move_winning"
    if willy_eval is not None and willy_eval <= -CLEAR_PAWNS:
        return "your_move_losing"
    return "your_move"


def decided_event(board: chess.Board, move: chess.Move) -> tuple[str, dict[str, str]]:
    """What Willy says about his own move, in the position before it."""
    facts = {"san": board.san(move)}
    captured = chess.PAWN if board.is_en_passant(move) else board.piece_type_at(move.to_square)
    after = board.copy(stack=False)
    after.push(move)
    if after.is_checkmate():
        return "decided_mate", facts
    if after.is_check():
        return "decided_check", facts
    if captured is not None:
        return "decided_capture", {**facts, "piece": chess.piece_name(captured)}
    return "decided", facts


def human_move_event(board: chess.Board, move: chess.Move, willy_before: float | None,
                     willy_after: float | None) -> tuple[str, dict[str, str]]:
    """
    What Willy says about the human's move, in the position before it: a piece of his taken, a check, or how much
    the evaluation (pawns from Willy's view) swung.
    """
    captured = chess.PAWN if board.is_en_passant(move) else board.piece_type_at(move.to_square)
    if captured is not None:
        return "lost_piece", {"piece": chess.piece_name(captured)}
    if board.gives_check(move):
        return "in_check", {}
    if willy_before is not None and willy_after is not None:
        if willy_after - willy_before <= -SWING_PAWNS:
            return "good_move", {}
        if willy_after - willy_before >= SWING_PAWNS:
            return "blunder", {}
    return "seen", {}
