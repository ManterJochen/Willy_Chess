"""
The game on screen: the digital board, Willy as a character, what he thinks and what the camera sees.

Left, the room around the board: Willy above it, you below it, the evaluation bar beside it. Right, Willy's side,
dark like a teach pendant: his face with a stack light, what he says, the moves he weighs (also drawn on the
board as arrows), what the game needs from you, the camera's view and the moves so far. Once you made a move
against the rules, his patience shows above what he says.

Keys: Space = move done, R = calibrate again, G = resign, D = offer a draw, Esc = end,
W/B = colour, Q/R/B/N = promotion, F = the board is right (where Willy offers it).
Clicking two squares enters a move by hand, for when the camera could not recognise it.
"""
import math
import os
import time

import chess
import cv2 as cv
import numpy as np
import pygame

from src.game.game import GameAborted, ViewState
from src.game.persona import willy_pawns
from src.ui.face import Face

SIZE = (1240, 760)
SQUARE = 76
BOARD = 8 * SQUARE
BOARD_X, BOARD_Y = 64, 60
PANEL_X = BOARD_X + BOARD + 32
PX, PW = PANEL_X + 28, SIZE[0] - PANEL_X - 56          # the panel's content column

FOG = (216, 221, 225)           # the room around the board
FOG_DEEP = (190, 197, 203)
INK = (30, 42, 51)
INK_MUTED = (90, 107, 119)
SLATE = (47, 59, 69)            # Willy's side, a teach pendant
SLATE_HI = (60, 75, 87)
CHALK = (236, 241, 244)
CHALK_MUTED = (160, 175, 186)
SAFETY = (244, 181, 38)         # Willy: his arrows, his thoughts, the amber light
SIGNAL = (43, 179, 163)         # you: your turn, your selection
STOP = (224, 72, 62)
LIGHT, DARK = (232, 211, 169), (168, 123, 85)
WHITE_PIECE, BLACK_PIECE = (250, 248, 244), (28, 30, 34)

FAMILY = ["bahnschrift", "dinalternate", "segoeui", "helveticaneue", "arial"]     # DIN first, the cell's signage
GLYPHS = {"K": "♔", "Q": "♕", "R": "♖", "B": "♗", "N": "♘", "P": "♙",
          "k": "♚", "q": "♛", "r": "♜", "b": "♝", "n": "♞", "p": "♟"}
LETTERS = {"k": "K", "q": "Q", "r": "R", "b": "B", "n": "N", "p": "P"}     # without a font with chess glyphs
SYMBOL_FONTS = ("segoeuisymbol", "applesymbols", "arialunicodems", "dejavusans", "notosanssymbols2", "freeserif")
PROMOTION_KEYS = {pygame.K_q: "q", pygame.K_r: "r", pygame.K_b: "b", pygame.K_n: "n"}
VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
START = {chess.PAWN: 8, chess.KNIGHT: 2, chess.BISHOP: 2, chess.ROOK: 2, chess.QUEEN: 1}
SLIDE_S = 0.28                  # a move on the board slides this long
SLEEPY_S = 45.0                 # Willy dozes off when you take longer than this


class PygameView:
    """Display and input of ChessGame. Use as a context manager so the window always closes."""

    def __init__(self, title: str = "Willy_Chess"):
        pygame.init()
        scaled = os.environ.get("SDL_VIDEODRIVER") != "dummy"     # resized with the window, on any screen
        try:
            self.screen = pygame.display.set_mode(SIZE, pygame.SCALED | pygame.RESIZABLE if scaled else 0)
        except pygame.error:
            self.screen = pygame.display.set_mode(SIZE)
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self.small = pygame.font.SysFont(FAMILY, 15)
        self.body = pygame.font.SysFont(FAMILY, 18)
        self.speech = pygame.font.SysFont(FAMILY, 20)
        self.strong = pygame.font.SysFont(FAMILY, 22, bold=True)
        self.coords = pygame.font.SysFont(FAMILY, 13, bold=True)
        symbol_font = next((f for f in SYMBOL_FONTS if pygame.font.match_font(f)), None)
        self.glyphs = pygame.font.SysFont(symbol_font, 60) if symbol_font else None
        self.mini = pygame.font.SysFont(symbol_font, 24) if symbol_font else None
        self.letters = pygame.font.SysFont(FAMILY, 32, bold=True)
        self.selected: int | None = None
        self.face = Face()
        self._asking = False
        self._waiting_since = time.monotonic()
        self._bar = 0.5                         # the evaluation bar, White's share, as drawn
        self._plies = 0
        self._slide: tuple[chess.Move, float] | None = None
        self._camera: tuple[int, pygame.Surface] | None = None
        self._pieces: dict[tuple[str, int], pygame.Surface] = {}

    def __enter__(self) -> "PygameView":
        return self

    def __exit__(self, *exc) -> None:
        pygame.quit()

    # ------------------------------------------------------------------ View protocol
    def show(self, state: ViewState) -> None:
        pygame.event.pump()             # keeps the window responsive while the robot moves
        self._asking = False
        self._draw(state)
        pygame.display.flip()
        while self._slide is not None:  # a move just made slides into place before the game goes on
            pygame.event.pump()
            self._draw(state)
            pygame.display.flip()
            self.clock.tick(60)

    def wait(self, state: ViewState, accept: set[str]) -> str:
        self.selected = None
        self._asking = True
        self._waiting_since = time.monotonic()
        # What was pressed while Willy thought or moved answers no question asked now: a stray space bar would open
        # the gripper before anybody has a hand under it. A closed window stays in the queue.
        pygame.event.clear((pygame.KEYDOWN, pygame.KEYUP, pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP))
        while True:
            for event in pygame.event.get():
                answer = self._answer(event, state, accept)
                if answer is not None:
                    self.selected = None
                    self._asking = False
                    return answer
            self._draw(state)
            pygame.display.flip()
            self.clock.tick(30)

    def save(self, path: str) -> None:
        """Write the current window to an image file (for documentation and tests)."""
        pygame.image.save(self.screen, path)

    # ------------------------------------------------------------------ input
    def _answer(self, event: pygame.event.Event, state: ViewState, accept: set[str]) -> str | None:
        if event.type == pygame.QUIT:
            raise GameAborted
        if event.type == pygame.KEYDOWN:
            promotion = accept & {"q", "r", "b", "n"}
            if promotion and event.key in PROMOTION_KEYS and PROMOTION_KEYS[event.key] in promotion:
                return PROMOTION_KEYS[event.key]
            if event.key == pygame.K_ESCAPE:
                if "quit" in accept:
                    return "quit"
                raise GameAborted
            if event.key == pygame.K_SPACE and "done" in accept:
                return "done"
            if event.key in (pygame.K_w, pygame.K_b) and {"white", "black"} <= accept:
                return "white" if event.key == pygame.K_w else "black"
            if event.key == pygame.K_r and "recalibrate" in accept:
                return "recalibrate"
            if event.key == pygame.K_g and "resign" in accept:
                return "resign"
            if event.key == pygame.K_d and "draw" in accept:
                return "draw"
            if event.key == pygame.K_f and "force" in accept:
                return "force"
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and "move" in accept:
            square = self._square_at(event.pos, state.human_color)
            if square is None:
                self.selected = None
            elif self.selected is None or self.selected == square:
                self.selected = square if self.selected is None else None
            else:
                move = chess.Move(self.selected, square)
                return f"move:{move.uci()}"
        return None

    def _square_at(self, pos: tuple[int, int], white_below: bool) -> int | None:
        col, row = (pos[0] - BOARD_X) // SQUARE, (pos[1] - BOARD_Y) // SQUARE
        if not (0 <= col < 8 and 0 <= row < 8):
            return None
        file, rank = (col, 7 - row) if white_below else (7 - col, row)
        return chess.square(file, rank)

    # ------------------------------------------------------------------ drawing
    def _draw(self, state: ViewState) -> None:
        now = time.monotonic()
        if len(state.board.move_stack) == self._plies + 1:
            self._slide = (state.board.peek(), now)
        self._plies = len(state.board.move_stack)
        if self._slide is not None and now - self._slide[1] >= SLIDE_S:
            self._slide = None

        self.screen.fill(FOG)
        pygame.draw.rect(self.screen, SLATE, (PANEL_X, 0, SIZE[0] - PANEL_X, SIZE[1]))
        self._draw_eval_bar(state)
        self._draw_board(state, now)
        self._draw_players(state)
        self._draw_keys()
        self._draw_willy(state, now)
        self._draw_patience(state)
        y = self._draw_thoughts(state, 250)
        y = self._draw_prompt(state, y + 14)
        self._draw_camera(state, PX, max(y + 16, 500))
        self._draw_moves(state, PX + 256, max(y + 16, 500))

    def _xy(self, square: int, white_below: bool) -> tuple[int, int]:
        file, rank = chess.square_file(square), chess.square_rank(square)
        col, row = (file, 7 - rank) if white_below else (7 - file, rank)
        return BOARD_X + col * SQUARE, BOARD_Y + row * SQUARE

    def _centre(self, square: int, white_below: bool) -> tuple[int, int]:
        x, y = self._xy(square, white_below)
        return x + SQUARE // 2, y + SQUARE // 2

    def _draw_eval_bar(self, state: ViewState) -> None:
        pawns = willy_pawns(state.evaluation, chess.WHITE)
        target = 0.5 if pawns is None else 1.0 / (1.0 + 10 ** (-pawns / 4.0))
        self._bar += (target - self._bar) * 0.25
        bar = pygame.Rect(26, BOARD_Y, 14, BOARD)
        pygame.draw.rect(self.screen, BLACK_PIECE, bar, border_radius=7)
        white = int(BOARD * self._bar)
        below = state.human_color == chess.WHITE
        part = pygame.Rect(bar.x, bar.bottom - white if below else bar.y, bar.width, white)
        pygame.draw.rect(self.screen, WHITE_PIECE, part, border_radius=7)
        pygame.draw.line(self.screen, SAFETY, (bar.x - 3, bar.centery), (bar.right + 2, bar.centery), 2)

    def _draw_board(self, state: ViewState, now: float) -> None:
        white_below = state.human_color == chess.WHITE
        pygame.draw.rect(self.screen, FOG_DEEP, (BOARD_X + 6, BOARD_Y + 8, BOARD, BOARD), border_radius=4)
        last = state.board.move_stack[-1] if state.board.move_stack else None
        marked = pygame.Surface((SQUARE, SQUARE), pygame.SRCALPHA)
        marked.fill((*SAFETY, 90) if last is not None and state.board.turn == state.human_color else (*SIGNAL, 90))
        sliding = self._slide[0] if self._slide is not None else None
        for sq in chess.SQUARES:
            x, y = self._xy(sq, white_below)
            dark = (chess.square_file(sq) + chess.square_rank(sq)) % 2 == 0
            pygame.draw.rect(self.screen, DARK if dark else LIGHT, (x, y, SQUARE, SQUARE))
            if last is not None and sq in (last.from_square, last.to_square):
                self.screen.blit(marked, (x, y))
            self._draw_coordinates(sq, x, y, dark, white_below)
        overlay = pygame.Surface((BOARD, BOARD), pygame.SRCALPHA)
        for move, colour, alpha, width in self._arrows(state):
            self._arrow(overlay, move, colour, alpha, width, white_below)
        for sq in chess.SQUARES:
            piece = state.board.piece_at(sq)
            if piece is None or (sliding is not None and sq == sliding.to_square):
                continue
            x, y = self._xy(sq, white_below)
            self._draw_piece(piece.symbol(), x, y)
            physical = state.physical.get(sq)
            if physical is not None and physical.lower() != piece.symbol().lower():
                badge = self.coords.render(f"={LETTERS[physical.lower()]}", True, STOP)
                self.screen.blit(badge, (x + SQUARE - badge.get_width() - 4, y + 3))
        self.screen.blit(overlay, (BOARD_X, BOARD_Y))
        if sliding is not None and state.board.piece_at(sliding.to_square) is not None:
            p = min(1.0, (now - self._slide[1]) / SLIDE_S)
            p = 1 - (1 - p) ** 3                                    # eased out: it lands softly
            (x0, y0), (x1, y1) = self._xy(sliding.from_square, white_below), self._xy(sliding.to_square, white_below)
            self._draw_piece(state.board.piece_at(sliding.to_square).symbol(), x0 + (x1 - x0) * p, y0 + (y1 - y0) * p)
        for sq in state.uncertain:
            x, y = self._xy(sq, white_below)
            pygame.draw.rect(self.screen, STOP, (x + 2, y + 2, SQUARE - 4, SQUARE - 4), 4, border_radius=4)
        if state.taking is not None:                                # the piece Willy takes off: crossed out
            x, y = self._xy(state.taking, white_below)
            pygame.draw.rect(self.screen, STOP, (x + 2, y + 2, SQUARE - 4, SQUARE - 4), 4, border_radius=4)
            for a, b in (((x + 16, y + 16), (x + SQUARE - 16, y + SQUARE - 16)),
                         ((x + SQUARE - 16, y + 16), (x + 16, y + SQUARE - 16))):
                pygame.draw.line(self.screen, STOP, a, b, 7)
        if self.selected is not None:
            x, y = self._xy(self.selected, white_below)
            pygame.draw.rect(self.screen, SIGNAL, (x + 2, y + 2, SQUARE - 4, SQUARE - 4), 4, border_radius=4)
        pygame.draw.rect(self.screen, INK, (BOARD_X - 2, BOARD_Y - 2, BOARD + 4, BOARD + 4), 2, border_radius=3)

    def _draw_coordinates(self, sq: int, x: int, y: int, dark: bool, white_below: bool) -> None:
        colour = LIGHT if dark else DARK
        if chess.square_rank(sq) == (0 if white_below else 7):
            label = self.coords.render(chess.FILE_NAMES[chess.square_file(sq)], True, colour)
            self.screen.blit(label, (x + SQUARE - label.get_width() - 4, y + SQUARE - label.get_height() - 1))
        if chess.square_file(sq) == (0 if white_below else 7):
            self.screen.blit(self.coords.render(chess.RANK_NAMES[chess.square_rank(sq)], True, colour), (x + 4, y + 2))

    def _arrows(self, state: ViewState) -> list[tuple[chess.Move, tuple[int, int, int], int, int]]:
        """Willy's move while the arm carries it out; while he thinks, the moves he weighs, the one in his eye bright."""
        if state.intent is not None:
            return [(state.intent, SAFETY, 230, 15)]
        if state.thought is None or state.board.turn == state.human_color:
            return []
        arrows = [(c.move, SAFETY, 75, 10) for c in state.thought.candidates if c.move != state.focus]
        if state.focus is not None:
            arrows.append((state.focus, SAFETY, 220, 15))
        return arrows

    def _arrow(self, surface: pygame.Surface, move: chess.Move, colour, alpha: int, width: int, white_below: bool) -> None:
        (x0, y0), (x1, y1) = self._centre(move.from_square, white_below), self._centre(move.to_square, white_below)
        x0, y0, x1, y1 = x0 - BOARD_X, y0 - BOARD_Y, x1 - BOARD_X, y1 - BOARD_Y
        length = math.hypot(x1 - x0, y1 - y0) or 1.0
        ux, uy = (x1 - x0) / length, (y1 - y0) / length
        nx, ny = -uy, ux
        head = width * 2.3
        bx, by = x1 - ux * head, y1 - uy * head
        w = width / 2
        points = [(x0 + nx * w, y0 + ny * w), (bx + nx * w, by + ny * w), (bx + nx * head * 0.75, by + ny * head * 0.75),
                  (x1, y1), (bx - nx * head * 0.75, by - ny * head * 0.75), (bx - nx * w, by - ny * w),
                  (x0 - nx * w, y0 - ny * w)]
        pygame.draw.polygon(surface, (*colour, alpha), points)

    def _piece_surface(self, symbol: str, size: int) -> pygame.Surface | None:
        """A piece as a picture, drawn once per size: a filled glyph under its outline, a shadow beneath."""
        key = (symbol, size)
        if key not in self._pieces:
            font = self.glyphs if size == 60 else self.mini
            if font is None:
                return None
            fill = font.render(GLYPHS[symbol.lower()], True, WHITE_PIECE if symbol.isupper() else BLACK_PIECE)
            outline = font.render(GLYPHS[symbol] if symbol.isupper() else GLYPHS[symbol.upper()], True, BLACK_PIECE)
            shadow = font.render(GLYPHS[symbol.lower()], True, (0, 0, 0))
            shadow.set_alpha(60)
            out = pygame.Surface((fill.get_width() + 3, fill.get_height() + 3), pygame.SRCALPHA)
            out.blit(shadow, (3, 3))
            out.blit(fill, (0, 0))
            out.blit(outline, (0, 0))
            self._pieces[key] = out
        return self._pieces[key]

    def _draw_piece(self, symbol: str, x: float, y: float) -> None:
        cx, cy = x + SQUARE // 2, y + SQUARE // 2
        picture = self._piece_surface(symbol, 60)
        if picture is not None:
            self.screen.blit(picture, picture.get_rect(center=(cx + 1, cy + 3)))
            return
        white = symbol.isupper()
        pygame.draw.circle(self.screen, WHITE_PIECE if white else BLACK_PIECE, (cx, cy), 28)
        pygame.draw.circle(self.screen, INK, (cx, cy), 28, 2)
        letter = self.letters.render(LETTERS[symbol.lower()], True, INK if white else WHITE_PIECE)
        self.screen.blit(letter, letter.get_rect(center=(cx, cy)))

    def _draw_players(self, state: ViewState) -> None:
        """Willy above the board, you below it: who is to move, the pieces each has taken, who is ahead."""
        willy = not state.human_color
        for colour, name, y, accent in ((willy, "Willy", 22, SAFETY), (state.human_color, "You", BOARD_Y + BOARD + 12, SIGNAL)):
            x = BOARD_X
            to_move = state.board.turn == colour and not state.board.is_game_over()
            pygame.draw.circle(self.screen, accent if to_move else FOG_DEEP, (x + 7, y + 13), 7)
            label = self.strong.render(name, True, INK)
            self.screen.blit(label, (x + 22, y))
            x += 22 + label.get_width() + 14
            pygame.draw.circle(self.screen, WHITE_PIECE if colour == chess.WHITE else BLACK_PIECE, (x, y + 13), 7)
            pygame.draw.circle(self.screen, INK, (x, y + 13), 7, 2)
            x += 18
            for kind, count in self._taken(state.board, colour):
                picture = self._piece_surface(chess.Piece(kind, not colour).symbol(), 24)
                for _ in range(count):
                    if picture is not None:
                        self.screen.blit(picture, (x, y - 1))
                    x += 15
            lead = self._material(state.board, colour) - self._material(state.board, not colour)
            if lead > 0:
                self.screen.blit(self.body.render(f"+{lead}", True, INK_MUTED), (x + 12, y + 3))

    @staticmethod
    def _taken(board: chess.Board, colour: bool) -> list[tuple[int, int]]:
        """(piece type, count) of the opponent's pieces `colour` has taken, the most valuable first."""
        return [(kind, max(0, START[kind] - len(board.pieces(kind, not colour))))
                for kind in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN)]

    @staticmethod
    def _material(board: chess.Board, colour: bool) -> int:
        return sum(VALUES[kind] * len(board.pieces(kind, colour)) for kind in VALUES)

    def _draw_keys(self) -> None:
        x, y = BOARD_X, BOARD_Y + BOARD + 48
        for key, action in (("Space", "done"), ("R", "calibrate"), ("G", "resign"), ("D", "draw"), ("Esc", "end"),
                            ("Click", "two squares to move")):
            text = self.small.render(key, True, INK)
            cap = pygame.Rect(x, y, text.get_width() + 12, 22)
            pygame.draw.rect(self.screen, FOG_DEEP, cap.move(0, 2), border_radius=5)
            pygame.draw.rect(self.screen, CHALK, cap, border_radius=5)
            pygame.draw.rect(self.screen, INK_MUTED, cap, 1, border_radius=5)
            self.screen.blit(text, (x + 6, y + 3))
            label = self.small.render(action, True, INK_MUTED)
            self.screen.blit(label, (cap.right + 5, y + 3))
            x = cap.right + label.get_width() + 14

    # ------------------------------------------------------------------ Willy's side
    def _draw_willy(self, state: ViewState, now: float) -> None:
        mood, speech = state.mood, state.speech
        waiting = self._asking and state.board.turn == state.human_color and mood in ("idle", "smug", "worried")
        if waiting and now - self._waiting_since > SLEEPY_S:
            mood, speech = "sleepy", "Zzz... oh, still your move?"
        white_below = state.human_color == chess.WHITE
        look = None
        if state.focus is not None:
            look = self._centre(state.focus.to_square, white_below)
        elif state.intent is not None:
            look = self._centre(state.intent.from_square, white_below)
        elif state.taking is not None:
            look = self._centre(state.taking, white_below)
        elif self.selected is not None:
            look = self._centre(self.selected, white_below)
        elif state.board.move_stack and state.board.turn != state.human_color:
            look = self._centre(state.board.peek().to_square, white_below)       # what you just did
        if mood == "dizzy":
            lamp = "red"
        elif (state.board.turn != state.human_color or mood in ("thinking", "focused") or state.intent is not None
              or state.taking is not None):
            lamp = "amber"
        else:
            lamp = "green"
        self.face.draw(self.screen, pygame.Rect(PX, 16, 236, 214), mood, look, lamp,
                       pygame.Rect(BOARD_X, BOARD_Y, BOARD, BOARD), now)
        if speech:
            self._bubble(speech, PX + 256, 46, PW - 256, (PX + 226, 112))

    def _draw_patience(self, state: ViewState) -> None:
        """How many more moves against the rules Willy lets pass before he takes the piece, once you made one."""
        if not state.strikes or not state.patience:
            return
        left = max(0, state.patience - state.strikes)
        x, y = PX + 256, 16
        label = self.small.render("Patience" if left else "No patience left", True, CHALK_MUTED if left else STOP)
        self.screen.blit(label, (x, y + 2))
        x += label.get_width() + 16
        for i in range(state.patience):
            if i < left:
                pygame.draw.circle(self.screen, SAFETY, (x + i * 20, y + 11), 7)
            else:
                pygame.draw.circle(self.screen, STOP, (x + i * 20, y + 11), 7, 2)

    def _bubble(self, text: str, x: int, y: int, width: int, tail: tuple[int, int]) -> None:
        lines = self._wrap(text, self.speech, width - 32)
        box = pygame.Rect(x, y, width, max(56, 26 * len(lines) + 26))
        pygame.draw.polygon(self.screen, CHALK, [tail, (x + 2, y + 16), (x + 2, y + 36)])
        pygame.draw.rect(self.screen, CHALK, box, border_radius=18)
        for i, line in enumerate(lines):
            self.screen.blit(self.speech.render(line, True, INK), (x + 16, y + 13 + 26 * i))

    def _draw_thoughts(self, state: ViewState, y: int) -> int:
        thought = state.thought
        willys_turn = state.board.turn != state.human_color
        if thought is not None and thought.candidates:
            heading = f"Thinking, depth {thought.depth}" if willys_turn and state.intent is None else "What Willy weighed"
        else:
            pawns = willy_pawns(state.evaluation, not state.human_color)
            heading = ("Willy is waiting to think" if pawns is None else
                       f"Willy rates his position {pawns:+.1f}" if abs(pawns) < 50 else
                       ("Willy sees a mate for him" if pawns > 0 else "Willy sees a mate against him"))
        self.screen.blit(self.body.render(heading, True, CHALK_MUTED), (PX, y))
        y += 30
        candidates = thought.candidates if thought is not None else ()
        for candidate in candidates[:3]:
            focused = candidate.move == state.focus or candidate.move == state.intent
            if focused:
                pygame.draw.rect(self.screen, SAFETY, (PX - 14, y + 2, 5, 26), border_radius=2)
            self.screen.blit(self.strong.render(candidate.san, True, CHALK if focused else CHALK_MUTED), (PX, y))
            track = pygame.Rect(PX + 92, y + 10, PW - 172, 9)
            pygame.draw.rect(self.screen, SLATE_HI, track, border_radius=5)
            share = 1.0 / (1.0 + 10 ** (-candidate.cp / 400.0))
            fill = track.copy()
            fill.width = max(6, int(track.width * share))
            pygame.draw.rect(self.screen, SAFETY if focused else CHALK_MUTED, fill, border_radius=5)
            score = self.body.render(candidate.label, True, CHALK if focused else CHALK_MUTED)
            self.screen.blit(score, (PX + PW - score.get_width(), y + 2))
            y += 34
        return y + max(0, 3 - len(candidates[:3])) * 34

    def _draw_prompt(self, state: ViewState, y: int) -> int:
        """What the game needs from you, on a band that turns your colour while it waits for you."""
        lines = self._wrap(state.status, self.body, PW - 30)[:5]
        band = pygame.Rect(PX - 14, y, PW + 28, 24 * len(lines) + 22)
        pygame.draw.rect(self.screen, SLATE_HI, band, border_radius=10)
        pygame.draw.rect(self.screen, SIGNAL if self._asking else CHALK_MUTED, (band.x, band.y, 6, band.height),
                         border_top_left_radius=10, border_bottom_left_radius=10)
        for i, line in enumerate(lines):
            self.screen.blit(self.body.render(line, True, CHALK), (PX + 6, y + 11 + 24 * i))
        return band.bottom

    def _draw_camera(self, state: ViewState, x: int, y: int) -> None:
        """The last view with the piece masks: white pieces blue, black pieces orange."""
        obs = state.observation
        if obs is None:
            return
        if self._camera is None or self._camera[0] != id(obs):
            image = obs.frame.bgr.copy()
            if obs.frame.labels is not None:
                tint = np.zeros_like(image)
                for k, symbol in obs.frame.classes.items():
                    tint[obs.frame.labels == k] = (255, 160, 60) if symbol.isupper() else (40, 140, 255)
                mask = obs.frame.labels > 0
                image[mask] = (0.55 * image[mask] + 0.45 * tint[mask]).astype(np.uint8)
            w = 236
            h = int(image.shape[0] * w / image.shape[1])
            rgb = cv.cvtColor(cv.resize(image, (w, h), interpolation=cv.INTER_AREA), cv.COLOR_BGR2RGB)
            self._camera = (id(obs), pygame.image.frombuffer(rgb.tobytes(), (w, h), "RGB").copy())
        picture = self._camera[1]
        self.screen.blit(picture, (x, y))
        pygame.draw.rect(self.screen, SLATE_HI, (x, y, picture.get_width(), picture.get_height()), 2)
        caption = {"corrected": "Board confirmed", "predicted": "Board pose predicted",
                   "moved": "The board moved"}[obs.pose_status]
        if obs.hands:
            caption += f", {obs.hands} hand{'s' if obs.hands > 1 else ''} seen"
        self.screen.blit(self.small.render(caption, True, CHALK_MUTED), (x, y + picture.get_height() + 6))

    def _draw_moves(self, state: ViewState, x: int, y: int) -> None:
        """The moves so far, the latest bright; a piece Willy took off the board stands between them."""
        rows: list[tuple[str, list[str] | None]] = []         # ("12.", [white, black]) or (note, None)
        for game, taken in [(t.before, t) for t in state.earlier] + [(state.board, None)]:
            board = game.root()
            first = board.fullmove_number
            sans = [] if board.turn == chess.WHITE else ["..."]      # a part that starts with Black to move
            for move in game.move_stack:
                sans.append(board.san(move))
                board.push(move)
            rows += [(f"{first + i // 2}.", sans[i:i + 2]) for i in range(0, len(sans), 2)]
            if taken is not None:
                rows.append((f"{chess.piece_name(taken.piece.piece_type)} {chess.square_name(taken.square)} "
                             f"taken off", None))
        room = max(0, (SIZE[1] - 24 - y) // 24)
        shown = rows[max(0, len(rows) - room):] if room else []
        for r, (head, pair) in enumerate(shown):
            if pair is None:
                self.screen.blit(self.body.render(head, True, STOP), (x, y))
            else:
                self.screen.blit(self.body.render(head, True, CHALK_MUTED), (x, y))
                for c, san in enumerate(pair):
                    latest = r == len(shown) - 1 and c == len(pair) - 1
                    self.screen.blit(self.body.render(san, True, CHALK if latest else CHALK_MUTED), (x + 44 + c * 82, y))
            y += 24

    def _wrap(self, text: str, font: pygame.font.Font, width: int) -> list[str]:
        lines, line = [], ""
        for word in text.split():
            trial = f"{line} {word}".strip()
            if font.size(trial)[0] > width and line:
                lines.append(line)
                line = word
            else:
                line = trial
        return lines + [line] if line else lines


def headless() -> None:
    """Render without a screen (tests, CI)."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
