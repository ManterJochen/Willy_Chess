"""
The game on screen: the digital board, what the camera sees, and what Willy is doing.

Keys: Space = move done, R = calibrate again, G = resign, D = offer a draw, Esc = end,
W/B = colour, Q/R/B/N = promotion.
Clicking two squares enters a move by hand, for when the camera could not recognise it.
"""
import os

import chess
import cv2 as cv
import numpy as np
import pygame

from src.game.game import GameAborted, ViewState

SQUARE = 80
MARGIN = 40
PANEL_X = 2 * MARGIN + 8 * SQUARE
SIZE = (PANEL_X + 460, 2 * MARGIN + 8 * SQUARE)

LIGHT, DARK = (238, 216, 181), (181, 136, 99)
BACKGROUND, TEXT, MUTED = (30, 32, 36), (235, 235, 235), (150, 150, 160)
LAST_MOVE, UNCERTAIN, SELECTED = (246, 246, 105, 110), (220, 60, 60), (70, 140, 255)
GLYPHS = {"K": "♔", "Q": "♕", "R": "♖", "B": "♗", "N": "♘", "P": "♙",
          "k": "♚", "q": "♛", "r": "♜", "b": "♝", "n": "♞", "p": "♟"}
LETTERS = {"k": "K", "q": "Q", "r": "R", "b": "B", "n": "N", "p": "P"}     # without a font with chess glyphs
SYMBOL_FONTS = ("segoeuisymbol", "applesymbols", "arialunicodems", "dejavusans", "notosanssymbols2", "freeserif")
PROMOTION_KEYS = {pygame.K_q: "q", pygame.K_r: "r", pygame.K_b: "b", pygame.K_n: "n"}


class PygameView:
    """Display and input of ChessGame. Use as a context manager so the window always closes."""

    def __init__(self, title: str = "Willy_Chess"):
        pygame.init()
        self.screen = pygame.display.set_mode(SIZE)
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial", 18)
        self.small = pygame.font.SysFont("arial", 14)
        self.big = pygame.font.SysFont("arial", 26, bold=True)
        symbol_font = next((f for f in SYMBOL_FONTS if pygame.font.match_font(f)), None)
        self.glyphs = pygame.font.SysFont(symbol_font, 64) if symbol_font else None
        self.letters = pygame.font.SysFont("arial", 34, bold=True)
        self.selected: int | None = None

    def __enter__(self) -> "PygameView":
        return self

    def __exit__(self, *exc) -> None:
        pygame.quit()

    # ------------------------------------------------------------------ View protocol
    def show(self, state: ViewState) -> None:
        pygame.event.pump()             # keeps the window responsive while the robot moves
        self._draw(state)
        pygame.display.flip()

    def wait(self, state: ViewState, accept: set[str]) -> str:
        self.selected = None
        while True:
            for event in pygame.event.get():
                answer = self._answer(event, state, accept)
                if answer is not None:
                    self.selected = None
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
        col, row = (pos[0] - MARGIN) // SQUARE, (pos[1] - MARGIN) // SQUARE
        if not (0 <= col < 8 and 0 <= row < 8):
            return None
        file, rank = (col, 7 - row) if white_below else (7 - col, row)
        return chess.square(file, rank)

    # ------------------------------------------------------------------ drawing
    def _draw(self, state: ViewState) -> None:
        self.screen.fill(BACKGROUND)
        self._draw_board(state)
        self._draw_panel(state)

    def _xy(self, square: int, white_below: bool) -> tuple[int, int]:
        file, rank = chess.square_file(square), chess.square_rank(square)
        col, row = (file, 7 - rank) if white_below else (7 - file, rank)
        return MARGIN + col * SQUARE, MARGIN + row * SQUARE

    def _draw_board(self, state: ViewState) -> None:
        white_below = state.human_color == chess.WHITE
        overlay = pygame.Surface((SQUARE, SQUARE), pygame.SRCALPHA)
        overlay.fill(LAST_MOVE)
        last = state.board.move_stack[-1] if state.board.move_stack else None
        for sq in chess.SQUARES:
            x, y = self._xy(sq, white_below)
            dark = (chess.square_file(sq) + chess.square_rank(sq)) % 2 == 0
            pygame.draw.rect(self.screen, DARK if dark else LIGHT, (x, y, SQUARE, SQUARE))
            if last is not None and sq in (last.from_square, last.to_square):
                self.screen.blit(overlay, (x, y))
            piece = state.board.piece_at(sq)
            if piece is not None:
                self._draw_piece(piece.symbol(), x, y)
                physical = state.physical.get(sq)
                if physical is not None and physical.lower() != piece.symbol().lower():
                    badge = self.small.render(f"={LETTERS[piece.symbol().lower()]}", True, (200, 30, 30))
                    self.screen.blit(badge, (x + SQUARE - badge.get_width() - 4, y + 3))
            if sq in state.uncertain:
                pygame.draw.rect(self.screen, UNCERTAIN, (x + 2, y + 2, SQUARE - 4, SQUARE - 4), 4)
            if sq == self.selected:
                pygame.draw.rect(self.screen, SELECTED, (x + 2, y + 2, SQUARE - 4, SQUARE - 4), 4)
        for i in range(8):
            file_label = self.small.render("abcdefgh"[i if white_below else 7 - i], True, MUTED)
            rank_label = self.small.render(str(8 - i if white_below else i + 1), True, MUTED)
            self.screen.blit(file_label, (MARGIN + i * SQUARE + SQUARE // 2 - 4, MARGIN + 8 * SQUARE + 8))
            self.screen.blit(rank_label, (MARGIN - 18, MARGIN + i * SQUARE + SQUARE // 2 - 8))

    def _draw_piece(self, symbol: str, x: int, y: int) -> None:
        cx, cy = x + SQUARE // 2, y + SQUARE // 2
        if self.glyphs is not None:
            glyph = self.glyphs.render(GLYPHS[symbol], True, (20, 20, 20))
            self.screen.blit(glyph, glyph.get_rect(center=(cx, cy + 2)))
            return
        white = symbol.isupper()
        pygame.draw.circle(self.screen, (245, 245, 245) if white else (35, 35, 35), (cx, cy), 30)
        pygame.draw.circle(self.screen, (20, 20, 20), (cx, cy), 30, 2)
        letter = self.letters.render(LETTERS[symbol.lower()], True, (20, 20, 20) if white else (240, 240, 240))
        self.screen.blit(letter, letter.get_rect(center=(cx, cy)))

    def _draw_panel(self, state: ViewState) -> None:
        x, y = PANEL_X, MARGIN
        self.screen.blit(self.big.render("Willy_Chess", True, TEXT), (x, y))
        turn = "Your move" if state.board.turn == state.human_color else "Willy's move"
        if state.board.is_game_over():
            turn = f"Game over: {state.board.result()}"
        self.screen.blit(self.font.render(turn, True, MUTED), (x, y + 36))
        if state.evaluation:
            self.screen.blit(self.font.render(f"Evaluation (White): {state.evaluation}", True, MUTED), (x, y + 60))

        y = self._draw_text(state.status, x, y + 96, 420, TEXT)
        y = self._draw_camera(state, x, y + 12)
        y = self._draw_moves(state, x, y + 12)
        for i, help_text in enumerate(("Space: move done    R: calibrate again    G: resign    D: offer a draw",
                                       "Click two squares: enter a move by hand    Esc: end")):
            self.screen.blit(self.small.render(help_text, True, MUTED), (x, SIZE[1] - MARGIN - 8 + 18 * i))

    def _draw_text(self, text: str, x: int, y: int, width: int, colour) -> int:
        line = ""
        for word in text.split():
            trial = f"{line} {word}".strip()
            if self.font.size(trial)[0] > width and line:
                self.screen.blit(self.font.render(line, True, colour), (x, y))
                y, line = y + 22, word
            else:
                line = trial
        if line:
            self.screen.blit(self.font.render(line, True, colour), (x, y))
            y += 22
        return y

    def _draw_camera(self, state: ViewState, x: int, y: int) -> int:
        """The last view with the piece masks: white pieces blue, black pieces orange."""
        obs = state.observation
        if obs is None:
            return y
        image = obs.frame.bgr.copy()
        if obs.frame.labels is not None:
            tint = np.zeros_like(image)
            for k, symbol in obs.frame.classes.items():
                tint[obs.frame.labels == k] = (255, 160, 60) if symbol.isupper() else (40, 140, 255)
            mask = obs.frame.labels > 0
            image[mask] = (0.55 * image[mask] + 0.45 * tint[mask]).astype(np.uint8)
        w = 420
        h = int(image.shape[0] * w / image.shape[1])
        rgb = cv.cvtColor(cv.resize(image, (w, h), interpolation=cv.INTER_AREA), cv.COLOR_BGR2RGB)
        self.screen.blit(pygame.image.frombuffer(rgb.tobytes(), (w, h), "RGB"), (x, y))
        caption = {"corrected": "board pose confirmed", "predicted": "board pose predicted",
                   "moved": "board moved!"}[obs.pose_status] + "   (blue: White, orange: Black)"
        self.screen.blit(self.small.render(caption, True, MUTED), (x, y + h + 4))
        return y + h + 22

    def _draw_moves(self, state: ViewState, x: int, y: int) -> int:
        board = state.board.root()
        first = board.fullmove_number
        sans = [] if board.turn == chess.WHITE else ["..."]      # a game that started with Black to move
        for move in state.board.move_stack:
            sans.append(board.san(move))
            board.push(move)
        pairs = [f"{first + i // 2}. {' '.join(sans[i:i + 2])}" for i in range(0, len(sans), 2)]
        for line in pairs[-6:]:
            self.screen.blit(self.font.render(line, True, MUTED), (x, y))
            y += 22
        return y


def headless() -> None:
    """Render without a screen (tests, CI)."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
