"""
Willy's face: a robot head with big eyes that look where he is thinking, and a stack light on top.

Drawn with pygame primitives, no images. The moods are src/game/persona.py's; everything that moves (blinking, the
gaze, the light while he thinks, the head bobbing) runs on the clock, so the face lives as long as the window is
drawn. The light is a cell's stack light: green, the human's turn; amber, Willy thinks or moves; red, a stop.
"""
import math
import random

import pygame

INK = (30, 42, 51)
HEAD = (199, 211, 219)
HEAD_SHADE = (170, 186, 197)
VISOR = (34, 48, 58)
SCLERA = (247, 249, 250)
BLUSH = (242, 154, 142)
SWEAT = (126, 196, 232)
TONGUE = (232, 112, 106)
ANGER = (224, 72, 62)
STEAM = (236, 241, 244)
LAMPS = {"red": (224, 72, 62), "amber": (244, 181, 38), "green": (43, 179, 163)}
LAMP_OFF = (74, 88, 98)

# mood -> (left brow outer, inner, right brow inner, outer) in px, negative is up
BROWS = {"idle": (0, 0, 0, 0), "happy": (-3, -3, -3, -3), "smug": (0, 2, -6, -9), "thinking": (-8, -6, 2, 3),
         "focused": (-2, 5, 5, -2), "surprised": (-10, -10, -10, -10), "worried": (3, -7, -7, 3),
         "sad": (4, -6, -6, 4), "embarrassed": (2, -5, -5, 2), "dizzy": (-4, 2, -4, 2), "sleepy": (3, 3, 3, 3),
         "annoyed": (5, 5, 5, 5), "angry": (-7, 10, 10, -7)}
# mood -> how far the upper and lower lids close, 0 open .. 1 shut
LIDS = {"smug": (0.38, 0.0), "focused": (0.28, 0.18), "sad": (0.3, 0.0), "sleepy": (0.62, 0.05),
        "embarrassed": (0.12, 0.0), "worried": (0.05, 0.0), "annoyed": (0.48, 0.12), "angry": (0.18, 0.1)}


class Face:
    """One Willy face, remembering where it looks and when it blinks next."""

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self.gaze = [0.0, 0.0]                  # pupil offset, -1..1 each way
        self.next_blink = 0.0
        self.blink_until = 0.0
        self.wander: tuple[float, float] | None = None
        self.next_wander = 0.0
        self.last = None
        self.zfont: pygame.font.Font | None = None

    def draw(self, surface: pygame.Surface, rect: pygame.Rect, mood: str, look_at: tuple[float, float] | None,
             lamp: str, wander_area: pygame.Rect, now: float) -> None:
        """
        Draw the face into rect.

        Args:
            mood (str): One of the persona's moods.
            look_at (tuple | None): A point on the window the eyes look at; None lets them wander over wander_area.
            lamp (str): "green", "amber" or "red".
            now (float): Seconds, a steady clock.
        """
        dt = 0.0 if self.last is None else min(now - self.last, 0.2)
        self.last = now
        bob = 2.0 * math.sin(now * 1.7)
        shake = 2.0 * math.sin(now * 47.0) if mood == "angry" else 0.0          # trembling with rage
        head = pygame.Rect(rect.x + 8 + shake, rect.y + 48 + bob, rect.width - 16, rect.height - 56)
        self._stack_light(surface, head.centerx, rect.y + 2 + bob, head.top, lamp, mood, now)

        pygame.draw.rect(surface, HEAD_SHADE, head.move(0, 5), border_radius=36)
        pygame.draw.rect(surface, HEAD, head, border_radius=36)
        pygame.draw.rect(surface, INK, head, 4, border_radius=36)
        visor = pygame.Rect(head.x + 20, head.y + 30, head.width - 40, 74)
        pygame.draw.rect(surface, VISOR, visor, border_radius=28)

        eyes = [(visor.centerx - visor.width // 4 - 2, visor.centery), (visor.centerx + visor.width // 4 + 2, visor.centery)]
        self._look(eyes, look_at, wander_area, now, dt)
        if mood == "happy":
            for x, y in eyes:
                pygame.draw.arc(surface, SCLERA, (x - 24, y - 12, 48, 40), math.radians(20), math.radians(160), 7)
        elif mood == "dizzy":
            for i, (x, y) in enumerate(eyes):
                self._spiral(surface, x, y, now * (5 if i else -5))
        else:
            for x, y in eyes:
                self._eye(surface, x, y, mood, visor, now)
        self._brows(surface, eyes, head.y + 18, mood)
        self._mouth(surface, head.centerx, head.y + 126, mood, now)
        self._extras(surface, head, visor, eyes, mood, now)

    # ------------------------------------------------------------------ parts
    def _look(self, eyes, look_at, area: pygame.Rect, now: float, dt: float) -> None:
        if look_at is None:
            if self.wander is None or now >= self.next_wander:
                # a square of the board, or now and then the person in front of the screen
                if self.rng.random() < 0.25:
                    self.wander = ((eyes[0][0] + eyes[1][0]) / 2, eyes[0][1] + 600)
                else:
                    self.wander = (area.x + self.rng.random() * area.width, area.y + self.rng.random() * area.height)
                self.next_wander = now + self.rng.uniform(1.2, 3.5)
            look_at = self.wander
        cx, cy = (eyes[0][0] + eyes[1][0]) / 2, eyes[0][1]
        dx, dy = look_at[0] - cx, look_at[1] - cy
        distance = math.hypot(dx, dy) or 1.0
        reach = min(distance / 220.0, 1.0)
        target = (dx / distance * reach, dy / distance * reach)
        k = min(1.0, dt * 10.0)
        self.gaze = [g + (t - g) * k for g, t in zip(self.gaze, target)]

    def _eye(self, surface, x: int, y: int, mood: str, visor: pygame.Rect, now: float) -> None:
        surprised = mood == "surprised"
        w, h = (64, 66) if surprised else (56, 60)
        pygame.draw.ellipse(surface, SCLERA, (x - w // 2, y - h // 2, w, h))
        radius = 9 if surprised else 15
        px = x + self.gaze[0] * (w / 2 - radius - 4)
        py = y + self.gaze[1] * (h / 2 - radius - 4)
        pygame.draw.circle(surface, INK, (px, py), radius)
        pygame.draw.circle(surface, SCLERA, (px - radius * 0.35, py - radius * 0.4), max(2, radius // 3))

        upper, lower = LIDS.get(mood, (0.0, 0.0))
        if mood == "angry":                 # lids slanting down to the nose
            top = y - h // 2 - 3
            low, high = top + h * 0.5, top + h * 0.12
            outer_y, inner_y = (high, low) if x < visor.centerx else (low, high)       # (left edge, right edge)
            pygame.draw.polygon(surface, VISOR, [(x - w // 2 - 3, top), (x + w // 2 + 3, top),
                                                 (x + w // 2 + 3, inner_y), (x - w // 2 - 3, outer_y)])
            pygame.draw.line(surface, INK, (x - w // 2, outer_y), (x + w // 2, inner_y), 3)
        if now >= self.next_blink:
            self.blink_until, self.next_blink = now + 0.14, now + self.rng.uniform(2.5, 6.0)
        if now < self.blink_until:
            upper = 1.0 - abs((self.blink_until - now) / 0.07 - 1.0) * (1.0 - upper)     # down and up again
        top, bottom = y - h // 2 - 2, y + h // 2 + 2
        if upper > 0:
            pygame.draw.rect(surface, VISOR, (x - w // 2 - 2, top, w + 4, (h + 4) * upper))
            pygame.draw.line(surface, INK, (x - w // 2, top + (h + 4) * upper), (x + w // 2, top + (h + 4) * upper), 3)
        if lower > 0:
            pygame.draw.rect(surface, VISOR, (x - w // 2 - 2, bottom - (h + 4) * lower, w + 4, (h + 4) * lower + 1))

    def _spiral(self, surface, x: int, y: int, turn: float) -> None:
        pygame.draw.ellipse(surface, SCLERA, (x - 28, y - 30, 56, 60))
        points = [(x + r * math.cos(a + turn), y + r * math.sin(a + turn))
                  for a, r in ((i * 0.35, 1.0 + i * 0.38) for i in range(60))]
        pygame.draw.lines(surface, INK, False, points, 3)

    def _brows(self, surface, eyes, y: int, mood: str) -> None:
        lo, li, ri, ro = BROWS.get(mood, BROWS["idle"])
        (lx, _), (rx, _) = eyes
        for x0, y0, x1, y1 in ((lx - 26, y + lo, lx + 18, y + li), (rx - 18, y + ri, rx + 26, y + ro)):
            pygame.draw.line(surface, INK, (x0, y0), (x1, y1), 7)
            for end in ((x0, y0), (x1, y1)):
                pygame.draw.circle(surface, INK, end, 3)

    def _mouth(self, surface, x: int, y: int, mood: str, now: float) -> None:
        if mood == "happy":
            pygame.draw.ellipse(surface, INK, (x - 28, y - 16, 56, 34))
            pygame.draw.rect(surface, HEAD, (x - 30, y - 18, 60, 17))
            pygame.draw.ellipse(surface, TONGUE, (x - 10, y + 4, 20, 10))
        elif mood == "smug":
            pygame.draw.lines(surface, INK, False, [(x - 22, y + 2), (x - 6, y + 5), (x + 10, y + 2), (x + 24, y - 8)], 5)
        elif mood == "surprised":
            pygame.draw.ellipse(surface, INK, (x - 10, y - 12, 20, 24))
        elif mood == "thinking":
            wobble = 3 * math.sin(now * 3.0)
            pygame.draw.line(surface, INK, (x - 4, y + 2), (x + 20, y - 3 + wobble), 5)
        elif mood == "focused":
            pygame.draw.line(surface, INK, (x - 18, y), (x + 18, y), 5)
            pygame.draw.ellipse(surface, TONGUE, (x + 10, y - 1, 12, 12))         # the tongue of concentration
        elif mood in ("worried", "embarrassed", "dizzy"):
            points = [(x - 22 + i * 4, y + 3 * math.sin(i * 1.4 + now * 4)) for i in range(12)]
            pygame.draw.lines(surface, INK, False, points, 4)
        elif mood == "sad":
            pygame.draw.arc(surface, INK, (x - 20, y - 2, 40, 22), math.radians(20), math.radians(160), 5)
        elif mood == "sleepy":
            pygame.draw.ellipse(surface, INK, (x - 6, y - 4, 12, 10), 3)
        elif mood == "annoyed":
            pygame.draw.line(surface, INK, (x - 8, y + 1), (x + 22, y - 3), 5)               # pushed to one side
        elif mood == "angry":                                                               # gritted teeth
            pygame.draw.rect(surface, INK, (x - 27, y - 11, 54, 24), border_radius=8)
            pygame.draw.rect(surface, SCLERA, (x - 22, y - 6, 44, 14), border_radius=3)
            pygame.draw.line(surface, INK, (x - 22, y + 1), (x + 21, y + 1), 2)
            for tooth in (-11, 0, 11):
                pygame.draw.line(surface, INK, (x + tooth, y - 6), (x + tooth, y + 7), 2)
        else:
            pygame.draw.arc(surface, INK, (x - 18, y - 16, 36, 22), math.radians(200), math.radians(340), 5)

    def _extras(self, surface, head: pygame.Rect, visor: pygame.Rect, eyes, mood: str, now: float) -> None:
        if mood in ("happy", "embarrassed", "smug"):
            blush = pygame.Surface((30, 16), pygame.SRCALPHA)
            pygame.draw.ellipse(blush, (*BLUSH, 150), blush.get_rect())
            for x, _ in eyes:
                surface.blit(blush, (x - 15 + (-18 if x < head.centerx else 18), visor.bottom + 6))
        if mood in ("worried", "embarrassed", "dizzy"):
            x, y = head.right - 16, head.y + 6 + 3 * math.sin(now * 2)
            pygame.draw.polygon(surface, SWEAT, [(x, y), (x - 7, y + 14), (x + 7, y + 14)])
            pygame.draw.circle(surface, SWEAT, (x, y + 16), 7)
        if mood == "sad":
            x, y = eyes[0][0] - 10, visor.bottom - 6 + (now * 30) % 26
            pygame.draw.circle(surface, SWEAT, (x, y), 5)
        if mood == "thinking":
            for i in range(3):
                phase = (now * 2.2 - i * 0.35) % 2.0
                radius = 4 + i * 2 + (2 if phase < 1.0 else 0)
                pygame.draw.circle(surface, SCLERA, (head.right + 6 + i * 13, head.y - 6 - i * 12), radius)
        if mood in ("annoyed", "angry"):
            self._steam(surface, head, now, both=mood == "angry")
        if mood == "angry":
            flush = pygame.Surface((38, 18), pygame.SRCALPHA)
            pygame.draw.ellipse(flush, (*ANGER, 120), flush.get_rect())
            for x, _ in eyes:
                surface.blit(flush, (x - 19 + (-16 if x < head.centerx else 16), visor.bottom + 5))
            self._anger_mark(surface, head.right - 34, head.y + 18, now)
        if mood == "sleepy":
            if self.zfont is None:
                self.zfont = pygame.font.SysFont(None, 26, bold=True)
            for i in range(3):
                phase = (now * 0.6 + i / 3) % 1.0
                z = self.zfont.render("z", True, SCLERA)
                z.set_alpha(int(255 * (1 - phase)))
                surface.blit(z, (head.right - 6 + phase * 24 + i * 4, head.y - 4 - phase * 40))

    def _steam(self, surface, head: pygame.Rect, now: float, both: bool) -> None:
        """Puffs rising from the side of the head away from the speech bubble; both sides, faster, when really angry."""
        puffs = pygame.Surface((head.width + 120, 120), pygame.SRCALPHA)
        origin = (head.x - 60, head.y - 60)
        speed = 1.6 if both else 0.9
        for side in ((-1, 1) if both else (-1,)):
            for i in range(3):
                phase = (now * speed + i / 3) % 1.0
                x = (head.centerx - origin[0]) + side * (head.width / 2 + 4 + phase * 22)
                y = (head.y + 46 - origin[1]) - phase * 70
                pygame.draw.circle(puffs, (*STEAM, int(200 * (1 - phase))), (x, y), 5 + phase * 9)
        surface.blit(puffs, origin)

    def _anger_mark(self, surface, cx: float, cy: float, now: float) -> None:
        """The cross-popping vein of a cartoon temper, pulsing."""
        size = 7 + 1.5 * math.sin(now * 9)
        for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            centre = (cx + dx * size * 1.2, cy + dy * size * 1.2)
            facing = math.atan2(dy, -dx)                # towards the middle, in pygame's upward angles
            box = pygame.Rect(0, 0, 2 * size, 2 * size)
            box.center = centre
            pygame.draw.arc(surface, ANGER, box, facing - 0.9, facing + 0.9, 4)

    def _stack_light(self, surface, x: int, top: float, head_top: float, lamp: str, mood: str, now: float) -> None:
        pygame.draw.rect(surface, INK, (x - 3, top + 34, 6, head_top - top - 34))
        for i, name in enumerate(("red", "amber", "green")):
            lit = name == lamp
            if lit and (name == "red" or mood == "thinking"):
                lit = math.sin(now * (9 if name == "red" else 6)) > -0.3              # a stop flashes, thought pulses
            colour = LAMPS[name] if lit else LAMP_OFF
            box = pygame.Rect(x - 13, top + i * 11, 26, 11)
            if lit:
                glow = pygame.Surface((44, 26), pygame.SRCALPHA)
                pygame.draw.ellipse(glow, (*LAMPS[name], 45), glow.get_rect())
                surface.blit(glow, (x - 22, top + i * 11 - 8))
            pygame.draw.rect(surface, colour, box, border_radius=3)
            pygame.draw.rect(surface, INK, box, 2, border_radius=3)
