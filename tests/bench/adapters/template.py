"""Vorlage für deinen Adapter. Kopieren, z. B. nach bench/adapters/mine.py, und die Funktionen
an deinen Code anbinden. Was fehlt oder NotImplementedError wirft, wird übersprungen.

Konventionen (siehe bench/sim.py):
- Brettrahmen: Ursprung Brettmitte, x zur h-Linie, y zur 8. Reihe, z nach oben, Meter.
- Pose: X_cam = R @ X_brett + t. Felder: python-chess (a1 = 0, h8 = 63).
- Frame.depth: z-Tiefe in Metern (0 = ungültig). Frame.labels: 0 = keine Figur, k = Figur k.
- Frame.classes: k -> FEN-Symbol ('P' weißer Bauer, 'n' schwarzer Springer, ...).
"""
from __future__ import annotations

import numpy as np

from bench.sim import Calibration, Frame

NAME = "mine"


def corners(gray: np.ndarray) -> np.ndarray:
    """Suite ecken: innere Brettecken als (N, 2)-Array in Pixeln."""
    raise NotImplementedError


def board_homography(bgr: np.ndarray) -> np.ndarray | None:
    """Suite brett: 3x3-Homographie Bild -> Gitter. Die 49 inneren Ecken liegen auf ganzzahligen
    Gitterpunkten (i, j) mit -3 <= i, j <= 3; Drehung und Spiegelung sind hier egal. None = nicht gefunden."""
    raise NotImplementedError


def calibrate(frame: Frame) -> Calibration | None:
    """Suite einmessen: Grundstellung, Feldgröße unbekannt (45-65 mm). Volle Pose mit Orientierung
    (a1 links unten aus Sicht von Weiß), Feldgröße in Metern, optional Figurenmaße."""
    raise NotImplementedError


def assign(frame: Frame, calib: Calibration) -> dict[int, int]:
    """Suiten zuordnung, stress, einmessen: Label k -> Feld 0..63 für jede Figur in frame.classes."""
    raise NotImplementedError


class Tracker:
    """Suite tracking: Partie aus der Grundstellung, ein Zug pro Bild."""

    def __init__(self, calib: Calibration):
        raise NotImplementedError

    def reset(self, fen: str) -> None:
        """Zustand auf diese Stellung setzen (Start und nach markiertem oder falschem Zug)."""

    def step(self, frame: Frame) -> str | None:
        """Vollständige FEN nach dem Zug, oder None, wenn unsicher (zählt als markiert, nicht als Fehler)."""
        return None
