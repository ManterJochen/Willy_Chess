# Willy_Chess-Benchmark: vom Brett zur FEN

Simulierte Szenen mit exakt bekannter Wahrheit, feste Seeds, eine eingefrorene Referenz-Implementierung
und eine Baseline. Deine Implementierung läuft über einen Adapter gegen dieselben Szenen; das Skript sagt,
wo sie signifikant schlechter ist, und endet dann mit Exit-Code 1.

## Installation

Den Ordner `bench/` in die Wurzel des Repos legen (neben `src/`), dann:

```bash
pip install -r bench/requirements.txt
python -m bench.run --quick            # Referenz, alle Suiten, etwa 3 Minuten
```

Alle Befehle laufen aus der Repo-Wurzel, unter Windows genauso.

## Deine Implementierung testen

1. `bench/adapters/template.py` nach `bench/adapters/mine.py` kopieren und die Funktionen an deinen Code
   anbinden. Was fehlt oder `NotImplementedError` wirft, wird übersprungen.
2. Einmal die Referenz auf deinem Rechner laufen lassen (die Szenen hängen leicht von NumPy- und
   OpenCV-Version ab, so vergleichst du garantiert auf identischen Eingaben):
   ```bash
   python -m bench.run --write-baseline bench/baseline_local.json
   ```
3. Deinen Adapter dagegen laufen lassen:
   ```bash
   python -m bench.run --adapter bench/adapters/mine.py --baseline bench/baseline_local.json
   ```

Steht hinter einer Suite `[Eingaben identisch]`, liefen exakt dieselben Bilder wie in der Baseline.
`--quick` nimmt ein Fünftel der Szenen mit denselben Seeds, für schnelle Runden beim Entwickeln.
`--dump-failures DIR` legt jede fehlgeschlagene Szene als PNG und NPZ ab (Bild, Tiefe, Labels, Wahrheit).

## Die Suiten

| Suite | Prüft | Szenen (voll) | Adapter-Funktion |
| --- | --- | --- | --- |
| `ecken` | Subpixel-Genauigkeit der inneren Ecken, leeres Brett 0-45° | 40 | `corners(gray)` |
| `brett` | Brett aus dem Bild, mit flachen Figuren oder Störmustern, 30-60° | 4 x 30 | `board_homography(bgr)` |
| `einmessen` | Grundstellung, Feldgröße 45-65 mm unbekannt, Figurensatz ±15 %: Pose, Orientierung, Skala | 4 x 30 | `calibrate(frame)` + `assign` |
| `zuordnung` | Figur aufs Feld bei bekannter Pose, zufällige Partiestellungen, 0-70° | 7 x 60 | `assign(frame, calib)` |
| `tracking` | Partien aus der Grundstellung, ein Zug pro Bild, 50/60/70° | 3 x 8 Partien x 50 Züge | `Tracker` |
| `stress` | 70°, je eine verletzte Annahme: Maskenrand, Figurensatz, Tiefenversatz, Wölbung | 8 x 20 | `assign(frame, calib)` |

Volle Laufzeit der Referenz etwa 13 Minuten, davon 7 für `tracking`.

## Baseline (Referenz, voller Lauf)

| Suite | Ergebnis |
| --- | --- |
| `ecken` | 1958/1960 Ecken gefunden, RMSE 0,042 px (Original: 0,349 px) |
| `brett` | 120/120 Bretter richtig (Original: 12/120) |
| `einmessen` | 30/30/29/27 eingemessen bei 35/50/60/70°, 0 falsch eingemessen, Feldgröße im Mittel 0,22-0,29 % daneben |
| `zuordnung` | 11 812/11 812 Figuren richtig |
| `tracking` | 1196/1200 Züge richtig, 4 markiert, 0 still falsch |
| `stress` | schwächster Fall: Tiefe 3 px gegen das Farbbild versetzt, 98,2 % der Figuren richtig |

## Konventionen

- Bild 640 x 480, `K` mit f = 600 px, keine Verzeichnung.
- Brettrahmen: Ursprung in der Brettmitte, x zur h-Linie, y zur 8. Reihe, z nach oben, Meter.
- Pose: `X_cam = R @ X_brett + t`. Felder in python-chess-Nummerierung (a1 = 0, h8 = 63).
- `Frame.depth`: z-Tiefe in Metern, 0 = ungültig. `Frame.labels`: 0 = keine Figur, k = Figur k.
  `Frame.classes`: k -> FEN-Symbol (`'P'` weißer Bauer, `'n'` schwarzer Springer).
- Figuren mit weniger als 25 sichtbaren Pixeln gelten als nicht erkannt und fehlen in `labels`.

## Auswertung

Jede Zählmetrik steht mit 95 %-Konfidenzintervall (Clopper-Pearson) da. Der Vergleich mit der Baseline
ist ein einseitiger Fisher-Test; `SCHLECHTER` heißt p < 0,05. Bei `tracking` gilt eine `None`-Antwort als
markiert, nicht als Fehler; nach markiertem oder falschem Zug setzt der Harness den Tracker auf die
Wahrheit zurück, wie eine Bestätigung durch den Menschen. Am wichtigsten sind `still falsch` und
`falsch eingemessen`: Das sind die Fälle, in denen Stockfish eine falsche Stellung bekäme.

## Grenzen

Masken und Klassen sind perfekt, wie im Dokument angenommen. Nicht simuliert: Linsenverzeichnung,
Glanzlichter, Tiefenlöcher auf schwarzen Figuren, Hände, der asymmetrische Springer. Das Brettbild bleibt
bei der Wölbung eben, nur Figuren und Tiefe folgen ihr. Der Maßstab für den echten Betrieb bleibt ein
Testsatz aus echten Partien.

## Aufbau

- `sim.py`: Szenen-Generator (Brett, Figuren als taillierte Rotationskörper, Tiefe mit Rauschen,
  Flying Pixels und ToF-Fehler, Partien).
- `suites.py`: die sechs Suiten; jede Szene hat den Seed `[Suite, Zeile, Szene]`.
- `run.py`: Aufruf, Bericht, Vergleich, Baseline schreiben.
- `reference/`: die Algorithmen aus dem Dokument, eingefroren; `reference/frozen/` ist eine Kopie von
  `src/board` aus Commit f3d51b2, damit die Baseline nicht wandert, wenn du `src/board` änderst.
- `adapters/`: `reference`, `original` (dein Code vor den Änderungen) und `template`.
