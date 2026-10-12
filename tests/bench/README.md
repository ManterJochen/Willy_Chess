# Willy_Chess benchmark: from board to FEN

Simulated scenes with exactly known ground truth, fixed seeds, a frozen reference implementation
and a baseline. Your implementation runs through an adapter against the same scenes; the script tells you
where it is significantly worse, and then ends with exit code 1.

## Installation

The benchmark lives in `tests/bench/`. From `tests/`:

```bash
pip install -r bench/requirements.txt
python -m bench.run --quick            # reference, all suites, about 3 minutes
```

All commands run from `tests/`, on Windows just the same.

## Testing your implementation

1. Copy `bench/adapters/template.py` to `bench/adapters/mine.py` and connect the functions to your
   code. Anything missing or raising `NotImplementedError` is skipped.
2. Run the reference once on your machine (the scenes depend slightly on the NumPy and
   OpenCV version, this way you are sure to compare on identical inputs):
   ```bash
   python -m bench.run --write-baseline bench/baseline_local.json
   ```
3. Run your adapter against it:
   ```bash
   python -m bench.run --adapter bench/adapters/mine.py --baseline bench/baseline_local.json
   ```

If a suite is followed by `[inputs identical]`, exactly the same images ran as in the baseline.
`--quick` takes a fifth of the scenes with the same seeds, for quick rounds while developing.
`--dump-failures DIR` stores every failed scene as PNG and NPZ (image, depth, labels, truth).

## The suites

| Suite | Checks | Scenes (full) | Adapter function |
| --- | --- | --- | --- |
| `corners` | Subpixel accuracy of the inner corners, empty board 0-45° | 40 | `corners(gray)` |
| `board` | Board from the image, with flat pieces or clutter, 30-60° | 4 x 30 | `board_homography(bgr)` |
| `calibration` | Starting position, square size 45-65 mm unknown, piece set ±15 %: pose, orientation, scale | 4 x 30 | `calibrate(frame)` + `assign` |
| `mapping` | Piece to square at a known pose, random game positions, 0-70° | 7 x 60 | `assign(frame, calib)` |
| `tracking` | Games from the starting position, one move per image, 50/60/70° | 3 x 8 games x 50 moves | `Tracker` |
| `stress` | 70°, one violated assumption each: mask edge, piece set, depth shift, bulge | 8 x 20 | `assign(frame, calib)` |

Full run time of the reference about 13 minutes, 7 of them for `tracking`.

## Baseline (reference, full run)

| Suite | Result |
| --- | --- |
| `corners` | 1958/1960 corners found, RMSE 0.042 px (original: 0.349 px) |
| `board` | 120/120 boards correct (original: 12/120) |
| `calibration` | 30/30/29/27 calibrated at 35/50/60/70°, 0 miscalibrated, square size off by 0.22-0.29 % on average |
| `mapping` | 11,812/11,812 pieces correct |
| `tracking` | 1196/1200 moves correct, 4 flagged, 0 silently wrong |
| `stress` | weakest case: depth shifted 3 px against the colour image, 98.2 % of the pieces correct |

## Conventions

- Image 640 x 480, `K` with f = 600 px, no distortion.
- Board frame: origin at the board centre, x towards the h-file, y towards the 8th rank, z up, metres.
- Pose: `X_cam = R @ X_board + t`. Squares in python-chess numbering (a1 = 0, h8 = 63).
- `Frame.depth`: z-depth in metres, 0 = invalid. `Frame.labels`: 0 = no piece, k = piece k.
  `Frame.classes`: k -> FEN symbol (`'P'` white pawn, `'n'` black knight).
- Pieces with fewer than 25 visible pixels count as not detected and are missing from `labels`.

## Evaluation

Every count metric is shown with a 95 % confidence interval (Clopper-Pearson). The comparison with the baseline
is a one-sided Fisher test; `WORSE` means p < 0.05. In `tracking` a `None` answer counts as
flagged, not as an error; after a flagged or wrong move the harness resets the tracker to the
truth, like a confirmation by the human. Most important are `silently wrong` and
`miscalibrated`: these are the cases in which Stockfish would get a wrong position.

## Limits

Masks and classes are perfect, as assumed in the document. Not simulated: lens distortion,
specular highlights, depth holes on black pieces, hands, the asymmetric knight. The board image stays
flat under the bulge, only pieces and depth follow it. The yardstick for real operation remains a
test set from real games.

## Structure

- `sim.py`: scene generator (board, pieces as waisted solids of revolution, depth with noise,
  flying pixels and ToF errors, games).
- `suites.py`: the six suites; every scene has the seed `[suite, row, scene]`.
- `run.py`: invocation, report, comparison, writing the baseline.
- `reference/`: the algorithms from the document, frozen; `reference/frozen/` is a copy of
  `src/board` from commit f3d51b2, so that the baseline does not drift when you change `src/board`.
- `adapters/`: `reference`, `original` (your code before the changes) and `template`.
