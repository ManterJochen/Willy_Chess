# Setup

From an empty machine to the first game, on Windows 10/11 with Python 3.11. Run everything from the repository root
with the `.venv` active. The robot, camera and hand-eye calibration come from Workaholic-Willy: if your cell does not
run there yet, start with its [Quickstart](../external/Workaholic-Willy/docs/Quickstart.md) and
[calibration guide](../external/Workaholic-Willy/docs/calibration-setup.md) (both are there after step 1).

## 1. Install

Run `git lfs install` once, clone this repository, then:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
git clone https://github.com/ManterJochen/Workaholic-Willy.git external/Workaholic-Willy
pip install -r requirements.txt                       # without a GPU: requirements-cpu.txt
pip install -e .\external\Workaholic-Willy --no-deps
python -m src.pieces.fetch                            # RT-DETR base + SAM2, 1.1 GB
git lfs pull                                          # the detector weights
```

Stockfish is included for Windows. Elsewhere, install it or set `engine.path`.

## 2. Point it at your cell

In [`config/willy_chess.yaml`](../config/willy_chess.yaml):

```yaml
willy:
  data_dir: C:/path/to/your/cell/config
```

- It must be the **inner config folder**, the one that directly holds `camera/`, `robot/` and `models/`.
- Absolute, or relative to the repository. Use forward slashes.
- `--willy-data DIR` overrides it for one run.

Check it. This should print `OK: ... validates`, otherwise it names the file and line to fix:

```powershell
python -m src.config --data C:/path/to/your/cell/config
```

Close Willy's console before running anything below: it holds the cell, and the arm will not connect (`CellBusy`).

## 3. Set the look pose

In `robot/robot.yaml` of that folder, in degrees as the pendant shows them:

```yaml
robot:
  motion:
    look_deg:
      - [j1, j2, j3, j4, j5, j6]    # look_deg[0], the pose the game looks from; it may be the home pose
```

The whole board must be in the picture. A closer, more oblique pose detected the pieces clearly better at our cell.
A fixed camera needs no look pose: the arm goes home, out of its view.

## 4. Choose the game settings

All in `config/willy_chess.yaml`. Everything not listed can stay as it is.

| Key | What it does |
| --- | --- |
| `robot.grasp_height_fraction` | Where the middle of the jaws closes, as a fraction of the measured piece height, per type. The fingertips reach about 10 mm lower, so they must stay above the board |
| `robot.graveyard.side` | Where captured pieces go: `+x` beyond the h-file, `-x` beyond the a-file, `+y` beyond rank 8, `-y` beyond rank 1 |
| `robot.home_before_graveyard` | Back over the board (home) before a captured piece goes out, on by default |
| `robot.board_surface_mm` | The board surface above the robot base by ruler; `null` (default) takes the camera's. Only while the camera's height is off, see the update notes |
| `engine.elo`, `engine.think_time_s` | Willy's strength (1320 to 3190, `null` is full strength) and time per move |
| `game.human_color` | The colour preselected at the start |
| `game.promotion_mode` | `pawn_stays` (no spare queen, the pawn counts as the new piece) or `piece_swapped` |
| `game.look_views` | Look poses the game may use, see step 5 |
| `record.enabled` | Keep a log of every game, on by default, see step 7 |

Grasp heights at our cell (Robotiq Hand-E, Staunton set, 45 mm squares): p 0.78, n 0.60, b 0.60, r 0.80, q 0.68, k 0.65. The fingertips reach 10.5 mm below the grasp centre, so it stays at least ~28 mm above the board, clear of the foot; king and queen are held just under the collar, the knight at its neck.

## 5. Check the cell

Set up the starting position and go through these in order. The arm moves in the last two steps, so keep a hand on the
emergency stop.

| Command | Good result |
| --- | --- |
| `python -m src.game.detect_test` | 32 pieces, 16 of each colour. Bishops and knights swapped are fine. Pictures in `data/detect_test/` |
| `python -m src.game.mapping_test` | `right square` all of them. `grasp error`: few or none over 8 mm. `free board height` near 0, a steady rise across the files means the depth is tilted |
| `python -m src.game.touch_test` | The grasp centre stops over the middle of each square and above the graveyard corners. An offset that is the same everywhere is a calibration error |
| `python -m src.game.touch_test --corners` | Also goes down to a1, h1, h8 and a8. Take each rook off when it asks |

**Several look poses (optional).** Add them under `look_deg`, then run `python -m src.game.mapping_test --all-looks`.
You want 0 squares out of the picture and a board that stays put against look 0. Then set `game.look_views` to the
number of poses: what one view cannot settle is looked at from the next pose, by way of `look_deg[0]`, before anybody is
asked. Every piece out of the picture costs the move recognition, so leave it at 1 until this looks right.

## 6. Play

```powershell
python main.py
```

| Key | |
| --- | --- |
| W / B | Your colour |
| Space | Your move is done |
| Click two squares | Enter a move by hand, when the camera did not recognise it |
| R | Calibrate again, after the board was moved |
| F | The board is right, play on (offered when the camera keeps disagreeing with Willy's move) |
| Q / R / B / N | The promotion piece |
| D / G | Offer a draw / resign |
| Esc | End |

Games are saved as PGN in `data/games/`.

## 7. Get more training data

Every game keeps its log in `data/game_logs/`. After a few games:

```powershell
python -m src.pieces.export
```

Look through `data/cell_dataset/preview/` (the views the detector got wrong, with the labels drawn) and
`review/reasons.txt` (views left out, and why). Then train on the workstation, with a Willy that has the repaired loss:

```powershell
python -m src.pieces.train --data data/cell_dataset --base-model docs/weights/chess_pieces_rtdetr/last --epochs 10 --batch 24 --out docs/weights/chess_pieces_v3
```

Then set `detector.weights` to the new folder. The report must show `scores` around 0.8 and no `SCORES COLLAPSED`. `last/`
is not in Git, it exists only on the machine that trained the weights.

## If something goes wrong

| What you see | What to do |
| --- | --- |
| `no Willy data directory` | Set `willy.data_dir`, step 2 |
| The program prints a config refusal | `python -m src.config --data <dir>` names the file and line |
| `No module named 'willy'`, or the integration tests are skipped | `pip install -e .\external\Workaholic-Willy --no-deps`, also after moving the repository |
| `no detector weights`, or `is a Git LFS pointer` | `git lfs install`, then `git lfs pull` |
| `Stockfish not found` | Set `engine.path` |
| `CellBusy` | Close the console or whichever program the message names |
| `only X of 32 pieces stand on an expected square` | Usually one bad frame, the calibration tries again by itself. If it stays: is this the starting position, is the light on the board? |
| `no board grid found` | Is the whole board in the picture? A view straight along the board's edges finds it less often, turn the look pose a little |
| `N pieces stand on a square expected for another type` | The detector sees too little: move the look pose closer |
| Nothing is detected, scores around 0.07 | These weights were trained without the repaired loss. Put `"decoder_layers": 5` in their `config.json` |
| `Move not recognised` | Check that each piece stands on its square and press space again, or click the move |
| `the board looks different than expected` | Fix the board and press space. From the second time, F plays on |
| `The board was moved` | Press R. If you had already moved, take the move back first |
