"""The game's flight recorder and the training set made from it: labels from the position, the depth as the witness.

    cd tests
    python -m unittest test_game_log -v
"""
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import chess
import cv2 as cv
import numpy as np

TESTS = Path(__file__).resolve().parent
for p in (TESTS, TESTS.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.game.config import ChessConfig  # noqa: E402
from src.game.game import ChessGame  # noqa: E402
from src.game.play import play_games  # noqa: E402
from src.game.robot_moves import RobotMover  # noqa: E402
from src.game.tracker import MoveTracker, TrackResult  # noqa: E402
from src.mapping.calibration import CalibrationResult  # noqa: E402
from src.mapping.chessboard_mapping_3d import ChessBoardMapping3D  # noqa: E402
from src.mapping.frame import CLASSES, SYMBOL_TO_CLASS  # noqa: E402
from src.mapping.geometry import BoardCamera, square_centres  # noqa: E402
from src.pieces import export as exporter  # noqa: E402
from src.pieces.record import LOG_NAME, GameLog, load_view, read_game  # noqa: E402
from test_game_sim import FakeRobot, GameRig, RandomEngine, ScriptedHuman  # noqa: E402
from test_mapping_sim import SimDetector, SimPose  # noqa: E402

LOOK = (35.0, 20.0)


def played(root: Path, seed: int, plies: int, human_cls=ScriptedHuman, detect=None, setup=None):
    """One logged game on the simulated table. Returns the table, the game, the human and the log read back."""
    rig = GameRig(seed)
    rig.look(*LOOK)
    detector = SimDetector(rig)
    if detect is not None:
        real = detector.detect
        detector.detect = lambda crop, *, segment=False, threshold=None: detect(real(crop, segment=segment, threshold=threshold))
    mapping = ChessBoardMapping3D(rig, detector, SimPose(rig))
    config = ChessConfig()
    human = human_cls(rig, seed + 100, plies)
    log = GameLog(root)
    game = ChessGame(mapping, RandomEngine(rig, seed + 200), RobotMover(FakeRobot(rig), mapping, config.robot), human,
                     config, go_look=lambda: rig.look(*LOOK), log=log)
    if setup is not None:
        setup(rig, mapping)
    game.run()
    log.close()
    return rig, game, human, read_game(log.folder)


def bishops_seen_as_knights(found):
    """A detector that takes every white bishop for a knight."""
    for obj in found:
        if obj.label == "white bishop":
            obj.label = "white knight"
    return found


class GameLogTest(unittest.TestCase):
    def test_every_view_and_decision_is_kept(self):
        with tempfile.TemporaryDirectory() as folder:
            rig, game, _, logged = played(Path(folder), 1, plies=4)
            kinds = {event["event"] for event in logged.events}
            self.assertLessEqual({"game", "view", "calibration", "step", "verify", "settled", "ask", "status", "engine",
                                  "execution", "result"}, kinds)
            self.assertEqual(len(list((logged.folder / "views").glob("*.png"))), len(logged.views))
            self.assertEqual(len(list((logged.folder / "views").glob("*.npz"))), len(logged.views))
            result = next(event for event in logged.events if event["event"] == "result")
            self.assertEqual(result["moves"], [move.uci() for move in game.tracker.board.move_stack])

    def test_a_view_reads_back_as_the_game_saw_it(self):
        with tempfile.TemporaryDirectory() as folder:
            _, game, _, logged = played(Path(folder), 2, plies=3)
            seen = game.state.observation
            frame, calib, camera_to_base = load_view(logged, max(logged.views))
            np.testing.assert_array_equal(frame.bgr, seen.frame.bgr)
            np.testing.assert_array_equal(frame.labels, seen.frame.labels)
            np.testing.assert_array_equal(frame.depth, seen.frame.depth)
            self.assertEqual(frame.classes, seen.frame.classes)
            self.assertEqual(frame.roi, seen.frame.roi)
            np.testing.assert_allclose(calib.R, seen.calib.R)
            np.testing.assert_allclose(camera_to_base, seen.camera_to_base)

    def test_each_view_knows_the_position_it_showed(self):
        """A view of the human's move shows the position after it, though the tracker had not committed it yet."""
        with tempfile.TemporaryDirectory() as folder:
            _, game, _, logged = played(Path(folder), 3, plies=3)
            first_move = game.tracker.board.move_stack[0]
            view, (physical, how) = next((n, logged.truth[n]) for n in sorted(logged.truth)
                                         if logged.views[n]["moment"] == "your_move" and logged.truth[n][1] == "seen")
            self.assertEqual(logged.views[view]["ply"], 0)
            self.assertIn(first_move.to_square, physical)
            self.assertNotIn(first_move.from_square, physical)

    def test_a_refused_calibration_is_kept_with_its_position(self):
        def once_refused(rig, mapping):
            real, calls = mapping.calibrate, []

            def calibrate(expected=None):
                calls.append(1)
                result = real(expected)
                return CalibrationResult(None, "a bad frame") if len(calls) == 1 else result

            mapping.calibrate = calibrate

        with tempfile.TemporaryDirectory() as folder:
            _, _, _, logged = played(Path(folder), 4, plies=1, setup=once_refused)
            self.assertIsNone(logged.views[0]["square"])
            self.assertEqual(logged.truth[0][1], "calibration")
            self.assertEqual(len(logged.truth[0][0]), 32)

    def test_a_disk_that_fails_does_not_stop_the_game(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()) as out, \
                mock.patch("src.pieces.record.cv.imwrite", side_effect=OSError("disk full")):
            rig, game, _, _ = played(Path(folder), 5, plies=3)
        self.assertEqual(out.getvalue().count("has a problem"), 1)
        self.assertGreaterEqual(game.tracker.board.ply(), 3)

    def test_a_line_a_crash_cut_off_is_left_out(self):
        with tempfile.TemporaryDirectory() as folder:
            with GameLog(folder) as log:
                log.event("game", human="white")
                path = log.folder
            with (path / LOG_NAME).open("a", encoding="utf-8") as cut:
                cut.write('{"event": "vie')
            self.assertEqual([event["event"] for event in read_game(path).events], ["game"])

    def test_each_game_gets_its_log_and_a_late_question_is_no_problem(self):
        rig = GameRig(6)
        mapping = ChessBoardMapping3D(rig, SimDetector(rig), SimPose(rig))
        config = ChessConfig()
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()) as out:
            config.game.pgn_dir = folder
            cell = SimpleNamespace(mapping=mapping, mover=RobotMover(FakeRobot(rig), mapping, config.robot),
                                   go_look=lambda: rig.look(*LOOK), views=1)

            class TwoMoves(ScriptedHuman):
                def wait(self, state, accept):
                    return "resign" if "resign" in accept and self.rig.logical.ply() >= 2 else super().wait(state, accept)

            human = TwoMoves(rig, 7, 99)
            real_wait = human.wait
            human.wait = lambda state, accept: "quit" if "New game" in state.status else real_wait(state, accept)
            play_games(cell, RandomEngine(rig, 8), human, config, Path(folder, "logs"))
            logs = list(Path(folder, "logs").iterdir())
            self.assertEqual(len(logs), 1)
            self.assertEqual(next(e for e in read_game(logs[0]).events if e["event"] == "result")["reason"], "you resigned")
        self.assertNotIn("has a problem", out.getvalue())


class LabelViewTest(unittest.TestCase):
    """What the position says about one view, whatever the detector made of it."""

    def view(self, seed: int = 31):
        rig = GameRig(seed)
        rig.look(*LOOK)
        mapping = ChessBoardMapping3D(rig, SimDetector(rig), SimPose(rig))
        self.assertEqual(mapping.calibrate().reason, "ok")
        observation = mapping.observe()
        return rig, observation, {sq: k for k, sq in observation.squares.items()}

    def test_a_view_the_detector_got_right(self):
        rig, obs, _ = self.view()
        labels = exporter.label_view(obs.frame, obs.calib, rig.physical)
        self.assertEqual(labels.problems, [])
        self.assertFalse(labels.hard)
        self.assertGreaterEqual(len(labels.boxes), 30)

    def test_a_wrong_type_is_labelled_with_the_right_one(self):
        rig, obs, key = self.view()
        obs.frame.classes[key[chess.C1]] = "N"
        labels = exporter.label_view(obs.frame, obs.calib, rig.physical)
        self.assertEqual(labels.wrong_type, 1)
        self.assertTrue(labels.hard)
        ys, xs = np.nonzero(obs.frame.labels == key[chess.C1])
        box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
        self.assertEqual(next(b[0] for b in labels.boxes if b[1:] == tuple(int(v) for v in box)), "B")

    def test_a_wrong_colour_is_corrected_too(self):
        rig, obs, key = self.view()
        obs.frame.classes[key[chess.E2]] = "p"
        labels = exporter.label_view(obs.frame, obs.calib, rig.physical)
        self.assertEqual(labels.problems, [])
        self.assertEqual(labels.wrong_type, 1)

    def test_a_piece_the_detector_missed_gets_its_box_from_the_depth(self):
        rig, obs, key = self.view()
        k = key[chess.E2]
        ys, xs = np.nonzero(obs.frame.labels == k)
        a0, b0, a1, b1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        obs.frame.labels[obs.frame.labels == k] = 0
        del obs.frame.classes[k]
        labels = exporter.label_view(obs.frame, obs.calib, rig.physical)
        self.assertEqual(labels.problems, [])
        self.assertEqual(labels.missed, 1)
        self.assertTrue(labels.hard)

        def iou(box):
            x0, y0, x1, y1 = box
            inter = max(0, min(x1, a1) - max(x0, a0)) * max(0, min(y1, b1) - max(y0, b0))
            return inter / ((x1 - x0) * (y1 - y0) + (a1 - a0) * (b1 - b0) - inter)

        self.assertGreater(max(iou(b[1:]) for b in labels.boxes if b[0] == "P"), 0.6)

    def test_a_false_detection_on_the_bare_board_is_background(self):
        rig, obs, _ = self.view()
        cam = BoardCamera(obs.frame.K, obs.calib.R, obs.calib.t)
        centre = square_centres(obs.calib.square)[chess.E4]
        u, v = cam.project(np.array([[centre[0], centre[1], 0.0]]))[0].astype(int)
        marked = np.zeros(obs.frame.labels.shape, np.uint8)
        cv.circle(marked, (int(u), int(v)), 6, 1, -1)
        obs.frame.labels[marked > 0] = 99
        obs.frame.classes[99] = "P"
        labels = exporter.label_view(obs.frame, obs.calib, rig.physical)
        self.assertEqual(labels.problems, [])
        self.assertEqual(labels.false, 1)
        self.assertTrue(labels.hard)

    def test_a_piece_on_a_square_the_position_has_empty_is_refused(self):
        """Somebody's wrong move, seen before it was taken back: the position cannot explain it."""
        rig, _, _ = self.view()
        mapping = ChessBoardMapping3D(rig, SimDetector(rig), SimPose(rig))
        self.assertEqual(mapping.calibrate().reason, "ok")
        start = dict(rig.physical)
        wrong = dict(start)
        wrong[chess.G3] = wrong.pop(chess.G1)
        rig.set_physical(wrong)
        obs = mapping.observe()
        self.assertTrue(exporter.label_view(obs.frame, obs.calib, start).problems)

    def test_a_hand_over_the_board_is_refused(self):
        rig, _, _ = self.view()
        mapping = ChessBoardMapping3D(rig, SimDetector(rig), SimPose(rig))
        self.assertEqual(mapping.calibrate().reason, "ok")
        rig.hand = ((260, 180, 380, 260), 0.08)
        obs = mapping.observe()
        self.assertTrue(exporter.label_view(obs.frame, obs.calib, rig.physical).problems)

    def test_a_piece_the_position_has_and_nothing_shows_is_refused(self):
        rig, _, _ = self.view()
        mapping = ChessBoardMapping3D(rig, SimDetector(rig), SimPose(rig))
        self.assertEqual(mapping.calibrate().reason, "ok")
        start = dict(rig.physical)
        gone = dict(start)
        del gone[chess.E2]
        rig.set_physical(gone)
        obs = mapping.observe()
        problems = exporter.label_view(obs.frame, obs.calib, start).problems
        self.assertTrue(any("e2" in problem for problem in problems), problems)

    def test_beside_the_board_is_painted_over_not_labelled(self):
        rig, obs, _ = self.view()
        obs.frame.roi = (0, 0, obs.frame.bgr.shape[1], obs.frame.bgr.shape[0])
        labels = exporter.label_view(obs.frame, obs.calib, rig.physical)
        before = len(labels.boxes)
        labels.off_board = [(5, 5, 45, 45)]
        image, lines = exporter.crop_view(obs.frame, labels)
        self.assertEqual(len(lines), before)
        self.assertEqual(len(np.unique(image[5:45, 5:45].reshape(-1, 3), axis=0)), 1)


class ExportTest(unittest.TestCase):
    def test_labels_come_from_the_position_and_hard_views_count_more(self):
        with tempfile.TemporaryDirectory() as folder:
            logs, out = Path(folder, "logs"), Path(folder, "out")
            played(logs, 41, plies=2, detect=bishops_seen_as_knights)
            report = exporter.export(logs, out, hard_repeat=3)
            self.assertGreater(report["hard"], 0)
            self.assertGreater(report["wrong_type"], 0)
            labels = sorted((out / "train" / "labels").glob("*.txt"))
            first = labels[0].read_text().splitlines()
            names = [CLASSES[int(line.split()[0])] for line in first]
            self.assertEqual(names.count(SYMBOL_TO_CLASS["B"]), 2)
            self.assertEqual(names.count(SYMBOL_TO_CLASS["N"]), 2)
            for line in first:
                self.assertTrue(all(0.0 <= float(value) <= 1.0 for value in line.split()[1:]))
            stems = [p.stem for p in (out / "train" / "images").glob("*.png")]
            self.assertTrue(any(stem.endswith("_dup2") for stem in stems))
            self.assertEqual(len(stems), len(labels))
            self.assertEqual(report["boxes"][SYMBOL_TO_CLASS["B"]] % 2, 0)
            self.assertTrue(list((out / "preview").glob("*.jpg")))

    def test_the_image_is_what_the_detector_gets_in_a_game(self):
        with tempfile.TemporaryDirectory() as folder:
            logs, out = Path(folder, "logs"), Path(folder, "out")
            _, _, _, logged = played(logs, 42, plies=1)
            exporter.export(logs, out)
            frame, _, _ = load_view(logged, 0)
            x0, y0, x1, y1 = frame.roi
            image = cv.imread(str(out / "train" / "images" / f"{logged.name}_000000.png"))
            self.assertEqual(image.shape[:2], (y1 - y0, x1 - x0))

    def test_a_move_only_a_click_could_settle_comes_along(self):
        """The views the tracker could not explain are the ones the detector needs most."""
        def blind(tracker, frame, calib=None):
            return TrackResult(None, 99.0, None, flagged=True)

        with tempfile.TemporaryDirectory() as folder:
            logs, out = Path(folder, "logs"), Path(folder, "out")
            with mock.patch.object(MoveTracker, "step", blind):
                _, game, human, logged = played(logs, 43, plies=2)
            self.assertGreater(human.typed, 0)
            self.assertIn("clicked", {how for _, how in logged.truth.values()})
            self.assertGreater(len(logged.doubted), 0)
            clicked = [n for n, (_, how) in logged.truth.items() if how == "clicked"]
            exporter.export(logs, out)
            names = {p.stem for p in (out / "train" / "images").glob("*.png")}
            for number in clicked:
                self.assertIn(f"{logged.name}_{number:06d}", names)

    def test_a_wrong_move_is_labelled_with_what_the_board_showed(self):
        """The camera confirmed the knight on g3 before it was taken back: that view trains with the knight on g3, not
        with the position of the move played after it (which put it into review before)."""
        class CheatingHuman(ScriptedHuman):
            def wait(self, state, accept):
                if "not allowed" in state.status:
                    self.rig.set_physical(self.before)
                elif "move" in accept and not hasattr(self, "before"):
                    self.before = dict(self.rig.physical)
                    physical = dict(self.rig.physical)
                    physical[chess.G3] = physical.pop(chess.G1)
                    self.rig.set_physical(physical)
                    return "done"
                return super().wait(state, accept)

        with tempfile.TemporaryDirectory() as folder:
            logs, out = Path(folder, "logs"), Path(folder, "out")
            _, _, _, logged = played(logs, 44, plies=2, human_cls=CheatingHuman)
            wrong = min(logged.doubted)
            physical, how = logged.truth[wrong]
            self.assertEqual((how, physical.get(chess.G3), physical.get(chess.G1)), ("wrong_move", "N", None))
            report = exporter.export(logs, out)
            self.assertFalse(any(f"_{wrong:06d}" in reason for reason in report["review_reasons"]), report["review_reasons"])
            self.assertTrue(list(out.glob(f"*/images/*_{wrong:06d}*")), "the view of the wrong move was not exported")

    def test_a_refused_calibration_frame_is_labelled_from_the_board_locked_later(self):
        def once_refused(rig, mapping):
            real, calls = mapping.calibrate, []

            def calibrate(expected=None):
                calls.append(1)
                result = real(expected)
                return CalibrationResult(None, "a bad frame") if len(calls) == 1 else result

            mapping.calibrate = calibrate

        with tempfile.TemporaryDirectory() as folder:
            logs, out = Path(folder, "logs"), Path(folder, "out")
            _, _, _, logged = played(logs, 45, plies=1, setup=once_refused)
            exporter.export(logs, out)
            text = (out / "train" / "labels" / f"{logged.name}_000000.txt").read_text()
            self.assertGreaterEqual(len(text.splitlines()), 28)

    def test_whole_games_go_to_validation(self):
        games = [Path(f"g{i}") for i in range(11)]
        self.assertEqual(exporter.validation_games(games, 5), {games[4], games[9]})
        self.assertEqual(exporter.validation_games(games[:3], 5), {games[2]})
        self.assertEqual(exporter.validation_games(games[:1], 5), set())

    def test_no_view_of_a_validation_game_lands_in_training(self):
        with tempfile.TemporaryDirectory() as folder:
            logs, out = Path(folder, "logs"), Path(folder, "out")
            _, _, _, first = played(logs, 46, plies=1)
            _, _, _, second = played(logs, 47, plies=1)
            report = exporter.export(logs, out)
            self.assertEqual(report["validation_games"], [second.name])
            train = {p.name.split("_0")[0] for p in (out / "train" / "images").glob("*.png")}
            self.assertEqual(train, {first.name})
            self.assertTrue(all(p.name.startswith(second.name) for p in (out / "val" / "images").glob("*.png")))
            self.assertIn("val: val/images", (out / "data.yaml").read_text())

    def test_a_folder_it_did_not_write_is_left_alone(self):
        with tempfile.TemporaryDirectory() as folder:
            foreign = Path(folder, "data")
            foreign.mkdir()
            (foreign / "precious.txt").write_text("9218 training images")
            with self.assertRaises(ValueError):
                exporter.export(Path(folder, "logs"), foreign)
            self.assertTrue((foreign / "precious.txt").exists())

    def test_it_can_be_written_again_as_more_games_are_played(self):
        with tempfile.TemporaryDirectory() as folder:
            logs, out = Path(folder, "logs"), Path(folder, "out")
            played(logs, 48, plies=1)
            first = exporter.export(logs, out)
            second = exporter.export(logs, out)
            self.assertEqual(first["train"], second["train"])


if __name__ == "__main__":
    unittest.main()
