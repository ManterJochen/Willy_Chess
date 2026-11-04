"""
The whole program at a desk, without the robot: play_games, the same loop as main.py, from the first look to
the saved PGN.

Real: the game, the move tracking, ChessBoardMapping3D (corners, calibration, scale from the depth, the lock in
the robot frame, pieces to squares), RobotMover, Stockfish, the Pygame window, the PGN, and Willy: its
console_dummy tree, the hand from its registry and every motion through its Robot API on the dummy arm.
Simulated: the RGB-D camera (rendered scenes, a fixed camera), the detector (the renderer's masks; the trained
RT-DETR needs real photos) and the human (random legal moves). Where Willy's arm grips and sets down, the
simulated pieces move, but only after Willy reported the motion done.

    cd tests
    python desk_run.py                    # watch it in the window
    python desk_run.py --headless --shot final.png
"""
import argparse
import os
import sys
import time

import chess

from test_game_sim import FakeRobot, GameRig, ScriptedHuman  # also puts the repository on sys.path
from test_mapping_sim import SimDetector, SimPose

from src.game.config import load_config
from src.game.engine import Engine
from src.game.play import Cell, play_games
from src.game.robot_moves import DECLINE, JawGeometry, RobotMover
from src.game.tracker import physical_after
from src.mapping.chessboard_mapping_3d import ChessBoardMapping3D


class DeskRobot:
    """Willy's dummy arm commands every motion; the simulated pieces follow once Willy reported it done."""

    def __init__(self, robot, rig: GameRig):
        self.robot, self.hands = robot, FakeRobot(rig)

    def tool_down(self, x, y, z, *, closing_axis="x", yaw_deg=0.0):
        return (self.robot.tool_down(x, y, z, closing_axis=closing_axis, yaw_deg=yaw_deg),
                self.hands.tool_down(x, y, z, closing_axis=closing_axis, yaw_deg=yaw_deg))

    def pick(self, pose, width_mm, **kw):
        report = self.robot.pick(pose[0], width_mm, **kw)
        return self.hands.pick(pose[1], width_mm) if report.ok else report

    def place(self, pose, **kw):
        report = self.robot.place(pose[0], **kw)
        return self.hands.place(pose[1]) if report.ok else report

    def release(self):
        return self.robot.release()


class DeskEngine:
    """Stockfish; its move also becomes the simulated table's truth as it is chosen."""

    def __init__(self, engine: Engine, rig: GameRig):
        self.engine, self.rig = engine, rig
        self.before: tuple[chess.Board, dict[int, str]] | None = None

    def best_move(self, board: chess.Board) -> chess.Move:
        move = self.engine.best_move(board)
        self.before = (self.rig.logical.copy(), dict(self.rig.physical))
        self.rig.logical.push(move)
        return move

    def evaluation(self, board: chess.Board) -> str:
        return self.engine.evaluation(board)

    def score(self, board: chess.Board, color: bool) -> int:
        return self.engine.score(board, color)


class WatchedHuman(ScriptedHuman):
    """
    The scripted human in front of the Pygame window: every state is drawn before it answers, and where the
    game asks to finish or fix Willy's move by hand, the pieces are set as that move leaves them.
    """

    def __init__(self, rig: GameRig, seed: int, max_plies: int, window, pause_s: float, engine: DeskEngine):
        super().__init__(rig, seed, max_plies)
        self.window, self.pause_s, self.engine = window, pause_s, engine
        self.by_hand = 0

    def show(self, state) -> None:
        super().show(state)
        self.window.show(state)

    def wait(self, state, accept: set[str]) -> str:
        self.window.show(state)
        time.sleep(self.pause_s)
        if ("by hand" in state.status or "looks different" in state.status) and self.engine.before is not None:
            board, physical = self.engine.before
            self.rig.set_physical(physical_after(board, physical, self.rig.logical.peek(), "pawn_stays"))
            self.by_hand += 1
            return "done"
        return super().wait(state, accept)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Play a whole game at a desk: simulated camera, Willy's dummy arm.")
    parser.add_argument("--seed", type=int, default=1, help="the simulated table and the human's moves")
    parser.add_argument("--plies", type=int, default=200, help="the human stops after this many plies")
    parser.add_argument("--pause", type=float, default=0.3, help="seconds the window shows each question")
    parser.add_argument("--headless", action="store_true", help="no window (the view is still drawn)")
    parser.add_argument("--shot", default=None, help="save the final window to this image")
    parser.add_argument("--pgn-dir", default=None, help="where the PGN goes (default: game.pgn_dir of the config)")
    parser.add_argument("--log-dir", default=None, help="keep the game log here, as a real game does (default: none)")
    args = parser.parse_args(argv)
    if args.headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    from willy import Robot, load_tree
    from src.ui.pygame_view import PygameView

    config = load_config(None)
    config.engine.think_time_s = 0.2
    config.game.pgn_dir = args.pgn_dir or config.game.pgn_dir
    tree = load_tree("console_dummy")
    rig = GameRig(args.seed)
    robot = Robot.from_tree(tree)
    desk_robot = DeskRobot(robot, rig)
    mapping = ChessBoardMapping3D(rig, SimDetector(rig), SimPose(rig), detector_threshold=config.detector.threshold,
                                  grasp_height_fraction=config.robot.grasp_height_fraction)
    mover = RobotMover(desk_robot, mapping, config.robot, JawGeometry.from_tree(tree), camera_world=False)

    def go_look() -> None:
        report = robot.home(decline=DECLINE)
        if not report.ok:
            raise RuntimeError(f"the arm did not reach home:\n{report}")
        rig.look(35.0, 20.0)

    cell = Cell(desk_robot, mapping, mover, go_look, {"decline": DECLINE})
    started = time.perf_counter()
    with robot.connected(), Engine(config.engine) as stockfish, PygameView() as window:
        engine = DeskEngine(stockfish, rig)
        human = WatchedHuman(rig, args.seed + 100, args.plies, window, 0.0 if args.headless else args.pause, engine)
        game = play_games(cell, engine, human, config, args.log_dir)
        if args.shot:
            window.save(args.shot)

    tracked = game.tracker
    in_sync = tracked is not None and tracked.board.fen() == rig.logical.fen() and tracked.physical == rig.physical
    print(f"\n{rig.logical.ply()} plies in {time.perf_counter() - started:.0f} s, "
          f"{'game over: ' + rig.logical.result() if rig.logical.is_game_over() else 'stopped'}")
    print(f"tracked position and pieces equal the table: {in_sync}")
    print(f"grips beside a piece: {desk_robot.hands.misses} (Willy's move finished by hand: {human.by_hand}), "
          f"human moves not recognised at once: {human.flags}, typed: {human.typed}, "
          f"Willy moves not confirmed at once: {human.disagreements}")
    for missed in dict.fromkeys(s.split(". Please")[0] for s in human.statuses if "grasp refused" in s):
        print("  " + missed)
    if not in_sync:
        said = list(dict.fromkeys(s for s in human.statuses if s))
        print("last things the game said:\n  " + "\n  ".join(said[-6:]))
    return 0 if in_sync else 1


if __name__ == "__main__":
    sys.exit(main())
