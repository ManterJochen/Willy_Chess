"""
A chess move as robot motions: which piece goes from where to where, and how the hand grips it.

Planning is pure geometry in the board frame and runs without a robot. Execution hands every
pick and place to Workaholic-Willy (robot.pick / robot.place), which plans, guards and checks
the hold; a refusal stops the move with the piece where it is, or still in the hand.
"""
from dataclasses import dataclass, field
from typing import Any

import chess
import numpy as np

from src.game.config import GraveyardConfig, RobotConfig
from src.mapping.chessboard_mapping_3d import BoardObservation, ChessBoardMapping3D
from src.mapping.geometry import square_centres

PIECE_NAMES = {"p": "pawn", "n": "knight", "b": "bishop", "r": "rook", "q": "queen", "k": "king"}
# Why a motion needs no camera world, for a cell that plans without one (Willy asks every motion for it).
DECLINE = "chess: piece positions from ChessBoardMapping3D, no camera world"
# Closing directions on the board: along the files, along the ranks and the two diagonals.
_DIRECTIONS = [np.array(d, float) / np.linalg.norm(d) for d in ((1, 0), (0, 1), (1, 1), (1, -1))]


@dataclass(frozen=True)
class JawGeometry:
    """
    The parallel jaw as far as neighbouring pieces care, in metres. The defaults are a Robotiq Hand-E.

    Attributes:
        finger_thickness (float): Finger extent along the closing direction.
        finger_width (float): Finger extent across it.
        max_open (float): Widest opening between the jaw faces.
    """
    finger_thickness: float = 0.0108
    finger_width: float = 0.0292
    max_open: float = 0.050

    @classmethod
    def from_tree(cls, tree) -> "JawGeometry":
        """The cell's hand from Willy's config tree (robot.grasping.gripper_geometry, robot.gripper)."""
        jaw = tree.robot.grasping.gripper_geometry.parallel_jaw
        return cls(jaw.finger_thickness_mm / 1000.0, jaw.finger_width_mm / 1000.0, tree.robot.gripper.max_width_mm / 1000.0)


@dataclass
class Transfer:
    """
    Carry one piece.

    Attributes:
        symbol (str): FEN symbol of the physical piece.
        from_xy (np.ndarray): Where it stands, board frame, metres (the observed position).
        to_xy (np.ndarray): Where it goes, board frame, metres (square centre or graveyard slot).
        from_z (float): Height of the surface it stands on, board frame, metres.
        to_z (float): Height of the surface it is set on.
        from_square (int | None): Its square, None if it does not stand on the board.
        to_square (int | None): The target square, None for the graveyard.
    """
    symbol: str
    from_xy: np.ndarray
    to_xy: np.ndarray
    from_z: float = 0.0
    to_z: float = 0.0
    from_square: int | None = None
    to_square: int | None = None

    def describe(self) -> str:
        name = PIECE_NAMES[self.symbol.lower()]
        start = chess.square_name(self.from_square) if self.from_square is not None else "graveyard"
        end = chess.square_name(self.to_square) if self.to_square is not None else "graveyard"
        return f"{name} {start} -> {end}"


def graveyard_slot(index: int, square: float, config: GraveyardConfig) -> np.ndarray:
    """
    Board-frame position of graveyard slot `index` beside the board: columns of 8 slots parallel to
    the board edge, the first one `distance_squares` from the edge of the 8x8 squares.
    """
    column, row = divmod(index, 8)
    out = (4 + config.distance_squares + column) * square
    along = (row - 3.5) * square
    axis, sign = config.side[1], 1.0 if config.side[0] == "+" else -1.0
    return np.array([sign * out, along]) if axis == "x" else np.array([along, sign * out])


def plan_transfers(board: chess.Board, physical: dict[int, str], move: chess.Move, piece_xy: dict[int, np.ndarray],
                   square: float, graveyard_index: int, graveyard: GraveyardConfig) -> list[Transfer]:
    """
    The transfers of one move, in order: a captured piece to the graveyard first, then the moving piece,
    for castling the rook last.

    Args:
        board (chess.Board): Logical position before the move.
        physical (dict[int, str]): Square -> physical piece symbol before the move.
        move (chess.Move): A legal move.
        piece_xy (dict[int, np.ndarray]): Square -> observed board position of the piece on it; squares
            missing here use their centre.
        square (float): Square size in metres.
        graveyard_index (int): Next free graveyard slot.
        graveyard (GraveyardConfig): Where the graveyard is.
    """
    centres = square_centres(square)

    def at(sq: int) -> np.ndarray:
        return piece_xy.get(sq, centres[sq])

    transfers = []
    captured = None
    if board.is_en_passant(move):
        captured = move.to_square + (-8 if board.turn == chess.WHITE else 8)
    elif board.piece_at(move.to_square) is not None:
        captured = move.to_square
    if captured is not None:
        transfers.append(Transfer(physical[captured], at(captured), graveyard_slot(graveyard_index, square, graveyard),
                                  0.0, graveyard.table_offset_m, captured, None))
    transfers.append(Transfer(physical[move.from_square], at(move.from_square), centres[move.to_square],
                              0.0, 0.0, move.from_square, move.to_square))
    if board.is_castling(move):
        rank = chess.square_rank(move.from_square)
        kingside = board.is_kingside_castling(move)
        rook_from, rook_to = chess.square(7 if kingside else 0, rank), chess.square(5 if kingside else 3, rank)
        transfers.append(Transfer(physical[rook_from], at(rook_from), centres[rook_to], 0.0, 0.0, rook_from, rook_to))
    return transfers


def closing_directions(xy: np.ndarray, open_half_width: float, others: list[tuple[np.ndarray, float]],
                       jaw: JawGeometry = JawGeometry()) -> list[np.ndarray]:
    """
    Closing directions for the jaws on the board, most room first.

    Each open finger is a box outside its jaw face (open_half_width from the piece axis): finger_thickness
    along the closing direction, finger_width across it. Its room is the distance from that box to the
    nearest other piece, minus that piece's foot radius.

    Args:
        xy (np.ndarray): Axis of the piece to grip, board frame.
        open_half_width (float): Distance of each open jaw face from the axis, metres.
        others (list[tuple[np.ndarray, float]]): (position, foot radius) of every other piece.
        jaw (JawGeometry): The fingers.
    """
    half = np.array([jaw.finger_thickness / 2, jaw.finger_width / 2])

    def room(d: np.ndarray) -> float:
        across = np.array([-d[1], d[0]])
        gaps = []
        for side in (1.0, -1.0):
            centre = xy + side * (open_half_width + half[0]) * d
            for p, r in others:
                local = np.abs([(p - centre) @ d, (p - centre) @ across])
                gaps.append(np.linalg.norm(np.maximum(local - half, 0.0)) - r)
        return min(gaps, default=np.inf)

    return sorted(_DIRECTIONS, key=room, reverse=True)


@dataclass
class MoveExecution:
    """What the robot did with one move: done, or which transfer stopped and why."""
    ok: bool
    done: list[Transfer] = field(default_factory=list)
    failed: Transfer | None = None
    holding: bool = False
    reports: list[Any] = field(default_factory=list)
    message: str = ""


class RobotMover:
    """
    Execute transfers with a Workaholic-Willy robot.

    Args:
        robot: A connected willy.Robot.
        mapping (ChessBoardMapping3D): The calibrated board.
        config (RobotConfig): Grasp heights, clearance, standoff.
        jaw (JawGeometry): The cell's fingers (JawGeometry.from_tree).
        camera_world (bool): The arm plans around what the cameras see (CameraWorldPlan): picks then hand
            Willy the piece to grip as a keep-out, otherwise every motion declines the camera world.
    """

    def __init__(self, robot: Any, mapping: ChessBoardMapping3D, config: RobotConfig,
                 jaw: JawGeometry = JawGeometry(), camera_world: bool = False):
        self.robot = robot
        self.mapping = mapping
        self.config = config
        self.jaw = jaw
        self.camera_world = camera_world

    def execute(self, transfers: list[Transfer], observation: BoardObservation) -> MoveExecution:
        """Run the transfers in order; stop at the first one that does not finish."""
        result = MoveExecution(ok=True)
        for transfer in transfers:
            ok, holding, reports, message = self._transfer(transfer, observation)
            result.reports.extend(reports)
            if not ok:
                result.ok, result.failed, result.holding, result.message = False, transfer, holding, message
                return result
            result.done.append(transfer)
        return result

    def _transfer(self, tr: Transfer, observation: BoardObservation):
        calib = observation.calib
        height = calib.piece_dims(tr.symbol)[0]
        grasp = self.config.grasp_height_fraction[tr.symbol.lower()] * height
        width = self.mapping.grasp_width(tr.symbol)
        if width > self.jaw.max_open - 0.002:
            return False, False, [], f"{tr.describe()}: {width * 1000:.0f} mm wide, the gripper opens only {self.jaw.max_open * 1000:.0f} mm"
        open_mm = min(width * 1000 + 2 * self.config.finger_clearance_mm, self.jaw.max_open * 1000)
        others = [(xy, calib.piece_dims(observation.frame.classes[k])[1])
                  for k, xy in observation.grasp_positions.items() if np.linalg.norm(xy - tr.from_xy) > 0.5 * calib.square]
        keep_out = self._keep_out(tr, observation)

        reports = []
        for direction in closing_directions(tr.from_xy, open_mm / 2000.0, others, self.jaw):
            heading = self._heading(direction)
            pose = self.robot.tool_down(*self.mapping.board_to_base_mm(tr.from_xy, tr.from_z + grasp),
                                        closing_axis="x", yaw_deg=heading)
            picked = self.robot.pick(pose, width * 1000, standoff_mm=self.config.standoff_mm,
                                     pre_open_mm=open_mm, **keep_out)
            reports.append(picked)
            if picked.ok:
                break
            if not getattr(picked, "another_candidate_may_follow", False):
                return False, False, reports, f"{tr.describe()}: grasp refused ({picked.outcome})"
        else:
            return False, False, reports, f"{tr.describe()}: no grasp direction worked"

        z = tr.to_z + grasp + self.config.place_air_mm / 1000.0
        pose = self.robot.tool_down(*self.mapping.board_to_base_mm(tr.to_xy, z), closing_axis="x", yaw_deg=heading)
        placed = self.robot.place(pose, standoff_mm=self.config.standoff_mm, **self._decline())
        reports.append(placed)
        if not placed.ok:
            return False, True, reports, f"{tr.describe()}: placing not finished ({placed.outcome}), the piece may still be in the gripper"
        return True, False, reports, ""

    def release(self) -> Any:
        """Open the hand (after a place that did not finish, once a person holds the piece)."""
        return self.robot.release()

    def _heading(self, board_direction: np.ndarray) -> float:
        """Base-frame heading in degrees of a closing direction on the board, folded into (-90, 90]."""
        d = self.mapping.board_axis_base(board_direction)
        heading = float(np.degrees(np.arctan2(d[1], d[0])))
        while heading > 90.0:
            heading -= 180.0
        while heading <= -90.0:
            heading += 180.0
        return heading

    def _decline(self) -> dict:
        if self.camera_world:
            return {}
        return {"decline": DECLINE}

    def _keep_out(self, tr: Transfer, observation: BoardObservation) -> dict:
        """
        The piece to grip held out of the arm's live world (its depth points), every other piece in it
        as a named obstacle. Without a camera world: the decline instead.
        """
        if not self.camera_world:
            return self._decline()
        from src.robot.core.keep_out import SegmentationOffer  # Willy, imported only when it is used

        points = {}
        for k in observation.positions:
            P = observation.piece_points(k)
            if len(P):
                points[k] = self.mapping.board_points_to_base_mm(P)
        target = next((k for k, sq in observation.squares.items() if sq == tr.from_square), None)
        if target is None or target not in points:
            return {}
        named = tuple((observation.frame.classes[k], pts) for k, pts in points.items() if k != target)
        offer = SegmentationOffer(captured_at_s=observation.captured_at_s, target_points_base_mm=points[target],
                                  target_label=PIECE_NAMES[tr.symbol.lower()], named_points_base_mm=named)
        return {"keep_out": offer}
