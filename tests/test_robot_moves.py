"""Move -> transfers, graveyard slots, jaw direction and the physical pieces after special moves.

    cd tests
    python -m unittest test_robot_moves -v
"""
import sys
import unittest
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.game.config import GraveyardConfig  # noqa: E402
from src.game.robot_moves import closing_directions, graveyard_slot, plan_transfers  # noqa: E402
from src.game.tracker import physical_after  # noqa: E402
from src.mapping.geometry import square_centres  # noqa: E402

S = 0.05
GRAVE = GraveyardConfig(side="+x", distance_squares=1.5, columns=2, table_offset_m=-0.012)


def plan(fen: str, uci: str, graveyard_index: int = 0):
    board = chess.Board(fen)
    physical = {s: p.symbol() for s, p in board.piece_map().items()}
    return plan_transfers(board, physical, chess.Move.from_uci(uci), {}, S, graveyard_index, GRAVE)


class PlanTransfersTest(unittest.TestCase):
    def test_quiet_move_is_one_transfer(self):
        (t,) = plan(chess.STARTING_FEN, "g1f3")
        self.assertEqual((t.symbol, t.from_square, t.to_square), ("N", chess.G1, chess.F3))
        np.testing.assert_allclose(t.to_xy, square_centres(S)[chess.F3])

    def test_capture_clears_the_square_first(self):
        captured, mover = plan("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", "e4d5", graveyard_index=3)
        self.assertEqual((captured.symbol, captured.from_square, captured.to_square), ("p", chess.D5, None))
        np.testing.assert_allclose(captured.to_xy, graveyard_slot(3, S, GRAVE))
        self.assertAlmostEqual(captured.to_z, -0.012)
        self.assertEqual((mover.symbol, mover.from_square, mover.to_square), ("P", chess.E4, chess.D5))

    def test_en_passant_takes_the_pawn_beside(self):
        captured, mover = plan("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6")
        self.assertEqual(captured.from_square, chess.D5)
        self.assertEqual(mover.to_square, chess.D6)

    def test_castling_moves_king_then_rook(self):
        king, rook = plan("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1")
        self.assertEqual((king.from_square, king.to_square), (chess.E1, chess.G1))
        self.assertEqual((rook.from_square, rook.to_square), (chess.H1, chess.F1))
        king, rook = plan("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "e8c8")
        self.assertEqual((rook.from_square, rook.to_square), (chess.A8, chess.D8))

    def test_graveyard_slots_lie_beside_the_board_in_columns_of_eight(self):
        slots = np.array([graveyard_slot(i, S, GRAVE) for i in range(16)])
        self.assertTrue(np.all(slots[:, 0] >= 5.5 * S - 1e-9))
        self.assertEqual(len({tuple(np.round(p, 6)) for p in slots}), 16)
        black_side = graveyard_slot(0, S, GraveyardConfig(side="-y"))
        self.assertLess(black_side[1], -4 * S)


class PhysicalPiecesTest(unittest.TestCase):
    def test_pawn_stays_after_promotion_without_spare_queen(self):
        board = chess.Board("4k3/1P6/8/8/8/8/8/4K3 w - - 0 1")
        physical = {s: p.symbol() for s, p in board.piece_map().items()}
        move = chess.Move.from_uci("b7b8q")
        self.assertEqual(physical_after(board, physical, move, "pawn_stays")[chess.B8], "P")
        self.assertEqual(physical_after(board, physical, move, "piece_swapped")[chess.B8], "Q")

    def test_castling_moves_the_physical_rook(self):
        board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
        physical = {s: p.symbol() for s, p in board.piece_map().items()}
        after = physical_after(board, physical, chess.Move.from_uci("e1c1"), "pawn_stays")
        self.assertEqual((after.get(chess.C1), after.get(chess.D1), after.get(chess.A1)), ("K", "R", None))


class ClosingDirectionTest(unittest.TestCase):
    def test_jaws_close_where_the_neighbours_leave_room(self):
        # neighbours left and right on the rank: the jaws should not open along the rank
        xy = np.zeros(2)
        others = [(np.array([S, 0.0]), 0.015), (np.array([-S, 0.0]), 0.015)]
        best = closing_directions(xy, 0.02, others)[0]
        self.assertLess(abs(best[0]), 0.8)


if __name__ == "__main__":
    unittest.main()
