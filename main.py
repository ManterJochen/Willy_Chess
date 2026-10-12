"""
Willy_Chess: play chess against Willy.

    python main.py --willy-data <Willy config directory of the cell> [--config config/willy_chess.yaml]

The live view of the board detection alone: python -m src.board.demo
"""
import sys

from src.game.play import main

if __name__ == "__main__":
    sys.exit(main())
