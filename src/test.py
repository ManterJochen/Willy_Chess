import chess
import chess.engine

engine = chess.engine.SimpleEngine.popen_uci(
    "external/chess/windows/stockfish/stockfish-windows-x86-64-universal.exe"
)

board = chess.Board()

result = engine.play(
    board,
    chess.engine.Limit(time=1.0),
)

print("Willy:", result.move)

engine.quit()