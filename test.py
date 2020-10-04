import time
import asyncio
import chess
import chess.engine
import chess.polyglot

async def main() -> None:
    transport, engine = await chess.engine.popen_uci("C:\WinBoard-4.8.0\Stockfish\stockfish_20090216_x64_bmi2.exe")

    board = chess.Board()
    while not board.is_game_over():
        result = await engine.play(board, chess.engine.Limit(time=0.1))
        board.push(result.move)
        print(board.fen())

    info = await engine.analyse(board, chess.engine.Limit(time=0.1))
    print(info["score"])
    # Score: +20

    # board = chess.Board("r1bqkbnr/p1pp1ppp/1pn5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 2 4")
    # info = await engine.analyse(board, chess.engine.Limit(depth=20))
    # print(info["score"])
    # Score: #+1

    time.sleep(5)
    await engine.quit()

asyncio.set_event_loop_policy(chess.engine.EventLoopPolicy())
asyncio.run(main())

engine = chess.engine.SimpleEngine.popen_uci("C:\WinBoard-4.8.0\Stockfish\stockfish_20090216_x64_bmi2.exe")
result = engine.play(board, chess.engine.Limit(time=0.1))
result.move
result.move.uci()
board.push(result.move)

# board = chess.Board()
#
# with chess.polyglot.open_reader("C:\WinBoard-4.8.0\WinBoard\default_book.bin") as reader:
#     for entry in reader.find_all(board):
#         print(entry.move, entry.weight, entry.learn)

board = chess.Board()
board.set_fen('8/5p1k/6pP/8/4P1qP/8/r4P2/1r3K2 w - - 7 49')
board.is_stalemate()
board
