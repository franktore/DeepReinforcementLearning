import numpy as np
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

board = chess.Board()
info = engine.analyse(board, chess.engine.Limit(depth=16))
info['score']
info
board.set_fen('rnbqkbnr/p2p2pp/2p5/4pp2/p7/NP2P1P1/P1PP1P1P/R1B1KBNR w KQkq - 0 8')
board.is_stalemate()
board.can_claim_draw()
board.is_game_over(claim_draw=False)
board.is_checkmate()
board


3928 % 73
np.floor(3928/73)
3693 % 73
np.floor(3693/73)
3397 % 73
np.floor(3397/73)
3461 % 73
np.floor(3461/73)
4360 % 73
np.floor(4360/73)
3189 % 73
np.floor(3189/73)
2893 % 73
np.floor(2893/73)
2237 % 73
np.floor(2237/73)
2020-10-08 04:41:58,152 INFO action with highest Q + U...3928
2020-10-08 04:41:58,158 INFO PLAYER TURN...-1
2020-10-08 04:41:58,158 INFO action with highest Q + U...3693
2020-10-08 04:41:58,164 INFO PLAYER TURN...1
2020-10-08 04:41:58,165 INFO action with highest Q + U...3397
2020-10-08 04:41:58,171 INFO PLAYER TURN...-1
2020-10-08 04:41:58,171 INFO action with highest Q + U...3461
2020-10-08 04:41:58,177 INFO PLAYER TURN...1
2020-10-08 04:41:58,177 INFO action with highest Q + U...4360
2020-10-08 04:41:58,182 INFO PLAYER TURN...-1
2020-10-08 04:41:58,183 INFO action with highest Q + U...3189
2020-10-08 04:41:58,188 INFO PLAYER TURN...1
2020-10-08 04:41:58,188 INFO action with highest Q + U...2893
2020-10-08 04:41:58,194 INFO PLAYER TURN...-1
2020-10-08 04:41:58,194 INFO action with highest Q + U...2237
2020-10-08 04:41:58,200 INFO PLAYER TURN...1
2020-10-08 04:41:58,200 INFO action with highest Q + U...3928




# board = chess.Board()
#
# with chess.polyglot.open_reader("C:\WinBoard-4.8.0\WinBoard\default_book.bin") as reader:
#     for entry in reader.find_all(board):
#         print(entry.move, entry.weight, entry.learn)
