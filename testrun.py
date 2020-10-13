# -*- coding: utf-8 -*-
# %matplotlib inline
%load_ext autoreload
%autoreload 2

import os
# os.chdir('DeepReinforcementLearning/')
os.getcwd()

import numpy as np
np.set_printoptions(suppress=True)

from shutil import copyfile
import random
from importlib import reload

from keras.utils import plot_model

from game import Game, GameState
from agent import Agent
from memory import Memory
from model import Residual_CNN
from funcs import playMatches, playMatchesBetweenVersions

import loggers as lg

from settings import run_folder, run_archive_folder
import initialise
import pickle

import config
import chess
import chess.engine
import time
import multiprocessing
from selfplay import _selfplay

lg.logger_main.info('=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*')
lg.logger_main.info('=*=*=*=*=*=.      NEW LOG      =*=*=*=*=*')
lg.logger_main.info('=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*')


env = Game()
env.gameState
env.identities(env.gameState, env.gameState.board)

t0 = time.perf_counter()

t0
t1 = time.perf_counter() - t0
t1

500 in env.gameState.allowedActions
board = chess.Board()
board

engine = chess.engine.SimpleEngine.popen_uci("C:\WinBoard-4.8.0\Stockfish\stockfish_20090216_x64_bmi2.exe")

board.can_claim_draw()
board.is_check()
gstate = GameState(board, -1)

gstate.allowedActions
best_player.get_preds(gstate)

gstate.allowedActions
board.reset()

board.fen()
board.board_fen()
board.set_fen('6r1/5k2/8/n7/3K4/8/8/8 b - - 0 136')

board.can_claim_draw()
board.is_checkmate()
board.can_claim_fifty_moves()
board.is_insufficient_material()

result = engine.play(board, chess.engine.Limit(time=0.01))
result.ponder
result.move
info = engine.analyse(board,chess.engine.Limit(depth=8))
info['score'].white()<chess.engine.Cp(-800)
info['score'].relative<chess.engine.Cp(-800)
info['score'].white()
info['score'].black()<chess.engine.Cp(-800)
info['score'].black()
print(info['score'])
board.can_claim_threefold_repetition()
board.is_game_over(claim_draw=True)
board

statefen = board.fen()
statefen = statefen.split(' ')
id = '{0} {1}'.format(statefen[0],statefen[1])
id
board.can_claim_draw()
board.can_claim_threefold_repetition()
board.can_claim_fifty_moves()
board


board.fen()
board.board_fen()
board.set_board_fen('1p6/2P4P/8/8/8/8/2p5/8')
board.set_fen('1p1p4/2P4P/8/8/8/8/2p5/1P1P4 b')
board.set_fen('2r2k2/6b1/1p6/pn2p1R1/P3p2B/2NP3P/4Kn2/8 b - - 3 52')
board.legal_moves
lmvs = {}
for move in board.legal_moves:
	if move.promotion == None or move.promotion == 5:
		entry = move.from_square * 73 + move.to_square
	else:
		diff = abs(move.from_square-move.to_square) - 8
		if diff < 0:
			diff = 1
		elif diff > 0:
			diff = 2
		entry = move.from_square*73 + 63 + diff*3 + (move.promotion-1)
	lmvs[entry] = move
	print('{0} - {1}, {2}, {3}, {4}'.format(move, move.from_square,move.to_square, move.promotion, move.uci()))
	print(entry)


chess.piece_name(4)


chessenv = Game()
memory = Memory(config.MEMORY_SIZE)
memory.commit_stmemory(env.identities,env.gameState,env.actionSpace)
memory.stmemory


pool = multiprocessing.Pool(2)
out = zip(pool.map(_selfplay, range(0, 2)))
t = tuple(out)

len(t)

chessenv.action_size
chessenv.state_size
chessenv.grid_shape
current_NN = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, (119,)+chessenv.grid_shape, chessenv.action_size, config.HIDDEN_CNN_LAYERS)
best_NN = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, (119,)+chessenv.grid_shape, chessenv.action_size, config.HIDDEN_CNN_LAYERS)
best_NN.model.set_weights(current_NN.model.get_weights())

best_player_version  = 2
print('LOADING MODEL VERSION ' + str(2) + '...')
m_tmp = best_NN.read(chessenv.name, 2, best_player_version)
current_NN.model.set_weights(m_tmp.get_weights())
best_NN.model.set_weights(m_tmp.get_weights())

current_player = Agent('current_player', chessenv.state_size, chessenv.action_size, config.MCTS_SIMS, config.CPUCT, current_NN)
best_player = Agent('best_player', chessenv.state_size, chessenv.action_size, config.MCTS_SIMS, config.CPUCT, best_NN)


state = chessenv.reset()
state.render(None)
# action, pi, MCTS_value, NN_value = current_player.act(state, 0)

scores, memory, _, _ = playMatches(best_player, best_player, config.EPISODES, lg.logger_main, turns_until_tau0 = config.TURNS_UNTIL_TAU0, memory = memory)

current_player.replay(memory.ltmemory)

current_NN.write(chessenv.name, 3)

ltmem = memory.ltmemory
memory_samp = random.sample(memory.ltmemory, min(1000, len(memory.ltmemory)))
len(memory_samp)
for s in memory_samp:
    current_value, current_probs, _ = current_player.get_preds(s['state'])
    best_value, best_probs, _ = best_player.get_preds(s['state'])

    lg.logger_memory.info('MCTS VALUE FOR %s: %f', s['playerTurn'], s['value'])
    lg.logger_memory.info('CUR PRED VALUE FOR %s: %f', s['playerTurn'], current_value)
    lg.logger_memory.info('BES PRED VALUE FOR %s: %f', s['playerTurn'], best_value)
    lg.logger_memory.info('THE MCTS ACTION VALUES: %s', ['%.2f' % elem for elem in s['AV']]  )
    lg.logger_memory.info('CUR PRED ACTION VALUES: %s', ['%.2f' % elem for elem in  current_probs])
    lg.logger_memory.info('BES PRED ACTION VALUES: %s', ['%.2f' % elem for elem in  best_probs])
    lg.logger_memory.info('ID: %s', s['state'].id)
    # lg.logger_memory.info('INPUT TO MODEL: %s', current_player.model.convertToModelInput(s['state']))

    s['state'].render(lg.logger_memory)

len(memory.ltmemory)
memory = None
scores
scores, _, points, sp_scores = playMatches(best_player, current_player, config.EVAL_EPISODES, lg.logger_tourney, turns_until_tau0 = 0, memory = None)
best_player_version = 2
best_NN.model.set_weights(current_NN.model.get_weights())



actions = np.argwhere(memory.ltmemory[8]['AV'] == max(memory.ltmemory[8]['AV']))
actions = np.random.multinomial(1,memory.ltmemory[8]['AV'])
actions
np.where(actions==1)

random.choice(actions)[0]
memory.ltmemory[8]['AV'][370:390]
memory.ltmemory[0]['board']=None
memory.ltmemory['state'].engine=None
memory.clear_stmemory()best_player_version = best_player_version + 1
best_NN.model.set_weights(current_NN.model.get_weights())
len(memory.ltmemory)


pickle.dump(memory, open( run_folder + "memory/memory" + str(4).zfill(4) + ".p", "wb" ) )

memory.ltmemory[1]
mem = Memory(config.MEMORY_SIZE)

for m in memory.ltmemory:
	mem.commit_stmemory(m)

memory = pickle.load(open( run_folder + "memory/multiproc/memory" + str(2).zfill(4) + ".p",   "rb" ) )
len(memory[0].ltmemory)
memory[0].ltmemory[32]
memory[0].ltmemory[32]['state'].movetable
memory[0].ltmemory[32]['state'].board

sum(memory.ltmemory[9]['AV']>0)
memory.ltmemory[9]['state'].movetable
memory = mem2

minibatch = random.sample(memory.ltmemory, min(config.BATCH_SIZE, len(memory.ltmemory)))
len(minibatch)
training_states = np.array([current_NN.convertToModelInput(row['state']) for row in minibatch])
training_targets = {'value_head': np.array([row['value'] for row in minibatch])
					, 'policy_head': np.array([row['AV'] for row in minibatch])}

training_states.shape
minibatch[2]['state'].board
minibatch[2]['state'].playerTurn
training_states[0,]
np.max(training_targets['policy_head'][125])
testdic = {'a':1,'b':2}

current_player.mcts.root.edges

chessenv.gameState.board


memory

env = Game()
memory = Memory(config.MEMORY_SIZE)
current_NN = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, (2,) + env.grid_shape,   env.action_size, config.HIDDEN_CNN_LAYERS)
best_NN = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, (2,) +  env.grid_shape,   env.action_size, config.HIDDEN_CNN_LAYERS)
best_player_version = 0
best_NN.model.set_weights(current_NN.model.get_weights())

plot_model(current_NN.model, to_file=run_folder + 'models/model.png', show_shapes = True)

current_player = Agent('current_player', env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT, current_NN)
best_player = Agent('best_player', env.state_size, env.action_size, config.MCTS_SIMS, config.CPUCT, best_NN)

players = {1:{"agent": current_player, "name":current_player.name}
		, -1: {"agent": best_player, "name":best_player.name}
		}

state = env.reset()
action, pi, MCTS_value, NN_value = current_player.act(state, 0)
current_player.mcts.root.edges
state, val, don = state.takeAction(38)

state.render(lg.logger_main)


(state._binary())

action, pi, MCTS_value, NN_value = players[state.playerTurn]['agent'].act(state, 1)

memory.commit_stmemory(env.identities, state, pi)

state, value, done, _ = env.step(action)
state.render(lg.logger_main)

action, pi, MCTS_value, NN_value = players[state.playerTurn]['agent'].act(state, 1)

memory.commit_stmemory(env.identities, state, pi)
state, value, done, _ = env.step(action)
memory.stmemory

memory.commit_ltmemory()
minibatch[0]['AV']
minibatch = random.sample(memory.ltmemory, min(2, len(memory.ltmemory)))
current_player.model.convertToModelInput(minibatch[0]['state'])
minibatch[0]
env.state_size
env.grid_shape
env.action_size
(2,)+env.grid_shape
tstrnn=Residual_CNN(config.REG_CONST, config.LEARNING_RATE, (2,) + env.grid_shape,   env.action_size, config.HIDDEN_CNN_LAYERS)
inp = best_player.model.convertToModelInput(env.gameState)
inp
tstrnn.predict(np.array([inp]))
env.actionSpace.shape


bitboard.shape

binary = np.reshape(bitboard,(12*8*8))

board = chess.Board()
board.board_fen
board.push_san('e4')
board.board_fen
board
board
board.fen()
board.set_board_fen('rnbqkbnr/pppppppp/8/8/4P3/3P4/PPP2PPP/RNBQKBNR')
board.push_san('e4')
board.push_san('Nf6')
board.fen()
board


board.legal_moves
board2 = board.copy()
board2.push_san('Rg8')

board2
board.board_fen()

board

bitbord = getBitBoard(board)

bitbord[0,]
piecetypedic = {'P':0,'N':1,'B':2,'R':3,'Q':4,'K':5,'p':6, 'n':7, 'b':8, 'r':9,'q':10,'k':11}

bitboard = getbitboard(board)
bitboard[0,]
np.fliplr(np.flipud(bitboard[0,]))

board.board_fen()

np.mod(15,7)
((15/8)-np.floor(15/8))*8
np.floor(16/8)
np.floor(0.875*8)
piece

list(board.piece_map().keys())[list(board.piece_map().values()).index(piece)]
np.mod(65,8)
board.piece_map()

piece = board.piece_map()[4]
piece.piece_type
piece.color != chess.WHITE
piece.symbol()
board.piece_map().
chess.PAWN
hm = board.pieces(1,1)
type(hm)
for move in board.legal_moves:
	print(board.san(move))

from keras import backend as K
K.tensorflow_backend._get_available_gpus()
