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

lg.logger_main.info('=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*')
lg.logger_main.info('=*=*=*=*=*=.      NEW LOG      =*=*=*=*=*')
lg.logger_main.info('=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*')


env = Game()
env.gameState
env.identities(env.gameState, env.gameState.board)



500 in env.gameState.allowedActions
board = chess.Board()
board


board.can_claim_draw()
board.is_check()
gstate = GameState(board, -1)

gstate.allowedActions
best_player.get_preds(gstate)

gstate.allowedActions
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

chessenv.action_size
chessenv.state_size
chessenv.grid_shape

current_NN = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, (12,)+chessenv.grid_shape, chessenv.action_size, config.HIDDEN_CNN_LAYERS)
best_NN = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, (12,)+chessenv.grid_shape, chessenv.action_size, config.HIDDEN_CNN_LAYERS)
best_NN.model.set_weights(current_NN.model.get_weights())
best_player_version = 0
current_player = Agent('current_player', chessenv.state_size, chessenv.action_size, config.MCTS_SIMS, config.CPUCT, current_NN)
best_player = Agent('best_player', chessenv.state_size, chessenv.action_size, config.MCTS_SIMS, config.CPUCT, best_NN)

state = chessenv.reset()
state.render(None)
# action, pi, MCTS_value, NN_value = current_player.act(state, 0)

 _, memory, _, _ = playMatches(best_player, best_player, config.EPISODES, lg.logger_main, turns_until_tau0 = config.TURNS_UNTIL_TAU0, memory = memory)


len(memory.ltmemory)
np.sum((memory.ltmemory[1665]['AV'])>0)
pickle.dump( memory, open("run/memory/memory_chess" + ".p", "wb" ) )

mem2 = pickle.load(open("run/memory/memory_chess" + ".p",   "rb" ) )
len(mem2.ltmemory)
minibatch = random.sample(mem2.ltmemory, min(config.BATCH_SIZE, len(mem2.ltmemory)))
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
