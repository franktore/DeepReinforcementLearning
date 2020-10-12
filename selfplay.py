# -*- coding: utf-8 -*-
# %matplotlib inline
import os
# os.chdir('DeepReinforcementLearning/')
os.getcwd()

import numpy as np
np.set_printoptions(suppress=True)

from shutil import copyfile
import random
from importlib import reload
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
import time
import multiprocessing


def _selfplay(n):
    chessenv = Game()
    memory = Memory(config.MEMORY_SIZE)
    current_NN = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, (119,)+chessenv.grid_shape, chessenv.action_size, config.HIDDEN_CNN_LAYERS)
    best_NN = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, (119,)+chessenv.grid_shape, chessenv.action_size, config.HIDDEN_CNN_LAYERS)
    best_NN.model.set_weights(current_NN.model.get_weights())
    current_player = Agent('current_player', chessenv.state_size, chessenv.action_size, config.MCTS_SIMS, config.CPUCT, current_NN)
    best_player = Agent('best_player', chessenv.state_size, chessenv.action_size, config.MCTS_SIMS, config.CPUCT, best_NN)

    t0 = time.perf_counter()
    print ('Proc {0} start'.format(n))
    _, memory, _, _ = playMatches(best_player, best_player, config.EPISODES, lg.logger_main, turns_until_tau0 = config.TURNS_UNTIL_TAU0, memory = memory)
    t1 = time.perf_counter() - t0
    print ('Proc {0} done in {1} seconds'.format(n, t1))
    return memory

def main(n):
    print('multiproc start')
    pool = multiprocessing.Pool(4)
    out = zip(pool.map(_selfplay, range(0, n)))
    x = tuple(out)
    for i in range(0,len(x)):
        pickle.dump(x[i], open( run_folder + "memory/multiproc/memory" + str(i+1).zfill(4) + ".p", "wb" ) )
    print('multiproc done')


if __name__ == "__main__":
    main(4)
