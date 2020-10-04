import numpy as np
import chess as chess
import chess.engine
import logging

class Game:
	def __init__(self):
		self.currentPlayer = 1
		self.board = chess.Board()
		self.gameState = GameState(self.board, self.currentPlayer)

		# actionspace is the total number of possible moves at any given time.
		# it includes picking a random piece from the board (8x8) and moving it randomly according to queen rules (56 possibilities)
		# or according to horse moves (+8 possibilities) or underpromotions (+9 possibilities)
		self.actionSpace = np.zeros((8*8*73, 1), dtype=np.int)
		#self.pieces = {'1':'X', '0': '-', '-1':'O'}
		self.grid_shape = (8, 8)
		self.input_shape = (12, 8, 8)
		self.name = 'chess'
		self.state_size = len(self.gameState.binary)
		self.action_size = len(self.actionSpace)

	def reset(self):
		self.currentPlayer = 1
		self.gameState = GameState(self.board, self.currentPlayer)
		return self.gameState

	def step(self, action):
		next_state, value, done = self.gameState.takeAction(action)
		self.gameState = next_state
		self.currentPlayer = -self.currentPlayer
		info = None
		return ((next_state, value, done, info))

	def identities(self, state, actionValues):
		identities = [(state, actionValues)]

		# currentBoard = state.board
		# currentAV = actionValues

		# for i in range(6):
		# 	currentBoard[i,] = np.fliplr(np.flipud(currentBoard[i,]))
		# 	currentAV[i,] = np.fliplr(np.flipud(currentAV[i,]))

		# identities.append((GameState(currentBoard, state.playerTurn), currentAV))

		return identities

class GameState():
	def __init__(self, board, playerTurn):
		self.board = board
		self.pieces = {'1':'White', '0': '-', '-1':'Black'}
		self.playerTurn = playerTurn
		self.binary = self._binary()
		self.id = self._convertStateToId()
		self.allowedActions, self.legalMoves, self.moveToAction = self._allowedActions()
		self.isEndGame = self._checkForEndGame()
		self.value = self._getValue()
		self.score = self._getScore()

	def _allowedActions(self):
		allowed = []
		legalMoves = {}
		moveToAction = {}
		for move in self.board.legal_moves:
			if move.promotion == None or move.promotion == 5:
				entry = move.from_square * 73 + move.to_square
			else:
				diff = abs(move.from_square-move.to_square) - 8
				if diff < 0:
					diff = 1
				elif diff > 0:
					diff = 2
				entry = move.from_square*73 + 63 + diff*3 + (move.promotion-1)
			allowed.append(entry)
			legalMoves[entry] = move
			moveToAction[move.uci()] = entry

		return allowed, legalMoves, moveToAction

	def _binary(self):
		bitboard = self.getBitBoard(self.board)
		binary = np.reshape(bitboard,(12*8*8))
		return binary

	def _convertStateToId(self):
		return self.board.board_fen()

	def _checkForEndGame(self):
		fullmovecnt = int(self.board.fen().split(' ')[-1])
		halfmovecnt = int(self.board.fen().split(' ')[-2])
		endgame = self.board.is_checkmate() or \
				self.board.is_stalemate() or \
				self.board.is_insufficient_material() or \
				halfmovecnt>=50 or \
				fullmovecnt>=100 #self.board.can_claim_draw()
		return endgame

	def _getValue(self):
		if self.board.is_checkmate():
			return (-1, -1, 1)
		return (0, 0, 0)

	def _getScore(self):
		tmp = self.value
		return(tmp[1], tmp[2])

	def takeAction(self, action):
		if action in self.legalMoves:
			move = self.legalMoves[action]
			newboard = self.board.copy()
			newboard.push_uci(move.uci())
			newState = GameState(newboard, -self.playerTurn)
		else:
			print('Action {0} NOT in legalMoves Dictionary'.format(action))
			newState = self

		value = 0
		done = 0

		if newState.isEndGame:
			value = newState.value[0]
			done = 1

		return (newState, value, done)

	def expertChoice(self):
		engine = chess.engine.SimpleEngine.popen_uci("C:\WinBoard-4.8.0\Stockfish\stockfish_20090216_x64_bmi2.exe")
		result = engine.play(self.board, chess.engine.Limit(time=0.01))
		engine.quit()
		action = 1
		if result != None and result.move != None:
			action = self.moveToAction[result.move.uci()]

		return action

	def render(self,logger):
		if logger==None:
			print(self.board.fen())
		else:
			logger.info(self.board.fen())
			logger.info('---------------')

	def getBitBoard(self, board):
		pos = str(board.board_fen())
		lines = pos.split('/')
		bitboard = np.zeros((12,8,8))
		for square in board.piece_map():
			piece = board.piece_map()[square]
			entry = piece.piece_type - 1;
			if playerTurn==1 and  piece.color != chess.WHITE:
				entry += 6
			elif playerTurn==-1 and piece.color != chess.BLACK:
				entry += 6

			row = np.floor(square/8)
			col = 8*((square/8)-row)
			bitboard[int(entry),int(row),int(col)] = 1
		return bitboard
