import numpy as np
import chess as chess
import chess.engine
import logging

class Game:
	def __init__(self):
		self.currentPlayer = 1
		self.board = chess.Board()
		self.engine = chess.engine.SimpleEngine.popen_uci("C:\WinBoard-4.8.0\Stockfish\stockfish_20090216_x64_bmi2.exe")
		disableresignation = ((np.random.randn(1)+1)/2)[0]<0.1

		movtab = {}
		hashtab = {}
		movtab[0] = self.board.fen()
		position = self.board.fen()
		position = position.split(' ')
		hash = '{0}'.format(position[0])
		hashtab[hash] = 1
		self.gameState = GameState(self.board, self.currentPlayer, movtab, hashtab, 1, self.engine, disableresignation)

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
		disableresignation = ((np.random.randn(1)+1)/2)[0]<0.1
		print('disable resignation {0}'.format(disableresignation))
		self.board.reset()
		movtab = {}
		hashtab = {}
		movtab[0] = self.board.fen()
		position = self.board.fen()
		position = position.split(' ')
		hash = '{0}'.format(position[0])
		hashtab[hash] = 1
		self.gameState = GameState(self.board, self.currentPlayer, movtab, hashtab, 1, self.engine, disableresignation)
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
	def __init__(self, board, playerTurn, movetable, hashtable, repetitions, engine=None, disableresignation=False):
		self.T = 8
		self.board = board
		self.engine = engine
		self.pieces = {'1':'White', '0': '-', '-1':'Black'}
		self.playerTurn = playerTurn
		self.disableresignation = disableresignation
		self.movetable = movetable
		self.hashtable = hashtable
		self.repetitions = repetitions
		self.binary = self._binary()
		self.id = self._convertStateToId()
		self.allowedActions, self.legalMoves, self.moveToAction = self._allowedActions()
		self.resign = False
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
		planes = np.zeros((119,8,8))
		t = len(self.movetable)-1
		for i in range(0,self.T):
			if (t-i) >= 0:
				movestr = self.movetable[t-i]
				bitboard = self.getBitBoard(movestr)
				planes[i*14:(i+1)*14,] = bitboard
		planes[112,] = np.ones((8,8))*self.playerTurn
		state = self.movetable[t].split(' ')
		nmoves = int(state[5])
		planes[113,0,0] = nmoves
		whiteKingCastle = int('K' in state[2])
		whiteQueenCastle = int('Q' in state[2])
		blackKingCastle = int('k' in state[2])
		blackQueenCastle = int('q' in state[2])
		planes[114,] = np.ones((8,8))*whiteKingCastle
		planes[115,] = np.ones((8,8))*whiteQueenCastle
		planes[116,] = np.ones((8,8))*blackKingCastle
		planes[117,] = np.ones((8,8))*blackQueenCastle
		noprogress = int(state[4])
		planes[118,0,0] = noprogress
		binary = np.reshape(planes,(119*8*8))
		return binary

	def _convertStateToId(self):
		position = self.board.fen()
		position = position.split(' ')
		id = '{0} {1} {2} {3} {4}'.format(position[0],position[1],position[2],position[3],position[4])
#		id = position
		return id

	def _checkForEndGame(self):
		endgame = self.board.is_game_over(claim_draw=True)
		if not endgame and self.engine != None:
			info = self.engine.analyse(self.board, chess.engine.Limit(depth=8))
			if self.playerTurn < 0:
				score = info['score'].black()
			else:
				score = info['score'].white()
			if score < -chess.engine.Cp(800):
				# to allow 10% false positives we randomly disable resignation
				if not self.disableresignation:
					#print('{0} resigning at score {1}'.format(self.pieces[str(self.playerTurn)], score))
					self.resign = True
					endgame = True
		return endgame

	def _getValue(self):
		if self.board.is_checkmate() or self.resign:
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
			movtab = self.movetable.copy()
			hashtab = self.hashtable.copy()
			repetitions = self.repetitions
			movtab[len(movtab)] = newboard.fen()
			position = newboard.fen()
			position = position.split(' ')
			hash = '{0}'.format(position[0])
			if hash not in hashtab:
				hashtab[hash] = 1
			else:
				hashtab[hash] += 1
			if hashtab[hash] > self.repetitions:
				repetitions = hashtab[hash]
			newState = GameState(newboard, -self.playerTurn, movtab, hashtab, repetitions, self.engine, self.disableresignation)
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
		action = 1
		if self.engine != None:
		#engine = chess.engine.SimpleEngine.popen_uci("C:\WinBoard-4.8.0\Stockfish\stockfish_20090216_x64_bmi2.exe")
			result = self.engine.play(self.board, chess.engine.Limit(time=0.01))
			if result != None and result.move != None:
				action = self.moveToAction[result.move.uci()]
#		engine.quit()

		return action

	def render(self,logger):
		if logger==None:
			print(self.board.fen())
		else:
			logger.info(self.board.fen())
			logger.info('---------------')

	def getBitBoard(self, movestr):
		board = chess.Board()
		board.set_fen(movestr)
		position = movestr.split(' ')
		player = self.playerTurn
		# if position[1] == 'w':
		# 	player = 1
		# else:
		# 	player = -1
		hash = '{0}'.format(position[0])
		reps = self.hashtable[hash]
		lines = position[0].split('/')
		bitboard = np.zeros((14,8,8))
		for square in board.piece_map():
			piece = board.piece_map()[square]
			entry = piece.piece_type - 1;
			if player==1 and  piece.color != chess.WHITE:
				entry += 6
			elif player==-1 and piece.color != chess.BLACK:
				entry += 6

			row = np.floor(square/8)
			col = 8*((square/8)-row)
			bitboard[int(entry),int(row),int(col)] = 1
		if reps>1:
			bitboard[12,]= np.ones((8,8))
			if reps>2:
				bitboard[13,]= np.ones((8,8))
		return bitboard
