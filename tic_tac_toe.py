# required libraries
import numpy as np
import pandas as pd
import random, pprint
from scipy.ndimage.interpolation import shift

# AI moduls
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD


# Tic Tac Toe Game Class
class TicTacToe(object):
    def __init__(self):
        # 1: player1, 0: player0, 2: avaible
        self.board = np.full((3,3),2)

    # decide who will start the game first
    def whoStart(self):
        turn = np.random.randint(0,2, size=1)
        if turn == 0:
            self.activePlayer = 0
        elif turn == 1:
            self. activePlayer = 1
        return self.activePlayer
    
    # make a move
    def move(self, player, coord):
        if self.board[coord] != 2 or self.gameStatus() != 'In Progress' or self.activePlayer != player:
            raise ValueError('invalid Move')
        self.board[coord] = player
        self.activePlayer = 1 - player
        return self.gameStatus(), self.board
    
    def gameStatus(self):
        # win - row check
        for i in range(self.board.shape[0]):
            if 2 not in self.board[i, :] and len(set(self.board[i, :])) == 1:
                return 'Win'
        
        # win - column check
        for j in range(self.board.shape[1]):
            if 2 not in self.board[:, j] and len(set(self.board[:, j ])) == 1:
                return 'Win'
            
        # win - diagonal check
        if 2 not in np.diag(self.board) and len(set(np.diag(self.board))) == 1:
            return('Win')
        if 2 not in np.diag(np.fliplr(self.board)) and len(set(np.diag(np.fliplr(self.board)))) == 1:
            return 'Win'
        
        # draw - check
        if 2 not in self.board:
            return 'Draw'
        # the game continues
        else:
            return 'In Progress'
        


# Neural Network Model
model = Sequential() 
model.add(Dense(18, input_dim=9, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(9, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1, kernel_initializer='normal'))

# SGD optimizer
learningRate = 0.001
momentum = 0.8
sgd = SGD(lr=learningRate, momentum=momentum, nesterov= False)

model.compile(loss='mean_squared_error', optimizer=sgd)

#model.summary()

# list of possible moves
def moveGenerator(currentBoardState, activePlayer):
    avaibleMoves = {}
    for i in range(currentBoardState.shape[0]):
        for j in range(currentBoardState.shape[1]):
            if currentBoardState[i, j] == 2:
                boardStateCopy = currentBoardState.copy()
                boardStateCopy[i ,j] = activePlayer
                avaibleMoves[(i, j)] = boardStateCopy.flatten()
    return avaibleMoves

# move selector
def moveSelector(model, currentBoardState, activePlayer):
    tracker = {}
    avaibleMoves = moveGenerator(currentBoardState, activePlayer)
    for moveCoord in avaibleMoves:
        score = model.predict(avaibleMoves[moveCoord].reshape(1,9))
        tracker[moveCoord] = score

    selectedMove = max(tracker, key = tracker.get)
    newBoardState = avaibleMoves[selectedMove]
    score = tracker[selectedMove]
    return selectedMove, newBoardState, score


# --------------- #
# competitive move selections
# Does the player win after 1 step?
# row based win control
def rowWinMoveCheck(currentBoardState, avaibleMoves, activePlayer):
    avaibleMoveCoords = list(avaibleMoves.keys())
    random.shuffle(avaibleMoveCoords)
    for coord in avaibleMoveCoords:
        currentBoardStateCopy = currentBoardState.copy()
        currentBoardStateCopy[coord] = activePlayer
        for i in range(currentBoardStateCopy.shape[0]):
            if 2 not in currentBoardStateCopy[i, :] and len(set(currentBoardStateCopy[i, :])) == 1:
                selectedMove = coord
                return selectedMove
            
# column based win control
def colWinMoveCheck(currentBoardState, avaibleMoves, activePlayer):
    avaibleMoveCoords = list(avaibleMoves.keys())
    random.shuffle(avaibleMoveCoords)
    for coord in avaibleMoveCoords:
        currentBoardStateCopy = currentBoardState.copy()
        currentBoardStateCopy[coord] = activePlayer
        for i in range(currentBoardStateCopy.shape[1]):
            if 2 not in currentBoardStateCopy[:, i] and len(set(currentBoardStateCopy[:, i])) == 1:
                selectedMove = coord
                return selectedMove

# diagonal win check
def diagWinMoveCheck(currentBoardState, avaibleMoves, activePlayer):
    avaibleMoveCoords = list(avaibleMoves.keys())
    random.shuffle(avaibleMoveCoords)
    for coord in avaibleMoveCoords:
        currentBoardStateCopy = currentBoardState.copy()
        currentBoardStateCopy[coord] = activePlayer
        if 2 not in np.diag(currentBoardStateCopy) and len(set(np.diag(currentBoardStateCopy))) == 1:
            selectedMove = coord
            return selectedMove

# diagonal (reverse) win check
def diag2WinMoveCheck(currentBoardState, avaibleMoves, activePlayer):
    avaibleMoveCoords = list(avaibleMoves.keys())
    random.shuffle(avaibleMoveCoords)
    for coord in avaibleMoveCoords:
        currentBoardStateCopy = currentBoardState.copy()
        currentBoardStateCopy[coord] = activePlayer
        if 2 not in np.diag(np.fliplr(currentBoardStateCopy)) and len(set(np.diag(np.fliplr(currentBoardStateCopy)))) == 1:
            selectedMove = coord
            return selectedMove

# block if opponent wins (row)
def rowBlockMoveCheck(currentBoardState, avaibleMoves, activePlayer):
    avaibleMoveCoords = list(avaibleMoves.keys())
    random.shuffle(avaibleMoveCoords)
    for coord in avaibleMoveCoords:
        currentBoardStateCopy = currentBoardState.copy()
        currentBoardStateCopy[coord] = activePlayer
        for i in range(currentBoardStateCopy.shape[0]):
            if 2 not in currentBoardStateCopy[i, :] and (currentBoardStateCopy[i, :] == 1).sum() == 2:
                if not(2 not in currentBoardStateCopy[i, :] and (currentBoardStateCopy[i, :] == 1).sum() == 2):
                    selectedMove = coord
                    return selectedMove

# block if opponent wins (column)
def colBlockMoveCheck(currentBoardState, avaibleMoves, activePlayer):
    avaibleMoveCoords = list(avaibleMoves.keys())
    random.shuffle(avaibleMoveCoords)
    for coord in avaibleMoveCoords:
        currentBoardStateCopy = currentBoardState.copy()
        currentBoardStateCopy[coord] = activePlayer
        for i in range(currentBoardStateCopy.shape[0]):
            if 2 not in currentBoardStateCopy[:, i] and (currentBoardStateCopy[:, i] == 1).sum() == 2:
                if not(2 not in currentBoardStateCopy[:, i] and (currentBoardStateCopy[:, i] == 1).sum() == 2):
                    selectedMove = coord
                    return selectedMove

# block if opponent wins (diagonal)
def diagBlockMoveCheck(currentBoardState, avaibleMoves, activePlayer):
    avaibleMoveCoords = list(avaibleMoves.keys())
    random.shuffle(avaibleMoveCoords)
    for coord in avaibleMoveCoords:
        currentBoardStateCopy = currentBoardState.copy()
        currentBoardStateCopy[coord] = activePlayer
        if 2 not in np.diag(currentBoardStateCopy) and (np.diag(currentBoardStateCopy) == 1).sum() == 2:
            if not(2 not in np.diag(currentBoardStateCopy) and (np.diag(currentBoardStateCopy) == 1).sum() == 2):
                selectedMove = coord
                return selectedMove

# block if opponent wins (diagonal-reverse)
def diag2BlockMoveCheck(currentBoardState, avaibleMoves, activePlayer):
    avaibleMoveCoords = list(avaibleMoves.keys())
    random.shuffle(avaibleMoveCoords)
    for coord in avaibleMoveCoords:
        currentBoardStateCopy = currentBoardState.copy()
        currentBoardStateCopy[coord] = activePlayer
        if 2 not in np.diag(np.fliplr(currentBoardStateCopy)) and (np.diag(np.fliplr(currentBoardStateCopy)) == 1).sum() == 2:
            if not(2 not in np.diag(np.fliplr(currentBoardStateCopy)) and (np.diag(np.fliplr(currentBoardStateCopy)) == 1).sum() == 2):
                selectedMove = coord
                return selectedMove



# two-step check (row)
def rowBlockMoveCheck(currentBoardState, avaibleMoves, activePlayer):
    avaibleMoveCoords = list(avaibleMoves.keys())
    random.shuffle(avaibleMoveCoords)
    for coord in avaibleMoveCoords:
        currentBoardStateCopy = currentBoardState.copy()
        currentBoardStateCopy[coord] = activePlayer
        for i in range(currentBoardStateCopy.shape[0]):
            if 2 not in currentBoardStateCopy[i, :] and (currentBoardStateCopy[i, :] == 1).sum() == 2:
                if not(2 not in currentBoardStateCopy[i, :] and (currentBoardStateCopy[i, :] == 1).sum() == 2):
                    selectedMove = coord
                    return selectedMove

# two-step check (column)
def colBlockMoveCheck(currentBoardState, avaibleMoves, activePlayer):
    avaibleMoveCoords = list(avaibleMoves.keys())
    random.shuffle(avaibleMoveCoords)
    for coord in avaibleMoveCoords:
        currentBoardStateCopy = currentBoardState.copy()
        currentBoardStateCopy[coord] = activePlayer
        for i in range(currentBoardStateCopy.shape[0]):
            if 2 not in currentBoardStateCopy[:, i] and (currentBoardStateCopy[:, i] == 1).sum() == 2:
                if not(2 not in currentBoardStateCopy[:, i] and (currentBoardStateCopy[:, i] == 1).sum() == 2):
                    selectedMove = coord
                    return selectedMove

# two-step check (diagonal)
def diagBlockMoveCheck(currentBoardState, avaibleMoves, activePlayer):
    avaibleMoveCoords = list(avaibleMoves.keys())
    random.shuffle(avaibleMoveCoords)
    for coord in avaibleMoveCoords:
        currentBoardStateCopy = currentBoardState.copy()
        currentBoardStateCopy[coord] = activePlayer
        if 2 not in np.diag(currentBoardStateCopy) and (np.diag(currentBoardStateCopy) == 1).sum() == 2:
            if not(2 not in np.diag(currentBoardStateCopy) and (np.diag(currentBoardStateCopy) == 1).sum() == 2):
                selectedMove = coord
                return selectedMove

# two-step check (diagonal-reverse)
def diag2BlockMoveCheck(currentBoardState, avaibleMoves, activePlayer):
    avaibleMoveCoords = list(avaibleMoves.keys())
    random.shuffle(avaibleMoveCoords)
    for coord in avaibleMoveCoords:
        currentBoardStateCopy = currentBoardState.copy()
        currentBoardStateCopy[coord] = activePlayer
        if 2 not in np.diag(np.fliplr(currentBoardStateCopy)) and (np.diag(np.fliplr(currentBoardStateCopy)) == 1).sum() == 2:
            if not(2 not in np.diag(np.fliplr(currentBoardStateCopy)) and (np.diag(np.fliplr(currentBoardStateCopy)) == 1).sum() == 2):
                selectedMove = coord
                return selectedMove




# --------------- #



# ---- TESTS ---- #
# Game Test
def gameTest():
    game = TicTacToe()
    game.whoStart()
    print('Player', game.activePlayer, 'makes the first move')
    print('Board initial state\n', game.board)
    # make the first move
    gameStatus, board = game.move(game.activePlayer, (0,0))
    print('New Board status\n', game.board)
    print('Game Status', gameStatus)
    # make the next move
    gameStatus, board = game.move(game.activePlayer, (1,1))
    print('New Board status\n', game.board)
    print('Game Status', gameStatus)
    # make the next move
    gameStatus, board = game.move(game.activePlayer, (0,1))
    print('New Board status\n', game.board)
    print('Game Status', gameStatus)

    print('Possible moves: ') 
    pprint.pprint(moveGenerator(game.board, game.activePlayer))

    # make the next move
    gameStatus, board = game.move(game.activePlayer, (1,2))
    print('New Board status\n', game.board)
    print('Game Status', gameStatus)
    # make the next move
    gameStatus, board = game.move(game.activePlayer, (0,2))
    print('New Board status\n', game.board)
    print('Game Status', gameStatus)

#gameTest()

# Move Selection Test
def moveSelectorTest():
    game = TicTacToe()
    game.whoStart()
    print('Player', game.activePlayer, 'starts the game')
    print('current board status\n', game.board)

    selectedMove,newBoardState, score = moveSelector(model, game.board, game.activePlayer)
    print('Select of move made', selectedMove)
    print('New Board status')
    pprint.pprint(newBoardState.reshape(3,3))
    print('Score for the selected move', score)

#moveSelectorTest()

