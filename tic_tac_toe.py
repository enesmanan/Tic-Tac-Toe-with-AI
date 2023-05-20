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
def rowSecondMoveCheck(currentBoardState, avaibleMoves, activePlayer):
    avaibleMoveCoords = list(avaibleMoves.keys())
    random.shuffle(avaibleMoveCoords)
    for coord in avaibleMoveCoords:
        currentBoardStateCopy = currentBoardState.copy()
        currentBoardStateCopy[coord] = activePlayer
        for i in range(currentBoardStateCopy.shape[0]):
            if 1 not in currentBoardStateCopy[i, :] and (currentBoardStateCopy[i, :] == 0).sum() == 2:
                if not(1 not in currentBoardStateCopy[i, :] and (currentBoardStateCopy[i, :] == 0).sum() == 2):
                    selectedMove = coord
                    return selectedMove

# two-step check (column)
def colSecondMoveCheck(currentBoardState, avaibleMoves, activePlayer):
    avaibleMoveCoords = list(avaibleMoves.keys())
    random.shuffle(avaibleMoveCoords)
    for coord in avaibleMoveCoords:
        currentBoardStateCopy = currentBoardState.copy()
        currentBoardStateCopy[coord] = activePlayer
        for i in range(currentBoardStateCopy.shape[0]):
            if 1 not in currentBoardStateCopy[:, i] and (currentBoardStateCopy[:, i] == 0).sum() == 2:
                if not(1 not in currentBoardStateCopy[:, i] and (currentBoardStateCopy[:, i] == 0).sum() == 2):
                    selectedMove = coord
                    return selectedMove

# two-step check (diagonal)
def diagSecondMoveCheck(currentBoardState, avaibleMoves, activePlayer):
    avaibleMoveCoords = list(avaibleMoves.keys())
    random.shuffle(avaibleMoveCoords)
    for coord in avaibleMoveCoords:
        currentBoardStateCopy = currentBoardState.copy()
        currentBoardStateCopy[coord] = activePlayer
        if 1 not in np.diag(currentBoardStateCopy) and (np.diag(currentBoardStateCopy) == 0).sum() == 2:
            if not(1 not in np.diag(currentBoardStateCopy) and (np.diag(currentBoardStateCopy) == 0).sum() == 2):
                selectedMove = coord
                return selectedMove

# two-step check (diagonal-reverse)
def diag2SecondMoveCheck(currentBoardState, avaibleMoves, activePlayer):
    avaibleMoveCoords = list(avaibleMoves.keys())
    random.shuffle(avaibleMoveCoords)
    for coord in avaibleMoveCoords:
        currentBoardStateCopy = currentBoardState.copy()
        currentBoardStateCopy[coord] = activePlayer
        if 1 not in np.diag(np.fliplr(currentBoardStateCopy)) and (np.diag(np.fliplr(currentBoardStateCopy)) == 0).sum() == 2:
            if not(1 not in np.diag(np.fliplr(currentBoardStateCopy)) and (np.diag(np.fliplr(currentBoardStateCopy)) == 0).sum() == 2):
                selectedMove = coord
                return selectedMove

# competitive move selection
def openentMoveSelector(currentBoardState, activePlayer, mode):

    winingMoveChecks = [rowWinMoveCheck, colWinMoveCheck, diag2WinMoveCheck, diag2WinMoveCheck] 
    blockMoveChecks = [rowBlockMoveCheck, colBlockMoveCheck, diagBlockMoveCheck, diag2BlockMoveCheck]
    secondMoveCheck = [rowSecondMoveCheck, colSecondMoveCheck, diagSecondMoveCheck, diag2SecondMoveCheck]
    
    availableMoves = moveGenerator(currentBoardState, activePlayer)

    if mode == 'Easy':
        selectedMove = random.choice(list(availableMoves.keys()))
        return selectedMove
    
    elif mode == 'Hard':
        random.shuffle(winingMoveChecks)
        random.shuffle(blockMoveChecks)
        random.shuffle(secondMoveCheck)

        for fn in winingMoveChecks:
            if fn(currentBoardState, availableMoves, activePlayer):
                return fn(currentBoardState, availableMoves, activePlayer)
        for fn in blockMoveChecks:
            if fn(currentBoardState, availableMoves, activePlayer):
                return fn(currentBoardState, availableMoves, activePlayer)
        for fn in secondMoveCheck:
            if fn(currentBoardState, availableMoves, activePlayer):
                return fn(currentBoardState, availableMoves, activePlayer)
        
        selectedMove = random.choice(list(availableMoves.keys()))
        return selectedMove
# --------------- #


# Train
def train(model, mode, print_progress=False):
    if print_progress == True:
        print('----------------------------------------')
        print('New Game Started')

    game = TicTacToe()
    game.whoStart()
    scoreList = []
    correctedScoreList = []
    newBoardStateList = []

    while(1):
        if game.gameStatus() == 'In Progress' and game.activePlayer == 1:
            # If the turn is in the AI, use the move selector for the move
            selectedMove, newBoardState, score = moveSelector(model, game.board, game.activePlayer)
            scoreList.append(score[0][0])
            newBoardStateList.append(newBoardState)
            # Make the next move
            gameStatus, board = game.move(game.activePlayer, selectedMove)
            if print_progress == True:
                print('AI \'s move')
                print(board, '\n')
        elif game.gameStatus() == 'In Progress' and game.activePlayer == 0:
            selectedMove = openentMoveSelector(game.board, game.activePlayer, mode=mode)
            # make the next move
            gameStatus, board = game.move(game.activePlayer, selectedMove)
            if print_progress == True:
                print('Competitive bot\'s move')
                print(board, '\n')
        else:
            break


    # Score correction 1 / 0 / -1 --> win / draw / lose
    newBoardStateList = tuple(newBoardStateList) 
    newBoardStateList = np.vstack(newBoardStateList)

    if gameStatus == 'Win' and (1-game.activePlayer) == 1:
        correctedScoreList = shift(scoreList, -1, cval = 1.0)
        result = 'Win'
    if gameStatus == 'Win' and (1-game.activePlayer) != 1:
        correctedScoreList = shift(scoreList, -1, cval = -1.0)
        result = 'Lost'
    if gameStatus == 'Draw':
        correctedScoreList = shift(scoreList, -1, cval=0.0)
        result = 'Draw'
    if print_progress == True:
        print('AI: ', result)
        print('-------------------------------------------')


    x = newBoardStateList
    y = correctedScoreList

    def unisonShuffledCopies(a,b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    x, y = unisonShuffledCopies(x, y)
    x = x.reshape(-1, 9)

    # update weights and fit model
    model.fit(x,y, epochs=1, batch_size=1, verbose=0)
    return model, y, result


# one-time fit and update
#updatedModel, y, result = train(model, mode='Hard', print_progress=True)

# learning by playing competitive bots with iterations
gameCounter = 1
modeList = ['Easy', 'Hard']

while(gameCounter<2000):
    modeSelected = np.random.choice(modeList, 1, p=[0.5, 0.5])
    model, y, result = train(model, mode=modeSelected[0], print_progress=False)
    if gameCounter % 5 == 0:
        print('Game #{} - Result: {}'.format(gameCounter, result))
        print('Game Mode: {}'.format(modeSelected[0]))
    gameCounter += 1


# Save the unsupervised trained model
model.save('tictactoeModel.h5')

# Loading the saved model
from keras.models import load_model
model = load_model('tictactoeModel.h5')







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

