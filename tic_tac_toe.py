# required libraries
import numpy as np
import pandas as pd
import random, pprint
from scipy.ndimage.interpolation import shift

# AI moduls
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD

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
        
        # winn - column check
        for j in range(self.board.shape[1]):
            if 2 not in self.board[:, j] and len(set(self.board[:, j ])) == 1:
                return 'Win'
            
        # win - cross check
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
        

