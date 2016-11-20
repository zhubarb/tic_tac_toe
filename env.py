import numpy as np
from operator import mul
from copy import deepcopy

class tic_tac_toe(object):
    """
    An emulator for the Tic-Tac-Toe game. The board size is hard-coded as 3*3
    for the time being. When the game starts, the state of the board is initialised
    a list of zeros with length (3*3).
    Actions are flattened cell indexes of the board, ranging from 0 to 8.
    For ease of calculation, instead of the original ['X', 'O', ' '] markers, this
    emulator uses -1, 0, and 1. According to this, 0 represents an empty cell, while
    -1 and 1 represent an occupied one.

    Attributes:
    ----------------------
    board_dimensions - (tuple) The size of the tic-tac-toe board. For now hard-coded
                        as 3*3
    action_space     - (list) The actions available to players. These denote the locations
                        on the board, starting to count from 0 to 8 in the following order:
                        0 | 1 | 2
                       -----------
                        3 | 4 | 5
                       -----------
                        6 | 7 | 8
    state            - (np.array) Flattened matrix of board_dimensions
    markers          - (list) The allowed markers for each cell on the board
    action_space     - (list) all available actions when the board is empty, 
                       these correspond to the 9 squares within the 3*3 board.
    state_dimensions - (list) the state space, each square on the board is one state
                       and each state can rake on the value of self.markers
    state            - (list) initial state of the board, completely empty 
    """

    def __init__(self):
        '''
        Return a new tic-tac-toe environment with an empty 3*3 board. 
        '''
        
        self.__board_dimensions = (3,3)
        self.markers = [1, 0, -1]
        self.action_space = np.array( range(reduce(mul, self.__board_dimensions)) ) # dynamic, based on board_dimensions
        self.state_dimensions = [self.markers for _ in self.action_space]
        self.state = np.zeros(self.__board_dimensions, dtype=int).flatten() # initial board state
      

    def print_instructions(self):
        '''
        Print out environment instructions.
        '''
        inst = '---------------Instructions --------------- \n'\
        'Available actions are integers from 0 to 8 on a 3*3 board \n'\
        'with the location mapping as below:\n' \
        '0 | 1 | 2 \n'\
        '----------\n'\
        '3 | 4 | 5 \n'\
        '----------\n'\
        '6 | 7 | 8 \n'\
        'If the location you are choosing is already taken (is not 0),the computer \n' \
        'will keep on asking you to put in an available location integer. \n'\
        '------------------------------------------'
        print inst

    def reset(self):
        ''' 
        Reset the state of the environment and available actions for a new episode 
        (in this case empty board)
        '''
        self.state = np.zeros(self.__board_dimensions, dtype=int).flatten()


    def step(self, action, marker):
        '''
        Given the provided action index and the marker,
        Parameters:
        --------------
        action - (int) a legal move location on the board. 
                 For an empty board, action can be any of [0, 1, 2, 3, 4, 5, 6, 7, 8]
        marker - (int) +1 or -1 depending on the marker appointment of the agent.
                 These are used instead of 'X' and 'O' for ease of calculation.
        '''
        if marker not in self.markers:
            raise ValueError('The marker specified is not a member of [-1,0,1]')
        else:
            # update environment state
            self.state[action] = marker 
            
            # deepcopy the env state after move as the next observation
            next_obs = deepcopy(self.state)
    
            # update available actions for the current environment instance
            self.available_actions = np.array([i for i in xrange(len(self.state)) if self.state[i] ==0])
           
            # check terminal state and return reward and, where applicable, info
            reward, done, info = self.__isTerminal(marker)
            
            return (next_obs, reward, done, info)

    def render(self):
        ''' 
        Hard-coded visualisation for 3*3 environment states (board configurations)
        '''
        # TO DO: Visualise -1 and 1 with X and O if you have time.
        board = [str(i) for i in self.state]
        # "board" is a list of 10 strings representing the board (ignore index 0)
        print(' ' + board[0] + ' | ' + board[1] + ' | ' + board[2])
        print('-----------')
        print(' ' + board[3] + ' | ' + board[4] + ' | ' + board[5])
        print('-----------')
        print(' ' + board[6] + ' | ' + board[7] + ' | ' + board[8]) 

    def __isTerminal(self, marker):
        '''
        Check whether the episode has ended or is ongoing and return a corresponding 
        reward, terminal state indicator (done) and where applicable information (info).
        The game end reward is given as 1 regardless of the identity (human, 
        td-learner, teacher) of the winner. 
        While training, the rewards are translated to the td-learner perspective in the 
        main game play function: play_episode() in train_and_play.py. 
        An episode terminates when:
        i. One of the players wins, i.e. __is_Winner() returns True.
        ii.The board is full, i.e. __isBoardFull() returns True 
        Parameters:
        --------------
        marker - (int) +1 or -1 depending on the marker appointment of the agent.
                 These are used instead of 'X' and 'O' for ease of calculation.
        '''

        if self.__isWinner(): # there is a winner
            reward = 1
            done = True
            info = 'Player with marker: %i wins!'%(marker)
        elif self.__isBoardFull(): # draw
            reward = 0
            done = True
            info = 'The board is full, it is a draw.'
        else: # the game is going on
            reward = 0
            done = False
            info = 'Game goes on...'

        return (reward, done, info)

    def __isWinner(self, board=None):
        '''
        Given a board configuration, this method returns True if there is a winner.
        Parameters:
        -------------
        board  - (np.array / None) board state (configuration)
                  If None, we use the environment's state by default.
        '''
        if board is None:
            board = self.state

        board_mat =board.reshape(self.__board_dimensions)
        
        # episode ending criteria:
        row_complete = any(abs(board_mat.sum(axis=1))==3 )
        col_complete = any(abs(board_mat.sum(axis=0))==3 )
        diag_complete = abs(np.diag(board_mat).sum())==3 
        opp_diag_complete = abs(np.diag(np.fliplr(board_mat)).sum()) == 3

        # if any of the criteria is satisfied, episode complete
        if any([row_complete, col_complete, diag_complete, opp_diag_complete]) :
            return True
        else:
            False

    def __isBoardFull(self):
        '''
        Return True if every space on the board has been taken, otherwise
        return False.
        '''
        if (self.state==0).any():
            return False
        else:
            return True