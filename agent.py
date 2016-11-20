from copy import deepcopy
import numpy as np
import random
import itertools
import cPickle as pickle

class agent(object):
    '''
    This is an abstract agent class, however its abstractness is not enforced 
    via the ABC module. This is a deliberate design choice. 
    The brains of the agent class is the pick_action() method. 
    Within the project, three classes extend agent, namely:
    
    1- manual_agent: where pick_action() asks for human input. This is used
       if you would like to play with a hard-coded logic or a trained
       TD-learner
    2- teacher     : where pick_action() follows a (nearly) optimal hard-coded logic.
       This is used to train the td-learner
    3- td_learner  : where pick_action() follows an epsilon greedy policy selection,
       and depending on the user input does on-policy (SARSA) or off-policy (Q-learning)
       learning            
    Attributes:
    ----------------------
    name            -  (str) the name you would like to give to the agent
    marker          -  (int) the marker that the agent will use to play. 
                       This is either 1 or -1 (rather than 'X' and 'O')
    board_dimensions - (tuple) The size of the tic-tac-toe board. For now 
                       hard-coded as 3*3            

    '''
    def __init__(self, name, marker):
        
        self.name = name 
        self.marker = marker
        self.oppmarker = -1 if self.marker==1 else 1
        self.board_dimensions = (3,3) 

    def pick_action(self, env):
        '''
        Given the environment the agent is dealing with, pick one of the legal 
        environment actions
        Parameters:
        --------------------
        '''
        return 
    
class manual_agent(agent):
    ''' 
    Agent that relies on human input to pick actions. The pick_action() method
    renders the current state of the environment and asks for input from the 
    human user.
    Attributes:
    ----------------------    
    name            -  (str) the name you would like to give to the agent
    marker          -  (int) the marker that the agent will use to play. 
                       This is either 1 or -1 (rather than 'X' and 'O')
    board_dimensions - (tuple) The size of the tic-tac-toe board. For now 
                       hard-coded as 3*3     
    '''
    
    def __init__(self, name, marker):
        ''' 
        Return a manual agent object with the indicated
        name and marker (-1 or 1 ) of choice. '''
        
        super(manual_agent, self).__init__(name, marker)
        self.epsilon = 0 # TO DO: here for plotting convenience / remove


    def pick_action(self, obs):
        ''' 
        Ask for manual action input from the human user.  
        If necessary, this would allow printing instructions 
        for the human as well.
        Parameters:
        -------------
        obs - (np.array) the current environment (board) state
              For example: np.array([0,0,0,0,0,0,0,0,0]) for an
              empty board.
        '''
        
        action = None
        available_actions = np.where(obs==0)[0] 
        while action not in available_actions:
            # print out what to do
            self.instruct_human(obs, available_actions)
            action = raw_input()
            try:
                action = int(action)
                if action not in available_actions:
                    print 'Sorry, %s is not a valid action.'%(str(action))
            except:
                if action.lower().startswith('inst'):
                    self.print_instructions()
                else:
                    print 'Sorry, \'%s\' is not a valid input.'%(str(action))
                    

        return action

    def instruct_human(self, obs, available_actions):
        '''
        Based on the input board configuration and the available
        actions, instruct the human to input a vali (available)
        action.
        Parameters:
        --------------
        obs               -(np.array) the current environment (board) state
                           For example: np.array([0,0,0,0,0,0,0,0,0]) for an
                           empty board.
        available)actions -(list) the available actions based on the unoccupied
                           cells on the board. The player can only pick one of 
                           these.
        '''
        print 'Current board: '
        self.render(obs)
        print 'Your marker is %i. What is your next move?'%(self.marker) 
        print 'Please pick one of %s'%(str(available_actions))
        print 'Type: Instructions for locations mapping.'


    def print_instructions(self):
        '''
        Prints the instructions (as below) out to the reader.
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

    def render(self,obs):
        ''' 
        Hard-coded visualisation of the current state of the board
        Parameters:
        -------------
        obs - (np.array) the current environment (board) state
              For example: np.array([0,0,0,0,0,0,0,0,0]) for an
              empty board.
        '''
        board = [str(i) for i in obs]
        # "board" is a list of 10 strings representing the board (ignore index 0)
        print(' ' + board[0] + ' | ' + board[1] + ' | ' + board[2])
        print('-----------')
        print(' ' + board[3] + ' | ' + board[4] + ' | ' + board[5])
        print('-----------')
        print(' ' + board[6] + ' | ' + board[7] + ' | ' + board[8])


class teacher(agent):
    ''' 
    Agent with a hard-coded game-play logic for a 3*3 board configuration.
    The logic mainly involves looking one move ahead to see whether the agent can 
    win the game immediately, or whether it can avoid losing by blocking 
    an immediate win opportunity by the opponent.
    If neither is possible, the agent first tries to choose an unoccupied corner,
    then the centre and finally one of the sides (in this preference order).
    There is also a special logic to deflect certain moves when the opponent
    starts the game by picking one of the corners.
    
    Attributes:
    ----------------------    
    name             - (str) the name you would like to give to the agent
    marker           - (int) the marker that the agent will use to play. 
                       This is either 1 or -1 (rather than 'X' and 'O')
    board_dimensions - (tuple) The size of the tic-tac-toe board. For now 
                       hard-coded as 3*3 
    epsilon          - (float) the randomness factor. For instance, if this is 0.2,
                       the teacher object would act randomly 20% of the time and 
                       carries out its hard-coded logic 80% of the time.
                       This is set to 0.5 during training to allow some state
                       space exploration for the opponent (a td-learner).
    '''

    def __init__(self, name, marker, epsilon):
        '''
        Creates a teacher object that acts on a hard-coded logic as specified
        by its pick_action() method.
        '''
        super(teacher, self).__init__(name, marker)
        self.epsilon = epsilon
        

    def pick_action(self, obs):
        '''
        Given the environment the agent is dealing with, returns one of the legal 
        environment actions based on the hard-coded logic, whereby the agent
        looks one move ahead to see whether the agent can win the game immediately,
        or whether it can avoid losing by blocking an immediate win opportunity 
        by the opponent.
        If neither is possible, the agent first tries to choose an unoccupied corner,
        then the centre and finally one of the sides (in this preference order).
        There is also a special logic to deflect certain moves when the opponent
        starts the game by picking one of the corners.
        Parameters:
        ----------------
        obs - (np.array) the current (board) state
              For example: np.array([0,0,0,0,0,0,0,0,0]) for an
              empty board.        
        '''
        available_actions =  np.where(obs==0)[0] 

        if np.random.rand() < self.epsilon: # random action
            return np.random.choice(available_actions)
        else:
            if len(available_actions) == 1:  # agent has no option to pick another action
                return available_actions[0]

            else:  # pick action based on the hard-coded logic
                
                # special logic for when the opponent starts at a corner
                if (len(available_actions) ==8) and any(obs[[0,2,6,8]] !=0):
                    return 4
                # special logic for when the opponent starts with opposite corners
                elif (len(available_actions) ==6) and (sum(obs[[0,8]]) == 2*self.oppmarker or
                                                        sum(obs[[2,6]]) == 2*self.oppmarker):
                    sides = np.intersect1d(np.array([1, 3, 5, 7]), available_actions)
                    return random.choice(sides)
                else:
                    # 1. Check if we can win the game with this move 
                    for action in available_actions:
                        board_copy_attack = self.__considerMove(self.marker, action, obs)
                        if self.__isWinner(board_copy_attack):  # First, attempt to win with this move
                            return action

                    # 2. Check if the opponent can win on their next move, and block them.                         
                    for action in available_actions: 
                        board_copy_defend = self.__considerMove(self.oppmarker, action, obs)
                        if self.__isWinner(board_copy_defend):  # If not possible, defend
                            return action   
    
                    # 3. Take one of the corners, if they are free.
                    corners = np.intersect1d(np.array([0, 2, 6, 8]), available_actions)
                    if len(corners)>0:
                        return random.choice(corners)
    
                    # 4. Take the centre if it is free.
                    if 4 in available_actions:
                        return 4
    
                    # 5. If nothing else is free, take one of the free side locations
                    sides = np.intersect1d(np.array([1, 3, 5, 7]), available_actions)
                    return random.choice(sides)


    def __considerMove(self, mark, move, obs):
        '''
        Given a move, a player marker and the current board configuration, 
        return a (temporary) copy of the board with the considered move applied.
        Parameters:
        ----------------
        mark  - {-1,1} the numeric marker (equivalent to 'X' or 'O') for the
                tic-tac-toe game.
        move  - (int) a legal move location on the board.
        obs   - (np.array) the 3*3 board configuration to test a move on
        '''
        board_copy = deepcopy(obs)
        board_copy[move] = mark
        return board_copy

    def __isWinner(self, board):
        '''
        Given a board configuration, this method returns True if there is a winner.
        Note: There is an equivalent method in the tic_tac_toe class, however in 
        order to keep the agent and the environment isolated, this class has its 
        own _isWinner() implementation, at the expense of minor repetition.
        Parameters:
        -----------------
        obs - (np.array) the current (board) state
              For example: np.array([0,0,0,0,0,0,0,0,0]) for an
              empty board. 
         '''
        board_mat =board.reshape(self.board_dimensions)
        
        # episode ending criteria
        row_complete = any(abs(board_mat.sum(axis=1))==3 )
        col_complete = any(abs(board_mat.sum(axis=0))==3 )
        diag_complete = abs(np.diag(board_mat).sum())==3 
        opp_diag_complete = abs(np.diag(np.fliplr(board_mat)).sum()) == 3

        # if any of the criteria satisfies, episode complete
        if any([row_complete, col_complete, diag_complete, opp_diag_complete]) :
            return True
        else:
            False

class td_learner(agent):
    ''' 
    Epsilon-greedy temporal difference (td) learner that is trained based 
    on the equation:
    Q(s,a) <-- Q(s,a) + alpha * [target - prediction], where:
    prediction = Q(s,a), 
    and 
    target = r + gamma * max_a'[Q(s',a')] for Q-learning,
    or
    target = r + gamma * [ (1-epsilon)* max_a'[Q(s',a')] + 
                           epsilon* mean[Q(s',a') |a'!= optimal a'] ] for SARSA.
    
    Attributes:
    ----------------------    
    name             - (str) the name you would like to give to the agent
    marker           - (int) the marker that the agent will use to play. 
                       This is either 1 or -1 (rather than 'X' and 'O')
    board_dimensions - (tuple) The size of the tic-tac-toe board. For now 
                       hard-coded as 3*3 
    epsilon          - (float) the randomness factor. For instance, if this is 0.2,
                       the agent would act randomly 20% of the time and pick
                       the optimal action 80% of the time.
                       This is annealed from 1 to 0.05 during training to allow 
                       some state space exploration.
                       In other words, this adjusts the exploration / exploitation
                       balance.
    learning         - {'off-policy', 'on-policy'} defines whether update_q_table()
                       operates off-policy (i.e. Q-Learning) or on-policy (SARSA)
    learn_rate       - (float) the learning rate (alpha) for the td-update formula 
                       given above.
    gamma            - (float) the future reward discount factor in the td-update 
                       formula given above. Its choice should be informed by
                       average episode (game-play) duration.
    action_set_size  - (int) the number of available actions. By default, if the agent 
                        is used with the tic-tac-toe environment, this is equal to the
                        state dimension size, i.e. 3*3
    q_dict           - (dict) the Q-value lookup table of the td-learner, implemented
                        as a dictionary where keys are tuples of unique states and the
                        values are the available actions per each state.
                        This can either be initialised empty or loaded from an existing
                        pickled q value dictionary.       
    q_table_dir      - (str) the pickled file location of a previously trained
                       Q value dictionary. If this is not None, instead of creating
                       an empty Q value dictionary, the object is initialised by 
                       reading in the pickled dictionary at this location.
          
    '''

    def __init__(self, name, marker, state_dimensions, learning, epsilon, learn_rate, 
                 gamma, action_set_size = None, q_table_dir = None):
        '''
        Creates a Temporal-Difference Learner object. Depending on the user-defined
        'learning' parameter, the agent either does off-policy (Q-Learning) or 
        on-policy (SARSA) learning.
        '''         
        
        super(td_learner, self).__init__(name, marker)
        self.epsilon = epsilon
        self.final_epsilon = 0.05 # hard-coded, make dynamic
        self.learning_rate = learn_rate
        self.gamma = gamma
        self.learning = learning
        if action_set_size is None:
            self.__action_space_size =  len(state_dimensions)
        else:
            self.__action_space_size = action_set_size
        if q_table_dir is None:
            self.q_dict = self.__create_q_table(state_dimensions)
        else:
            self.q_dict = self.load_q_dict(q_table_dir)



    def __create_q_table(self, state_dimensions):
        '''
        Create a Q lookup dict by taking the Cartesian product of all the dimensions in the  
        state space. For the 3*3 tic-tac-toe environment, each cell can have [-1,0,1]. 
        So there are 3**(3*3) configurations. Some of these are not legal game plays,
        e.g.(1,1,1,1,1,1,1,1,1) but for the time being (due to time constraints) we do 
        not worry about these.
        Each product is then used as a tuple key, pointing to a numpy array of size 9, each 
        representing an available location, i.e. action index.
        The lookup dictionary is constrained in the sense that when we know a certain state 
        can not allow an action (i.e. that particular location is already occupied),
        we populate the Q value for that action as np.nan.
        For instance:
        q_dict[(1,1,1,1,1,1,1,1,1)] = array([ nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan])
        q_dict[(0,0,1,-1,0,0,0,0,0)] = array([-0.06, -0.04,   nan,   nan, -0.03, -0.03, -0.07,  0.04,  0.06])
        etc..
        Parameters:
        ------------
        state_dimensions - (list) the state dimensions for the environment that the 
                           agent will interact with. For the tic_tac_toe env, this is 
                           a list of all possible markers [-1,0,1] repeated 3*3 times.           
        '''
        n = self.__action_space_size # for brevity below, create temp variable
        q_dict = dict([(element, np.array([(i == 0 and [np.random.uniform(0, 0)] or [np.nan])[0]
                        for i in element]))
                      for element in itertools.product(*state_dimensions )])
        return q_dict
    

    def set_epsilon(self, val):
        '''
        Manually adjust the td_learner agent's epsilon. This is not very clean
        but is done while the td-learner is playing against a manual_agent (human)
        to ensure that we play with a fully exploitative, non-random td-learner agent.
        Parameters:
        ----------------
        val  - (float) the value we want to set the epsilon to
        '''
        self.epsilon = val

    def save_q_dict(self, name):
        '''
        Pickles and saves the current Q-value dictionary of the agent
        with the provided file name.
        This is used to save the results of a trained agent so we can play
        with it without having to retrain.
        Parameters:
        ----------------
        name  - (str) the directory and name that we want to give to the 
                pickled Q dictionary file. Example: 'on_policy_trained.p'
        '''
        with open(name, 'wb') as fp:
            pickle.dump(self.q_dict, fp)

    
    def load_q_dict(self, dir):
        '''
        Loads the pickled dictionary in the provided location as the agent's 
        q-value dictionary play. We can initialise a td-learner in this way and
        play with it directly, without having to retrain a blank one.
        Parameters:
        ----------------
        name  - (str) the directory and name of the pickled Q value dictionary
                file that we want to load. 
        '''
        with open(dir, 'rb') as fp:
            q_dict = pickle.load(fp)

        return q_dict

    def pick_action(self, obs):
        '''
        Pick action in an epsilon-greedy way. By self.epsilon probability 
        this returns a random action, and by (1-epsilon)
        probability it returns the action with the maximum q-value
        for the current environment state.
        Parameters:
        ---------------
        obs - (np.array) the current (board) state to pick an action on.
              For example: np.array([0,0,0,0,0,0,0,0,0]) for an
              empty board. 
        
        '''
        if np.random.rand() < self.epsilon: # random action
            action = np.random.choice(np.where(obs==0)[0])
        else:                               # action with the max q-value
            action = np.nanargmax(self.__get_state_vals(obs))


        return action
 
    def update_q_table(self, obs, action, reward, next_obs, done):
        '''
        Implementation of the temporal difference learning update:
        Q(s,a) <-- Q(s,a) + alpha * [target - prediction].
        where:
        prediction = Q(s,a), 
        and 
        target = r + gamma * max_a'[Q(s',a')] for Q-learning,
        or
    def update_q_table(self, obs, action, reward, next_obs, done):
        '''
        Implementation of the temporal difference learning update:
        Q(s,a) <-- Q(s,a) + alpha * [target - prediction].
        where:
        prediction = Q(s,a), 
        and 
        target = r + gamma * max_a'[Q(s',a')] for Q-learning,
        or
        target = r + gamma * [ (1-epsilon)* max_a'[Q(s',a')] + 
                               epsilon* mean[Q(s',a')] for SARSA.
    def update_q_table(self, obs, action, reward, next_obs, done, func):
        '''
        Implementation of the temporal difference learning update:
        Q(s,a) <-- Q(s,a) + alpha * [target - prediction].
        where:
        prediction = Q(s,a), 
        and 
        target = r + gamma * max_a'[Q(s',a')] for Q-learning,
        or
        target = r + gamma * [ (1-epsilon)* max_a'[Q(s',a')] + 
                               epsilon* mean[Q(s',a')] for SARSA.
        
        The definition of the target changes depending on whether the learning is done
        off-policy (Q-Learning) or on-policy (SARSA).
        Off-policy (Q-Learning) computes the difference between Q(s,a) and the maximum  
        action value, while on-policy (SARSA) computes the difference between Q(s,a) 
        and the weighted sum of the average action value and the maximum.

        Parameters:
        ---------------
        obs      - (np.array), the state we transitioned from (s).
        action   - (int) the action (a) taken at state=s.
        reward   - (int) the reward (r) resulting from taking the specific action (a)
                   at state = s.
        next_obs - (np.array) the next state (s') we transitioned into
                   after the taking the action at state=s.
        done     - (bool) episode termination indicator. If True, target (above) is
                   only equal to the immediate reward (r) and there is no discounted
                   future reward
        func     - (np.nanmax, np.nanmin) Should update with max if it is the agent's turn  
                   and should take min if the opponent's turn
        '''
        if self.learning == 'off-policy':  # Q-Learning

            if done: # terminal state, just immediate reward
                target = reward
            else: # within episode
                target = reward + self.gamma*func(self.__get_state_vals(next_obs))            
            prediction = self.__get_state_vals(obs)[action]
            updated_q_val = prediction + self.learning_rate *(target - prediction)
            # update the q-value for the observed state,action pair     
            self.__set_q_val(obs, action, updated_q_val)

        elif self.learning == 'on-policy': # SARSA

            if done: # terminal state, just immediate reward
                target = reward
            else: # within episode
                on_policy_q = self.epsilon * np.nanmean(self.__get_state_vals(next_obs)) + \
                              (1- self.epsilon) * func(self.__get_state_vals(next_obs)) 
                target = reward + self.gamma*on_policy_q           
            prediction = self.__get_state_vals(obs)[action]
            updated_q_val = prediction + self.learning_rate *(target - prediction)

            # update the q-value for the observed state,action pair     
            self.__set_q_val(obs, action, updated_q_val)
        else:
            raise ValueError ('Learning method is not known.')
            
    def on_policy_q_target(self, next_obs):
        '''
        Calculate the target in the TD learning update function:
        Q(s,a) <-- Q(s,a) + alpha * [target - prediction] when the learning is
        done on policy. In this case, target is the difference between Q(s,a) 
        and the weighted sum of the average action value and the maximum. The
        weighting is done using self.epsilon.
        target = r + gamma * [ (1-epsilon)* max_a'[Q(s',a')] + epsilon* mean[Q(s',a') |a'!= optimal a'] ]
        Parameters:
        -----------------
        next_obs - (np.array) the next state (s') we transitioned into
                   after the taking the action at state=s.
        '''
        # next action candidates
        a_prime_candidates = deepcopy(self.__get_state_vals(next_obs)) 
        # optimum next state action (greedy selection)
        optimum_a_prime_idx = np.nanargmax(a_prime_candidates)
        # on_policy_q = (1-eps)*optimal a' + eps*E[non_optimal a']
        exp_greedy_q = self.epsilon*a_prime_candidates[optimum_a_prime_idx]
        if all(np.isnan((np.delete(a_prime_candidates,optimum_a_prime_idx)))  ):
            exp_random_q = 0 
        else:
            exp_random_q = (1-self.epsilon)* np.nanmean(np.delete(a_prime_candidates,optimum_a_prime_idx))

        return exp_greedy_q + exp_random_q


    def __set_q_val(self, state, action, q_val):
        '''
        Set the q value for a state-action pair in the object's q val dictionary.
        Parameters:
        -----------------
        state  -(list) the state index, for a 3*3 board
        action -(int) the action index
        q_val  -(float) the Q value to appoint to the state-action pair
        '''   
        self.q_dict[tuple(state)][action]  = q_val 

 
    def __get_state_vals(self, state):
        '''
        For a given state, look up and return the the action values from 
        the object's q val dictionary.The q values are returned as a dictionary with
        keys equal to action indices and the values the corresponding q values.
        The output is a dictionary to facilitate post-processing and filtering 
        out some q-values that belong to unavailable action locations.

        Parameters:
        -----------------
        state  -(list) the state index, for a 3*3 board
        '''   
        d = self.q_dict[tuple(state)]
        return d 
    
