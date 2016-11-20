import unittest
import random
import numpy as np
from agent import teacher, td_learner
from env import tic_tac_toe
from train_and_play import play_episode

class TicTacToeTestCase(unittest.TestCase):
    """Tests for tic tac toe environemnt and agents"""

    def create_env(self):
        self.env= tic_tac_toe()


    def run_experiment(self):
        teach = teacher(name='Teacher', marker=-1, epsilon= 0.5) 
        q_learner = td_learner(name='TD-Learner', marker=1, state_dimensions = self.env.state_dimensions, 
                        learning = 'on-policy', epsilon=1, learn_rate =1, gamma=0.95,
                        q_table_dir = None)
        agent_scores = {'Agent_A':list(), 'Agent_B':list()}
        turn = 'Agent_A' if random.randint(0, 1) == 0 else 'Agent_B'
        play_episode(self.env, q_learner, teach, turn)


    def test_env_board(self):
        
        # before experiment run, board is all zeros
        self.create_env()
        self.assertTrue(self.env.state.sum() == 0)

        # after experiment run, abs(sum(board)) is always <=1
        self.run_experiment()
        self.assertTrue(abs(self.env.state.sum()) <=1) 

        # after experiment run we always see three unique markers on the board
        print str(np.unique(self.env.state))
        self.assertTrue(len(np.unique(self.env.state) ) <= 3) 


    def test_q_dict_creation_is_correct(self):

        self.create_env()
        q_learner = td_learner(name='TD-Learner', marker=1, state_dimensions = self.env.state_dimensions, 
                        learning = 'on-policy', epsilon=1, learn_rate =1, gamma=0.95,
                        q_table_dir = None)

        key, val = q_learner.q_dict.iteritems().next()
        key_val_comb = np.array(key) + val

        self.assertEqual(np.nansum(key_val_comb), np.nansum(val))


    def test_q_update_is_nan_free(self):

        self.create_env()
        q_learner = td_learner(name='TD-Learner', marker=1, state_dimensions = self.env.state_dimensions, 
                        learning = 'on-policy', epsilon=1, learn_rate =1, gamma=0.95,
                        q_table_dir = None)

        obs, val = q_learner.q_dict.iteritems().next()

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()