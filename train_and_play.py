from env import tic_tac_toe
from agent import teacher
from agent import manual_agent
from agent import td_learner
import random
import os
from copy import deepcopy
import numpy as np
import datetime
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig

### Generic Utility Functions ###
def moving_average(arr, window) :
    '''
    Return the moving average of arr numpy array
    Parameters:
    ---------------
    arr    - (np.array)the anumpy array to apply the function on
    window - (int) window size of the moving average
    '''
    ret = np.cumsum(arr, dtype=float)
    ret[window:] = ret[window:] - ret[:-window]
    
    return ret[window - 1:] / window

def view_q_vals(q_learner, obs_a):
    print np.round(q_learner._td_learner__get_state_vals(obs_a),3).reshape(3,3)

def log_parameters(num_episodes, learning_mode, epsilon, eps_decay, teacher_epsilon, 
                   gamma, learning_rate):

    params = '%s training for %i episodes | eps:%s | eps decay:%f |  \n \
    teacher_eps:%s |gamma:%s | lrn_rate:%s'%(learning_mode, num_episodes, 
                                             str(round(epsilon,3)), eps_decay, 
                                             str(round(teacher_epsilon,3)), str(round(gamma,3)), 
                                             str(round(learning_rate,3)) )
    

    return params

def plot_rewards(agent_name, agent_scores, params, name, window):
    '''
    Parameters:
    -------------
    agent_name   - (str)
    agent_scores - (dict)
    params       - (str) important experiment parameters that we want
                   to plot to remember the configuration
    name         - (str) experiment name with the date-time of
                   experiment and any potential notes
    window       - (int) the mvoing average window to be used
    '''
    # calculate moving average 
    mov_avg_scores_A = moving_average(agent_scores['Agent_A'],window) 
    mov_avg_scores_B = moving_average(agent_scores['Agent_B'],window) 
    
    # separately plot the agent's rewards for games where the agent started 
    # and for the games the teacher started
    plt.figure(figsize=(12,8))
    plt.plot(range(len(mov_avg_scores_A)), mov_avg_scores_A, color='blue', 
             label='%s starts'%(agent_name) )
    plt.plot(range(len(mov_avg_scores_B)), mov_avg_scores_B, color='green', 
             label = 'Teacher starts')
    plt.title(params)
    plt.ylabel('%i-Game Moving Average of agent rewards'%(window))
    plt.xlabel('Number of games played')
    plt.tight_layout()
    plt.legend()
    
    fig_name = name + '.png'
    savefig(os.path.join(exp_dir,fig_name))

### Generic Utility Functions End###

def play_episode(env, agent_A, agent_B, turn):
    '''
    Play a single game of tic-tac-toe between two agents, namely q_learner
    and agent_B. If we want to train a td-learner (or or off-policy), we 
    define Agent_A as a td-learner.
    Parameters:
    ----------------
    env    - the tic tac toe game emulator / environment
    agentA - (td_learner) a td_learner object that learns to play based on the
             gameplay of the player
    agentB - (human or teacher) a human or teacher object (both of which extend
             the 'agent' class).
             If this is human, the computer asks for human input. If it is of type
             'teacher', it plays based on the hard-coded game heuristics.
    '''
    done = False # episode end indicator
    while not done:
        if turn == 'Agent_A':
            obs_a = deepcopy(env.state)
            action_a = agent_A.pick_action(obs_a)
            # view_q_vals(agent_A obs_a) ; env.render()
            next_obs_a, reward_a, done, info = env.step(action_a, agent_A.marker)

            if isinstance(agent_A, td_learner):
                agent_A.update_q_table(obs_a, action_a, reward_a, next_obs_a, done, np.nanmin)
            turn = 'Agent_B'

        else:
            obs_b = deepcopy(env.state)
            action_b = agent_B.pick_action(obs_b) 
            next_obs_b, reward_b, done, info = env.step(action_b, agent_B.marker)

           
            if isinstance(agent_A, td_learner):
                # Add agent_B's experience but multiplying the reward by -1 and max function
                agent_A.update_q_table(obs_b, action_b, -1*reward_b, next_obs_b, done, np.nanmax)
                # for A, reverse B's state transition
                #agent_A.update_q_table(-1*obs_b, action_b, reward_b, -1*next_obs_b, done)
                #if A's action resulted in B winning, associate that transition with minus B's reward
                #if (reward_b == 1):
                #    agent_A.update_q_table(obs_a, action_a, -1*reward_b, next_obs_a, False)

            turn = 'Agent_A'

            
    # translate rewards to q_learner's perspective        
    if reward_a == 1:
        reward = 1 
    elif reward_b == 1:
        reward = -1 
    else:
        reward = 0 

    return info, reward

def train_or_play(env, agent_A, agent_B, num_epis, eps_decay, 
                  window, verbose):
    '''
    Train or play tic-tac-toe 'num_epis' times, given the two
    agent objects and the environment.
    Parameters:
    --------------
    env      - the tic tac toe game emulator / environment
    agent_A  - (td_learner) a td_learner object that learns to play based on the
               game play of Agent_B
    agent_B  - (human or teacher) a human or teacher object (both of which extend
               the 'agent' class).
               If this is human, the computer asks for human input. If it is of type
              'teacher', it plays based on the hard-coded game heuristics.
    num_epis - (int) number of episodes to play or train for
    eps_decay- (float) the amount of decay we want to anneal Agent_A's epsilon by 
               at the end of each episode.
    window   - (int) the moving average window for plotting average rewards               
    verbose  - (bool) do you want training / playing to print out status and 
               instructions
    '''

    if isinstance(agent_B, manual_agent):
        # If playing against a human, the td-learner's epsilon has to be set to 0
        # so we play with a fully exploitative, non-random td-learner agent.
        agent_A.set_epsilon(0)   

    # parameter settings for this run
    params = log_parameters(num_epis, agent_A.learning, agent_A.epsilon, eps_decay,
                            agent_B.epsilon, agent_A.gamma, agent_A.learning_rate)
    
    name = agent_A.learning.replace('-','') + '_' + datetime.datetime.now().strftime("%d%m%Y_%H_%M")
    agent_scores = {'Agent_A':list(), 'Agent_B':list()}
    # Run training
    i_episode = 0 
    while i_episode < num_epis:

        env.reset()
        turn = 'Agent_A' if random.randint(0, 1) == 0 else 'Agent_B' # flip coin
        turn_indic = turn

        if verbose:
            print '## Playing match %i, %s will start ##'%(i_episode+1, turn)

        # play a single episode
        info, reward = play_episode(env, agent_A, agent_B, turn)
    
        # store td-learner's reward (for plotting) from this episode
        agent_scores[turn_indic].append(reward)
            
        # episode over
        if verbose:
            env.render()
            print info + '\n'
        assert abs(env.state.sum()) <= 1 # TO DO: remove
        if i_episode%10000 == 0:
            print 'Training for %i games: Played %i games so far.'%(num_epis, i_episode)
        if agent_A.epsilon > agent_A.final_epsilon:  
            agent_A.epsilon -= eps_decay
  
        i_episode += 1 
    
    # plot the rewards of agentA (the td learner)    
    plot_rewards(agent_A.name, agent_scores, params, name, window)
    
    # output the learned Q dictionary and the moving averages of episode scores.
    q_dict_name = 'table_' + name  +'.p'
    agent_A.save_q_dict(os.path.join(exp_dir,q_dict_name))

    return agent_A

# Game-play and / or agent training happens here:
if __name__ == '__main__':

    # Define experimental parameters
    verbose = False
    exp_dir = '' # /deemind_results/
    window = 1000 # window size for plotting agent's moving average rewards
    num_episodes = int(1e5) # number of episodes to play
    gamma = 0.95
    learning_rate = 1
    epsilon = 1
    teacher_epsilon = 0.8
    eps_decay = 2.5e-5
    q_table_dir = None

    # create tic-tac-toe environment
    env = tic_tac_toe() 
    
    # create a manual agent for human input (so we can against the td-learners)
    human =  manual_agent(name='Human', marker= -1) 

    # create a static agent with hard-coded logic, used to train the td-learners
    agent_B = teacher(name='Teacher', marker=-1, epsilon= teacher_epsilon) 

    # create a Q_learning agent
    agent_A = td_learner(name='Q-Learner', marker=1, 
                         state_dimensions = env.state_dimensions, 
                         learning = 'off-policy', epsilon=epsilon, learn_rate =1, 
                         gamma=gamma, q_table_dir = q_table_dir) 

    # create a SARSA agent
    sarsa_learner = td_learner(name='SARSA-Learner', marker=1, 
                               state_dimensions = env.state_dimensions, 
                               learning = 'on-policy', epsilon=epsilon, learn_rate =1, 
                                gamma=gamma, q_table_dir = q_table_dir) 

    
    # train Q-learner agent, plot perf and output the trained q look-up dict
    trained_q_learner = train_or_play(env, agent_A, agent_B, num_episodes, 
                                     eps_decay, window, False)

    # train SARSA agent, plot perf and output the trained q look-up dict
    trained_sars_learner = train_or_play(env, sarsa_learner, agent_B, num_episodes, 
                                         eps_decay, window, False)  
  
    # play with Q-learning agent: setting td-learner epsilon to 0 
    # to play against the SARSA agent, change the third parameter below with
    # trained_sars_learner
    train_or_play(env, trained_q_learner, human, 5, 0, 1, True)     

    
    
