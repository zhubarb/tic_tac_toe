# tic_tac_toe
-----------
TASK DESCRIPTION:
-----------
Your task is to create an agent that learns to play a game of tic-tac-toe. It's up to you to decide the training method:

- implementing a teacher that knows the optimal strategy

- bootstrapping two random agents against themselves

- train the agent yourself and store its state for us to load
or any other approach. However the trained agent should learn the value or action-value function or policy for each state purely from game play, rather than from knowing the rules and heuristics of the game. Code should be written in C++. If you prefer to use a different language, please check with us first.

Use Q-learning and Sarsa to solve the problem. Plot a reward per episode graph for both algorithms.

SARSA http://webdocs.cs.ualberta.ca/~sutton/book/ebook/node64.html
Q-learning http://webdocs.cs.ualberta.ca/~sutton/book/ebook/node65.html

We will assign high weight to good code documentation and style. Please include unit tests in your project. Only spare time should be spent on presentation of game play, text output to terminal is sufficient. Also we expect the agent to be pretty good at the game.

-----------
SOLUTION:
-----------
In this project, I have separately trained an on-policy (SARSA) and an off-policy (Q-Learning) Temporal Difference learner, using a teacher that knows a (nearly) optimal tic-tac-toe gameplay policy.

Both for on-policy and off-policy learners, I have plotted the moving averages of rewards per training episode and uploaded them as .png files under the names: 'SARSA_rewards_against_teacher.png' and 'QLearning_rewards_against_teacher.png', respectively. The increase in agent performance is evident, the reason for this is two-fold. First, the agent is indeed updating its q-value table per state and action and learning how to play better. Second, I anneal the agents' epsilon gradually so the agent transforms from being a fully explorative to fully exploitative in the end.

Observing both plots, the moving average rewards do not reach high values, however this is mainly due to the fact that the agents play against a (nearly) optmial 'teacher' who does random stuff half of the time (teacher object itself also has an epsilon parameter and I set it to 0.5 for both training sessions). Both images are reproducible, making use of the uploaded python files. 

-----------
REPO CONTENTS:
-----------
The repository contains the following files:

- README.md: this very file.

- SARSA_rewards_against_teacher.png: The 1000-game moving averages of the on-policy (SARSA) agent's rewards. I plot the rewards for games where the agent starts (blue) and games where the teacher starts (green) separately. 

- QLearning_rewards_against_teacher.png: The 1000-game moving averages of the off-policy (Q-Learning) agent's rewards. I plot the rewards for games where the agent starts (blue) and games where the teacher (green) starts separately. 

- env.py: 
	This file that contains the 'tic_tac_toe' class. This is an emulator for a 3*3 tic-tac-toe game. It interacts with an agent class through the .step() method, where it accepts the agent's action and outputs the i.next observation, ii.episode termination indicator, and iii. additional information.

- agent.py: 
	This file contains the abstract 'agent' class, and the three subclasses, that extend the agent class, namely:
	i.   manual_agent: Agent that relies on human input to pick actions. This allows us to play with the teacher or the trained td_learner.
	ii.  teacher: Agent with a hard-coded game-play logic for a 3*3 board configuration.
	iii. td_learner: Epsilon-greedy temporal difference (td) learner that can act off-policy or on-policy depending on how its 'learning' parameter is initialised.

- train_and_play.py
	This file creates an environment, initialises agent(s) and depending on the experimental parameters, allows the user to train a td_learner agent or to play with the teacher or an already trained td-learner.

- unit_tests.py
	This file contains the unit tests that check the environment and agent attributes and methods behave in the intended way.

-----------
NOTES:
-----------
- The teacher object has a (nearly) optimal game-play logic. However, in order to allow the td-learner explore different board configurations, I make the teacher play semi-randomly by entering a parameter name teacher_epsilon.
- In both experimental results I have attached, the td-learner is trained with a decaying epsilon as well. I start training with an epsilon vlaue of 1 (fully random) and anneal it slowly to 0.05 as the training progresses.
- I first attempted minimax (https://en.wikipedia.org/wiki/Minimax) as the core logic for the teacher. However, even with the alpha-beta pruning, the episode run times were too long. Due to lack of time I had to abandon this approach and manually code a teacher with a (nearly optimal) game strategy.


