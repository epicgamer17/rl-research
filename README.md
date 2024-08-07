RL recreations and implementation from papers

Includes:
1. DQN
2. Double DQN
3. Prioritized Experience Replay DQN 
4. Deuling DQN
5. Noisy DQN 
6. N Step DQN
7. Categorical DQN
8. Rainbow DQN
These can be used by passing in the correct config into RainbowAgent (You can also mix and match these by creating your own configs)
9. Ape-X
10. Neural Ficticious Self Play (NFSP)
NFSP allows traditional RL agents to work well on imperfect information games and multi agent environments. It also can be used to train Rainbow on multi agent games by passing in an anticipatory param of 1.0, this should really only be used for deterministic games though, like Tic Tac Toe or Connect 4. 
11. PPO 
12. AlphaZero
13. MuZero


Envs we have implimented: 
1. Tic Tac Toe
2. CartPole
3. Connect 4 
4. Mississippi Marbles
5. LeDuc Holdem


Some envs we want to test in the future:
1. Chess
2. Catan
3. Go
4. Shogi
5. Risk
6. Monopoly 
7. Starcraft
8. Clash Royale
9. RL Card (Card Games): https://rlcard.org/ https://github.com/datamllab/rlcard 
    Black Jack
    Leduc Hold'em
    Limit Texas Hold'em
    Dou Dizhu
    Simple Dou Dizhu 
    Mahjong 
    No-limit Texas Hold'em 
    UNO 
    Gin Rummy 
    Bridge
10. Eclipse Sumo (Traffic Simulation): https://eclipse.dev/sumo/about/ https://github.com/AndreaVidali/Deep-QLearning-Agent-for-Traffic-Signal-Control 
11. Any Trading (Simple): https://github.com/AminHP/gym-anytrading
12. MTSIM Trading (Complex): https://github.com/AminHP/gym-mtsim 
13. TensorTrade: https://www.tensortrade.org/en/latest/examples/train_and_evaluate_using_ray.html https://github.com/tensortrade-org/tensortrade?tab=readme-ov-file 
14. Atari 57: https://gymnasium.farama.org/environments/atari/ 
15. MineCraft: https://minerl.io/ 
16. Racing: https://aws.amazon.com/deepracer/ 
17. Robo Sumo: https://github.com/openai/robosumo 
18. Unity ML Agents: https://github.com/Unity-Technologies/ml-agents 
19. Multi Agent Emergence Environements: https://github.com/openai/multi-agent-emergence-environments/tree/master/examples 
20. All Open AI Gym Environments: https://gymnasium.farama.org/
    Classic Control
    Box 2D
    Toy Text
    MuJoCo
    Atari
21. All Open Spiel Environments: https://github.com/google-deepmind/open_spiel?tab=readme-ov-file
More at: https://github.com/clvrai/awesome-rl-envs?tab=readme-ov-file 



Tournaments/Challenges:
1. Battle Snake: https://play.battlesnake.com/ 
2. >_Terminal: https://terminal.c1games.com/ 
3. Lux AI: https://www.kaggle.com/c/lux-ai-2021 
4. Russian AI Cup: https://russianaicup.ru/ 
5. Coliseum: https://www.coliseum.ai/ 
6. Code Cup: https://www.codecup.nl/intro.php 
7. IEEE Conference on Games: https://2023.ieee-cog.org/ 


Some useful papers:
1. Muzero: https://arxiv.org/pdf/1911.08265.pdf
2. Rainbow: https://arxiv.org/pdf/1710.02298.pdf
3. Revisiting Rainbow: https://arxiv.org/pdf/2011.14826.pdf 
4. AlphaZero: https://arxiv.org/pdf/1712.01815.pdf 
5. Policy Value Alignment: https://arxiv.org/pdf/2301.11857.pdf 
6. A Disciplined Approach to Hyperparameters Part 1: https://arxiv.org/pdf/1803.09820.pdf
7. High Performance Algorithms for Turn Based Games Using Deep Learning: https://www.scitepress.org/Papers/2020/89561/89561.pdf 
8. KataGo: https://arxiv.org/pdf/2008.10080.pdf https://github.com/lightvector/KataGo/tree/master 
9. Never Give Up: https://arxiv.org/pdf/2002.06038.pdf 
10. Agent 57: https://arxiv.org/pdf/2003.13350.pdf
11. MEME: https://arxiv.org/pdf/2003.13350.pdf 
12. GDI: https://arxiv.org/pdf/2106.06232.pdf <- not used but interesting idea
13. Prioritized Experience Replay: https://arxiv.org/pdf/1511.05952.pdf 
14. PPO: https://arxiv.org/pdf/1707.06347.pdf
15. What Matters in On Policy RL: https://arxiv.org/pdf/2006.05990.pdf 
16. Population Based Training: https://arxiv.org/pdf/1711.09846.pdf <- not used but interesting idea for the future
17. RL Card: https://arxiv.org/abs/1910.04376
18. NFSP https://arxiv.org/pdf/1603.01121
19. CFR: https://proceedings.neurips.cc/paper/2007/file/08d98638c6fcd194a4b1e6992063e944-Paper.pdf 
20: Deep CFR: https://arxiv.org/pdf/1811.00164


To Look Into: 
1. Muesli 
2. DreamerV3
3. R2D2
4. NGU 
5. Agent 57
6. CFR (For imperfect information)
7. DeepCFR (For imperfect information)
8. StarCraft League 
9. Meta Learning 
10. World Models
