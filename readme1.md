# Reinforcement Learning for Board Games

RL for board games is a project under the McGill AI Lab, dedicated to developing and advancing reinforcement learning (RL) models aimed at mastering board games.
By tackling state representation, action space modeling, and multi-agent RL, the team aims to create RL agents capable learning to play common board games.
All the source code for the project can be found in the GitHub repository [epicgamer17/rl-research](https://github.com/epicgamer17/rl-research)

## Project Focus

The project's central goal is to create RL agents that can master both simple and complex board games.
While early successes include solving classic control environments like Cartpole and Mountain Car,
the primary focus now is on board game environments such as Tic-Tac-Toe, Checkers, Connect Four, with
future ambitions include creating RL bots to pioneer solutions for more complex games like Scrabble, Risk, Monopoly, and Catan.

Key challenges involve designing efficient state representations, modeling large action spaces,
and training multi-agent systems. The team actively reads and recreates RL papers,
encouraging collaboration and iterative learning, with an emphasis on adapting and
expanding advanced models such as AlphaZero, MuZero, and NFSP to these board game challenges.

## Implemented Algorithms

The repository includes a range of RL algorithms, such as DQN-based architectures
(Double DQN, Dueling DQN, Rainbow DQN) and Actor-Critic methods like A2C and PPO.
More advanced models, including AlphaZero, MuZero, and NFSP, provide a foundation
for tackling complex strategy games. The modular design allows for experimentation
with different configurations, fostering innovation and learning.

![](./figs/Rainbow_ClassicControl_Acrobot-v1-episode-4.mp4)
![](./figs/Rainbow_ClassicControl_Acrobot-v1-episode-79.mp4)
![](./figs/Rainbow_ClassicControl_Acrobot-v1-episode-154.mp4)

![](./figs/Rainbow_ClassicControl_CartPole-v1-episode-4.mp4)
![](./figs/Rainbow_ClassicControl_CartPole-v1-episode-79.mp4)
![](./figs/Rainbow_ClassicControl_CartPole-v1-episode-154.mp4)

## Custom Environments

To support RL agent development, the project provides custom OpenAI Gym environments tailored to board games:

- **Tic Tac Toe**: A classic two-player game for testing basic RL models.
- **Connect 4**: A strategic game that introduces complexity and planning depth.

Furthermore, to investigate imperfect information and multi-agent settings, we have implementations of 
the following environments

- **LeDuc Hold'em**: A simplified poker variant for studying imperfect information games.
- **Mississippi Marbles**: A custom environment with unique dynamics and strategies.

These environments allow for iterative training and evaluation, ensuring that agents can generalize across different strategic challenges.

## Future Directions

The most exciting aspect of the project lies in its forward-looking roadmap:

- **Advanced Board Game Mastery**: Building agents for games like Risk and Catan to explore the applications of RL agents in long-term strategic planning and dynamic, multi-agent scenarios

- **Meta-Learning and Self-Play**: Enhancing learning efficiency by implementing meta-learning techniques and improving self-play dynamics.

The project's ongoing commitment to theoretical learning and experimentation makes it a unique platform for students, researchers, and enthusiasts alike. By combining cutting-edge algorithms with practical applications, this repository aims to push the boundaries of reinforcement learning in board game environments.

In summary, [epicgamer17/rl-research](https://github.com/epicgamer17/rl-research) stands as a dynamic and ambitious project under the McGill AI Lab, committed to advancing RL techniques for mastering board games, with a clear vision for future innovation and collaboration.