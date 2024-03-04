from gym.envs.registration import register

register(
    id='Tic-Tac-Toc-v0',
    entry_point='environments.tic_tac_toe:TicTacToe',
    max_episode_steps=300,
)