from gymnasium.envs.registration import register

register(
    id="environments/TicTacToe-v0",
    entry_point="environments.tictactoe:TicTacToe",
    max_episode_steps=9,
)
