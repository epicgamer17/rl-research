from gymnasium.envs.registration import register

register(
    id="TicTacToe-v0",
    entry_point="environments.TicTacToe:TicTacToe",
    max_episode_steps=9,
)
