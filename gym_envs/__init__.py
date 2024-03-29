from gymnasium.envs.registration import register

register(
    id="gym_envs/Connect4-v0",
    entry_point="gym_envs.envs:Connect4Env",
    max_episode_steps=300,
    reward_threshold=1.0,
    kwargs={"size": (6, 7), "win_length": 4},
)


register(
    id="gym_envs/TicTacToe-v0",
    entry_point="gym_envs.envs:TicTacToeEnv",
    max_episode_steps=300,
    reward_threshold=1.0,
    kwargs={"size": 3, "win_length": 3},
)


register(
    id="gym_envs/GridWorld-v0",
    entry_point="gym_envs.envs:TicTacToeEnv",
    max_episode_steps=300,
    reward_threshold=1.0,
    kwargs={"size": 3, "win_length": 3},
)
