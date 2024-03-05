from gymnasium.envs.registration import register

register(
     id="environments_/TicTacToe-v0",
     entry_point="gym_examples.envs:GridWorldEnv",
     max_episode_steps=9,
)