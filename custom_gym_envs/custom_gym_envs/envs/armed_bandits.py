import numpy as np
import tensorflow_probability as tfp

import gymnasium as gym
from gymnasium import spaces


class ArmedBanditsEnv(gym.Env):
    metadata = {"render_fps": 4}

    def __init__(
        self,
        render_mode=None,
        means=[0.1, 0.2, 0.3, 0.4, 0.5],
        std_devs=[0.01, 0.01, 0.01, 0.01, 0.01],
    ):
        assert len(means) == len(std_devs)
        self.distributions = [
            tfp.distributions.Normal(loc=mean, scale=std_dev)
            for mean, std_dev in zip(means, std_devs)
        ]
        self.observation_space = spaces.Discrete(1)  # or 0
        self.action_space = spaces.Discrete(len(means))

        assert render_mode is None

    def _get_obs(self):
        return 0

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)  # might not set tfp seed

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        reward = self.distributions[action].sample()
        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, False, False, info

    def render(self):
        pass

    def _render_frame(self):
        pass

    def close(self):
        pass
