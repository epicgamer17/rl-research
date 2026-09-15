from typing import Optional, Sequence
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class NonstationaryArmedBanditsEnv(gym.Env):
    metadata = {"render_fps": 4}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        means: Optional[Sequence[float]] = None,
        std_devs: Optional[Sequence[float]] = None,
        steps: int = 1000,
        num_arms: int = 1000,
        mode: str = "random",
        target_means: Optional[Sequence[float]] = None,
    ):
        super().__init__()
        if means is None:
            means = np.random.uniform(-1.0, 1.0, size=num_arms)
        if std_devs is None:
            std_devs = np.random.uniform(0.0, 1.0, size=len(means))
        if target_means is None:
            target_means = np.random.uniform(-1.0, 1.0, size=len(means))

        assert len(means) == len(std_devs) == len(target_means)
        self.means = np.asarray(means, dtype=np.float32)
        self.std_devs = np.asarray(std_devs, dtype=np.float32)
        self.target_means = np.asarray(target_means, dtype=np.float32)

        self.observation_space = spaces.Discrete(1)
        self.action_space = spaces.Discrete(len(self.means))

        self.total_steps = steps
        self.reward_threshold = float(np.max(self.means) * self.total_steps)

        if getattr(self, "spec", None) is not None:
            self.spec.reward_threshold = self.reward_threshold

        self.mode = mode
        self.increments = (self.target_means - self.means) / self.total_steps
        assert render_mode is None

    def _get_obs(self):
        return 0

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action: int):
        self.current_step += 1
        reward = float(
            self.np_random.normal(self.means[action], self.std_devs[action])
        )
        if self.mode == "random":
            self.means += self.np_random.uniform(-1.0, 1.0, size=len(self.means))
        elif self.mode == "linear":
            self.means += self.increments

        observation = self._get_obs()
        info = self._get_info()
        truncated = self.current_step >= self.total_steps
        return observation, reward, False, truncated, info

    def render(self):
        pass

    def close(self):
        pass

