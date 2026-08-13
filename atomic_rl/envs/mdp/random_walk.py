import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces


class RandomWalkEnv(gym.Env):
    """
    1D Random Walk MDP (Sutton & Barto 2nd Ed. Ch. 6.2).
    """

    metadata = {"render_modes": []}

    def __init__(self, num_states: int = 5, start_state: int = 2):
        super().__init__()
        self.num_states = num_states
        self.start_state = start_state
        self.state = self.start_state

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(num_states,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.start_state
        return self._get_features(self.state), {"state": self.state}

    def step(self, action=None):
        if action is None:
            direction = 1 if self.np_random.random() > 0.5 else -1
        else:
            direction = 1 if action == 1 else -1

        self.state += direction

        terminated = False
        truncated = False
        reward = 0.0

        if self.state >= self.num_states:  # Right terminal
            reward = 1.0
            terminated = True
        elif self.state < 0:  # Left terminal
            reward = 0.0
            terminated = True

        next_features = (
            self._get_features(self.state)
            if not terminated
            else torch.zeros(self.num_states, dtype=torch.float32)
        )

        info = {"state": self.state}
        return next_features, reward, terminated, truncated, info

    def _get_features(self, state: int) -> torch.Tensor:
        phi = torch.zeros(self.num_states, dtype=torch.float32)
        if 0 <= state < self.num_states:
            phi[state] = 1.0
        return phi

