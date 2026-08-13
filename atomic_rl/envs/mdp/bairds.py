"""
Baird's Counterexample MDP (Baird 1995, Sutton & Barto 2nd Ed. Ch. 11.2).

Illustrates the 'Deadly Triad' of Reinforcement Learning:
1. Off-Policy Sampling (behavior policy mu vs target policy pi)
2. Function Approximation (linear features phi)
3. Bootstrapping (Temporal Difference learning)

Semi-gradient TD(0) mathematically diverges on this MDP,
while Gradient TD algorithms (GTD0, TDC) converge to V*(s) = 0.
"""

import torch


import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces


class BairdsCounterexampleEnv(gym.Env):
    """
    Baird's 7-state counterexample environment.
    States 0..5 (upper states) have feature vector [1, 0, ..., 2 at index i+1, ... 0].
    State 6 (lower state) has feature vector [2, 0, 0, 0, 0, 0, 0, 1].
    """

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.num_states = 7
        self.num_features = 8
        self.state = 0

        self.observation_space = spaces.Box(
            low=0.0, high=2.0, shape=(8,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)  # 0: dashed, 1: solid

        # Feature Matrix Phi (7 x 8)
        self.phi_matrix = torch.tensor(
            [
                [1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0],
                [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = int(self.np_random.integers(0, 7))
        return self._get_features(self.state), {"state": self.state}

    def step(self, action=None):
        if action is None:
            # Behavior policy mu: Solid (prob 1/7), Dashed (prob 6/7)
            is_solid = self.np_random.random() < (1.0 / 7.0)
        else:
            is_solid = action == 1

        if is_solid:
            self.state = 6
            rho = 7.0  # pi(solid)/mu(solid) = 1.0 / (1/7) = 7.0
        else:
            self.state = int(self.np_random.integers(0, 6))
            rho = 0.0  # pi(dashed)/mu(dashed) = 0.0 / (6/7) = 0.0

        reward = 0.0
        terminated = False
        truncated = False

        next_features = self._get_features(self.state)
        info = {"rho": rho, "state": self.state}
        return next_features, reward, terminated, truncated, info

    def _get_features(self, state: int) -> torch.Tensor:
        return self.phi_matrix[state].clone()

