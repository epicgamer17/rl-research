from replay_buffers.base_replay_buffer import BaseReplayBuffer
import numpy as np
from utils.utils import discounted_cumulative_sums
import torch

class A2CReplayBuffer(BaseReplayBuffer):
    def __init__(
        self,
        observation_dimensions,
        observation_dtype: np.dtype,
        max_size: int,
        gamma: float = 0.99,
        compressed_observations: bool = False,
    ):
        self.observation_dimensions = observation_dimensions
        self.observation_dtype = observation_dtype
        self.gamma = gamma
        super().__init__(
            max_size=max_size, compressed_observations=compressed_observations
        )

    def store(
        self,
        value: float,
        log_probability: float,
        reward: float,
        distribution=float,
        id=None,
    ):
        self.reward_buffer[self.size] = reward
        self.value_buffer[self.size] = value
        self.log_probability_buffer[self.size] = log_probability
        self.distribution_buffer[self.size] = distribution

        self.size = min(self.size + 1, self.max_size)

    def sample(self):
        # advantage_mean = np.mean(self.advantage_buffer)
        # advantage_std = np.std(self.advantage_buffer)
        # self.advantage_buffer = (self.advantage_buffer - advantage_mean) / (
        #     advantage_std + 1e-10
        # )  # avoid division by zero
        return dict(
            advantages=self.advantage_buffer[:self.size],
            values=self.value_buffer[:self.size],
            returns=self.return_buffer[:self.size],
            log_probabilities=self.log_probability_buffer[:self.size],
            distributions=self.distribution_buffer[:self.size],
        )

    def clear(self):
        self.reward_buffer = np.zeros(self.max_size, dtype=np.float16)
        self.advantage_buffer = np.zeros(self.max_size, dtype=np.float16)
        self.return_buffer = np.zeros(self.max_size, dtype=np.float16)
        self.value_buffer = np.zeros(self.max_size, dtype=torch.Tensor)
        self.log_probability_buffer = np.zeros(self.max_size, dtype=torch.Tensor)
        self.distribution_buffer = np.zeros(self.max_size, dtype=torch.distributions.Categorical)

        self.size = 0

    def compute_advantage_and_returns(self, last_value: float = 0):
        path_slice = slice(0, self.size)
        rewards = np.append(self.reward_buffer[:self.size], last_value)
        values = np.append(self.value_buffer[:self.size], last_value)

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        self.advantage_buffer[path_slice] = discounted_cumulative_sums(
            deltas, self.gamma 
        )
        self.return_buffer[path_slice] = discounted_cumulative_sums(
            rewards, self.gamma
        )[:-1]
        # print(discounted_cumulative_sums(deltas, self.gamma * self.gae_lambda))
        # print(discounted_cumulative_sums(deltas, self.gamma * self.gae_lambda)[:-1])
        # print(self.advantage_buffer)

