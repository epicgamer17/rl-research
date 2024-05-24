import numpy as np
from collections import deque
from time import time
import scipy.signal
from utils import calculate_observation_buffer_shape

from replay_buffers.base_replay_buffer import BaseReplayBuffer


def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class PPOReplayBuffer(BaseReplayBuffer):
    def __init__(
        self,
        observation_dimensions,
        max_size: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        self.observation_dimensions = observation_dimensions

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        super().__init__(max_size=max_size)

    def store(
        self,
        observation,
        action,
        value: float,
        log_probability: float,
        reward: float,
    ):
        self.observation_buffer[self.pointer] = observation
        self.action_buffer[self.pointer] = action
        self.reward_buffer[self.pointer] = reward
        self.value_buffer[self.pointer] = value
        self.log_probability_buffer[self.pointer] = log_probability

        self.pointer = (self.pointer + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self):
        self.pointer, self.trajectory_start_index = 0, 0
        advantage_mean = np.mean(self.advantage_buffer)
        advantage_std = np.std(self.advantage_buffer)
        self.advantage_buffer = (self.advantage_buffer - advantage_mean) / (
            advantage_std + 1e-10
        )  # avoid division by zero
        return dict(
            observations=self.observation_buffer,
            actions=self.action_buffer,
            advantages=self.advantage_buffer,
            returns=self.return_buffer,
            log_probabilities=self.log_probability_buffer,
        )

    def clear(self):
        observation_buffer_shape = calculate_observation_buffer_shape(
            self.observation_dimensions, self.max_size
        )
        self.observation_buffer = np.zeros(observation_buffer_shape, dtype=np.float16)
        self.action_buffer = np.zeros(self.max_size, dtype=np.int8)
        self.reward_buffer = np.zeros(self.max_size, dtype=np.float16)
        self.advantage_buffer = np.zeros(self.max_size, dtype=np.float16)
        self.return_buffer = np.zeros(self.max_size, dtype=np.float16)
        self.value_buffer = np.zeros(self.max_size, dtype=np.float16)
        self.log_probability_buffer = np.zeros(self.max_size, dtype=np.float16)
        self.pointer = 0
        self.trajectory_start_index = 0
        self.size = 0

    def finish_trajectory(self, last_value: float = 0):
        path_slice = slice(self.trajectory_start_index, self.pointer)
        rewards = np.append(self.reward_buffer[path_slice], last_value)
        values = np.append(self.value_buffer[path_slice], last_value)

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        self.advantage_buffer[path_slice] = discounted_cumulative_sums(
            deltas, self.gamma * self.gae_lambda
        )
        self.return_buffer[path_slice] = discounted_cumulative_sums(
            rewards, self.gamma
        )[:-1]
        # print(discounted_cumulative_sums(deltas, self.gamma * self.gae_lambda))
        # print(discounted_cumulative_sums(deltas, self.gamma * self.gae_lambda)[:-1])
        # print(self.advantage_buffer)

        self.trajectory_start_index = self.pointer
