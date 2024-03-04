import numpy as np
from collections import deque
from time import time
import scipy.signal


def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class ReplayBuffer:
    def __init__(self, observation_dimensions, max_size: int, batch_size: int):
        # self.observation_buffer = np.zeros((max_size,) + observation_dimensions, dtype=np.float32)
        # self.next_observation_buffer = np.zeros((max_size,) + observation_dimensions, dtype=np.float32)
        observation_buffer_shape = []
        observation_buffer_shape += [max_size]
        observation_buffer_shape += list(observation_dimensions)
        observation_buffer_shape = list(observation_buffer_shape)
        self.observation_buffer = np.zeros(observation_buffer_shape, dtype=np.float32)
        self.action_probabilities_buffer = np.zeros(max_size, dtype=np.int32)
        self.reward_buffer = np.zeros(max_size, dtype=np.float32)
        self.pointer, self.trajectory_start_index = 0, 0

        self.max_size = max_size
        self.batch_size = batch_size
        self.size = 0

    def store(self, observation, action_probabilities_buffer, reward):
        self.observation_buffer[self.pointer] = observation
        self.action_probabilities_buffer[self.pointer] = action_probabilities_buffer
        self.reward_buffer[self.pointer] = reward

        self.pointer = (self.pointer + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self):
        indices = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(
            observations=self.observation_buffer[indices],
            action_probabilities=self.action_probabilities_buffer[indices],
            rewards=self.reward_buffer[indices],
        )

    def __len__(self):
        return self.size
