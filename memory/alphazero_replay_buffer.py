import numpy as np
from collections import deque
from time import time
import scipy.signal


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
        self.overflow_count = 0
        self.overflow_reward = 0
        self.max_size = max_size
        self.batch_size = batch_size
        self.size = 0

    def store(self, observation, action_probabilities_buffer, reward):
        if self.size < self.max_size:
            self.observation_buffer[self.pointer] = observation
            self.action_probabilities_buffer[self.pointer] = action_probabilities_buffer
            self.reward_buffer[self.pointer] = reward

            self.pointer = (self.pointer + 1) % self.max_size
            self.size = min(self.size + 1, self.max_size)
        else:
            self.overflow_count += 1
            self.overflow_reward = reward

    def sample(self):
        indices = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(
            observations=self.observation_buffer[indices],
            action_probabilities=self.action_probabilities_buffer[indices],
            rewards=self.reward_buffer[indices],
        )

    def update_reward(self, game_start_index):
        # Update the reward buffer with the discounted cumulative sums of the rewards
        if self.size < self.max_size:
            reward = self.reward_buffer[self.size - 1]
        else:
            if self.overflow_count % 2 == 0:
                reward = self.overflow_reward
            else:
                reward = -self.overflow_reward
        updated_rewards = np.empty((self.max_size - game_start_index,), int)
        updated_rewards[::2] = 1 * reward
        updated_rewards[1::2] = -1 * reward
        updated_rewards = np.flip(updated_rewards)
        self.reward_buffer[game_start_index:] = updated_rewards

    def clear(self):
        self.observation_buffer = np.zeros(
            (self.max_size,) + self.observation_buffer.shape[1:], dtype=np.float32
        )
        self.action_probabilities_buffer = np.zeros(self.max_size, dtype=np.int32)
        self.reward_buffer = np.zeros(self.max_size, dtype=np.float32)
        self.pointer, self.trajectory_start_index = 0, 0
        self.size = 0
        self.overflow_count = 0
        self.overflow_reward = 0

    def __len__(self):
        return self.size
