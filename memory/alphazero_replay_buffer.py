import numpy as np
from collections import deque
from time import time
import scipy.signal


class ReplayBuffer:
    def __init__(
        self,
        observation_dimensions,
        max_size: int,
        batch_size: int,
        max_game_length: int,
    ):
        # self.observation_buffer = np.zeros((max_size,) + observation_dimensions, dtype=np.float32)
        # self.next_observation_buffer = np.zeros((max_size,) + observation_dimensions, dtype=np.float32)
        observation_buffer_shape = []
        observation_buffer_shape += [max_size]
        observation_buffer_shape += list(observation_dimensions)
        observation_buffer_shape = list(observation_buffer_shape)
        self.observation_buffer = np.zeros(observation_buffer_shape, dtype=np.float32)
        self.action_probabilities_buffer = np.zeros(max_size, dtype=np.int32)
        self.reward_buffer = np.zeros(max_size, dtype=np.float32)

        self.game_observation_buffer = np.zeros(
            list(
                [
                    max_game_length,
                ]
                + list(observation_dimensions)
            ),
            dtype=np.float32,
        )
        self.game_action_probabilities_buffer = np.zeros(
            max_game_length, dtype=np.int32
        )
        self.game_reward_buffer = np.zeros(max_game_length, dtype=np.float32)
        self.game_pointer = 0
        self.max_game_length = max_game_length
        self.game_length = 0

        self.pointer, self.trajectory_start_index = 0, 0
        self.max_size = max_size
        self.batch_size = batch_size
        self.size = 0

    def store(self, observation, action_probabilities_buffer, reward):
        self.game_observation_buffer[self.game_pointer] = observation
        self.game_action_probabilities_buffer[self.game_pointer] = (
            action_probabilities_buffer
        )
        self.game_reward_buffer[self.game_pointer] = reward
        self.game_pointer = (self.game_pointer + 1) % self.max_game_length
        self.game_length = min(self.game_length + 1, self.max_game_length)

    def store_game(self):
        reward = self.game_reward_buffer[self.game_length - 1]
        updated_rewards = np.empty((self.game_length), int)
        updated_rewards[::2] = 1 * reward
        updated_rewards[1::2] = -1 * reward
        updated_rewards = np.flip(updated_rewards)
        self.game_reward_buffer[0:] = updated_rewards

        if self.max_size - self.pointer < self.game_length:
            game_start_index = self.pointer
            game_end_index = self.max_size
            self.observation_buffer[game_start_index:game_end_index] = (
                self.game_observation_buffer[: self.max_size - game_start_index]
            )
            self.action_probabilities_buffer[game_start_index:game_end_index] = (
                self.game_action_probabilities_buffer[
                    : self.max_size - game_start_index
                ]
            )
            self.reward_buffer[game_start_index:game_end_index] = (
                self.game_reward_buffer[: self.max_size - game_start_index]
            )
            self.pointer = 0
            self.size = self.max_size
            self.game_length -= self.max_size - game_start_index
            game_start_index = 0
            game_end_index = self.game_length
            self.observation_buffer[game_start_index:game_end_index] = (
                self.game_observation_buffer[: self.game_length]
            )
            self.action_probabilities_buffer[game_start_index:game_end_index] = (
                self.game_action_probabilities_buffer[: self.game_length]
            )
            self.reward_buffer[game_start_index:game_end_index] = (
                self.game_reward_buffer[: self.game_length]
            )
            self.pointer = self.game_length
            self.size = min(self.size + self.game_length, self.max_size)
        else:
            game_start_index = self.pointer
            game_end_index = game_start_index + self.game_length
            self.observation_buffer[game_start_index:game_end_index] = (
                self.game_observation_buffer[: self.game_length]
            )
            self.action_probabilities_buffer[game_start_index:game_end_index] = (
                self.game_action_probabilities_buffer[: self.game_length]
            )
            self.reward_buffer[game_start_index:game_end_index] = (
                self.game_reward_buffer[: self.game_length]
            )
            self.pointer = (self.pointer + self.game_length) % self.max_size
            self.size = min(self.size + self.game_length, self.max_size)
        self.game_observation_buffer = np.zeros(
            list(
                [
                    self.max_game_length,
                ]
                + list(self.game_observation_buffer.shape[1:])
            ),
            dtype=np.float32,
        )
        self.game_action_probabilities_buffer = np.zeros(
            self.max_game_length, dtype=np.int32
        )
        self.game_reward_buffer = np.zeros(self.max_game_length, dtype=np.float32)
        self.game_pointer = 0

    def sample(self):
        indices = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(
            observations=self.observation_buffer[indices],
            action_probabilities=self.action_probabilities_buffer[indices],
            rewards=self.reward_buffer[indices],
        )

    # def clear(self):
    #     self.observation_buffer = np.zeros(
    #         (self.max_size,) + self.observation_buffer.shape[1:], dtype=np.float32
    #     )
    #     self.action_probabilities_buffer = np.zeros(self.max_size, dtype=np.int32)
    #     self.reward_buffer = np.zeros(self.max_size, dtype=np.float32)
    #     self.pointer, self.trajectory_start_index = 0, 0
    #     self.size = 0
    #     self.overflow_count = 0
    #     self.overflow_reward = 0

    def __len__(self):
        return self.size