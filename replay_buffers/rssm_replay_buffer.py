import numpy as np
from collections import deque

from replay_buffers.base_replay_buffer import BaseDQNReplayBuffer, BaseReplayBuffer


class RSSMReplayBuffer(BaseReplayBuffer):
    def __init__(
        self,
        observation_dimensions: tuple,
        observation_dtype: np.dtype,
        max_size: int,
        batch_size: int = 32,
        batch_length: int = 20,
        compressed_observations: bool = False,
    ):
        self.batch_length = batch_length

        self.observation_dimensions = observation_dimensions
        self.observation_dtype = observation_dtype

        super().__init__(
            max_size=max_size,
            batch_size=batch_size,
            compressed_observations=compressed_observations,
        )

    def store(
        self,
        observation,
        info: dict,
        action,
        reward: float,
        done: bool,
        id=None,
    ):
        self.id_buffer[self.pointer] = id
        self.action_buffer[self.pointer] = action
        self.reward_buffer[self.pointer] = reward
        self.done_buffer[self.pointer] = done
        self.info_buffer[self.pointer] = info
        self.observation_buffer[self.pointer] = observation

        self.pointer = (self.pointer + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self):
        # pick batch_size random indices
        indices = np.random.choice(
            self.size - self.batch_length, self.batch_size, replace=False
        )

        sequnce_indices = np.stack(
            [
                np.arange(indices[i], indices[i] + self.batch_length)
                for i in range(self.batch_size)
            ]
        )

        return dict(
            observations=self.observation_buffer[sequnce_indices],
            actions=self.action_buffer[sequnce_indices],
            rewards=self.reward_buffer[sequnce_indices],
            dones=self.done_buffer[sequnce_indices],
            infos=self.info_buffer[sequnce_indices],
            ids=self.id_buffer[sequnce_indices],
        )

    def clear(self):
        if self.compressed_observations:
            self.observation_buffer = np.zeros(self.max_size, dtype=np.object_)
        else:
            observation_buffer_shape = (self.max_size,) + self.observation_dimensions
            self.observation_buffer = np.zeros(
                observation_buffer_shape,  # self.observation_dtype
            )

        self.id_buffer = np.zeros(self.max_size, dtype=np.object_)
        self.action_buffer = np.zeros(self.max_size, dtype=np.uint8)
        self.reward_buffer = np.zeros((self.max_size, 1), dtype=np.float16)
        self.done_buffer = np.zeros((self.max_size, 1), dtype=np.bool_)
        self.info_buffer = np.zeros(self.max_size, dtype=np.object_)
        self.pointer = 0
        self.size = 0
