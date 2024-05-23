import numpy as np
from utils import calculate_observation_buffer_shape

from replay_buffers.base_replay_buffer import BaseReplayBuffer


class NFSPReservoirBuffer(BaseReplayBuffer):
    def __init__(
        self,
        observation_dimensions,
        max_size: int,
        batch_size=32,
    ):
        self.observation_dimensions = observation_dimensions
        super().__init__(max_size=max_size, batch_size=batch_size)

    def store(self, observation, target_policy, id=None):
        self.observation_buffer[self.pointer] = observation
        self.target_policy_buffer[self.pointer] = target_policy
        self.size = min(self.size + 1, self.max_size)
        self.pointer = (self.pointer + 1) % self.max_size

    def sample(self):
        # http://erikerlandson.github.io/blog/2015/11/20/very-fast-reservoir-sampling/
        assert len(self) >= self.batch_size
        observation_reservoir = self.observation_buffer[: self.batch_size]
        target_policy_reservoir = self.target_policy_buffer[: self.batch_size]
        threshold = self.batch_size * 4
        index = self.batch_size
        while (index < len(self)) and (index <= threshold):
            i = np.random.randint(0, index)
            if i < self.batch_size:
                observation_reservoir[i] = self.observation_buffer[index]
                target_policy_reservoir[i] = self.target_policy_buffer[index]
            index += 1

        while index < len(self):
            p = float(self.batch_size) / index
            u = np.random.rand()
            g = np.floor(np.log(u) / np.log(1 - p))
            index = index + int(g)
            if index < len(self):
                i = np.random.randint(0, self.batch_size - 1)
                observation_reservoir[i] = self.observation_buffer[index]
                target_policy_reservoir[i] = self.target_policy_buffer[index]
            index += 1

        return dict(
            observations=observation_reservoir,
            targets=target_policy_reservoir,
        )

    def clear(self):
        observation_buffer_shape = calculate_observation_buffer_shape(
            self.max_size, self.observation_dimensions
        )
        self.observation_buffer = np.zeros(observation_buffer_shape, dtype=np.float32)
        self.target_policy_buffer = np.zeros(self.max_size, dtype=np.int32)
        self.size = 0
        self.pointer = 0
