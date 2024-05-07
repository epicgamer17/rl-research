import numpy as np


class ReservoirBuffer:
    def __init__(
        self,
        observation_dimensions,
        max_size: int,
        batch_size=32,
    ):
        self.observation_dimensions = observation_dimensions
        self.batch_size = batch_size
        self.observation_buffer = np.empty(
            [max_size, self.observation_dimensions], dtype=np.float32
        )
        self.action_buffer = np.empty(max_size, dtype=np.int32)
        self.size = 0

    def store(self, observation, action, id=None):
        self.observation_buffer[self.size] = observation
        self.action_buffer[self.size] = action
        self.size += 1

    def sample(self):
        # http://erikerlandson.github.io/blog/2015/11/20/very-fast-reservoir-sampling/
        assert len(self) >= self.batch_size
        observation_reservoir = self.observation_buffer[: self.batch_size]
        action_reservoir = self.action_buffer[: self.batch_size]
        threshold = self.batch_size * 4
        index = self.batch_size
        while (index < len(self)) and (index <= threshold):
            i = np.random.randint(0, index)
            if i < self.batch_size:
                observation_reservoir[i] = self.observation_buffer[index]
                action_reservoir[i] = self.action_buffer[index]
            index += 1

        while index < len(self):
            p = float(self.batch_size) / index
            u = np.random.rand()
            g = np.floor(np.log(u) / np.log(1 - p))
            index = index + int(g)
            if index < len(self):
                i = np.random.randint(0, self.batch_size - 1)
                observation_reservoir[i] = self.observation_buffer[index]
                action_reservoir[i] = self.action_buffer[index]
            index += 1

        return dict(
            observations=observation_reservoir,
            actions=action_reservoir,
        )

    def __len__(self):
        return self.size
