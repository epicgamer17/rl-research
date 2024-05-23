class BaseReplayBuffer:
    def __init__(
        self,
        observation_dimensions,
        max_size,
        batch_size=32,
    ):
        self.observation_dimensions = observation_dimensions
        self.max_size = max_size
        self.batch_size = batch_size
        self.size = 0

    def store(self, *args, **kwargs):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError

    def __len__(self):
        return self.size
