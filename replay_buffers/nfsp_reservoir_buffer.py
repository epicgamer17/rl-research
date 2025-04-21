import numpy as np

from replay_buffers.base_replay_buffer import BaseReplayBuffer

from utils import augment_board


class NFSPReservoirBuffer(BaseReplayBuffer):
    def __init__(
        self,
        observation_dimensions,
        observation_dtype: np.dtype,
        max_size: int,
        num_actions: int,
        batch_size: int = 32,
        compressed_observations: bool = False,
    ):
        self.observation_dimensions = observation_dimensions
        self.observation_dtype = observation_dtype
        self.num_actions = num_actions
        super().__init__(
            max_size=max_size,
            batch_size=batch_size,
            compressed_observations=compressed_observations,
        )
        self.add_calls = 0

    def store(
        self,
        observation,
        info: int,
        target_policy: list[int],
        id=None,
    ):
        """
        Store a transition in the replay buffer.
        :param observation: the current observation
        :param target_policy: the target policy for the current observation, in this case it is of type list[int] since it will be a one-hot encoded vector of the action selected by the best agent network
        :param id: the id of the transition
        """
        if self.size < self.max_size:
            self.observation_buffer[self.add_calls] = observation
            self.info_buffer[self.add_calls] = info
            self.target_policy_buffer[self.add_calls] = target_policy
            self.size = min(self.size + 1, self.max_size)
        else:
            # RESERVOIR ADDING FOR NEW OBSERVATIONS
            # if max size is reached, we add the new observation with a probability of max_size / (add_calls + 1)
            # then the idx of the new observation is a random integer between 0 and max_size
            if np.random.rand() <= self.max_size / (self.add_calls + 1):
                idx = np.random.randint(0, self.max_size) # self.max_size excluded and 0 included
                self.observation_buffer[idx] = observation
                self.info_buffer[idx] = info
                self.target_policy_buffer[idx] = target_policy
        self.add_calls += 1

    def sample(self, type="random"):
        # http://erikerlandson.github.io/blog/2015/11/20/very-fast-reservoir-sampling/
        # assert len(self) >= self.batch_size
        if self.size < self.batch_size:
            return dict(
                observations=self.observation_buffer[: self.size],
                infos=self.info_buffer[: self.size],
                targets=self.target_policy_buffer[: self.size],
            )
        indices = np.random.choice(len(self), self.batch_size, replace=False)
        if type=="random":
            return dict(
                observations=self.observation_buffer[indices],
                infos=self.info_buffer[indices],
                targets=self.target_policy_buffer[indices],
            )
        # elif type=="reservoir":
        #     index = self.batch_size
        #     # else reservoir sampling with the following alg : store first batch_size elements
        #     obeservation_reservoir = self.observation_buffer[: self.batch_size]
        #     info_reservoir = self.info_buffer[: self.batch_size]
        #     target_policy_reservoir = self.target_policy_buffer[: self.batch_size]
        #     for i in len(self.size):
            # the for each new observation, with probability batch_size/index, replace the random uniform over k elements with the new observation


        # for each new 
        # observation_reservoir = self.observation_buffer[: self.batch_size]
        # info_reservoir = self.info_buffer[: self.batch_size]
        # target_policy_reservoir = self.target_policy_buffer[: self.batch_size]
        # threshold = self.batch_size * 4
        # index = self.batch_size
        # indices = []
        # while (index < len(self)) and (index <= threshold):
        #     i = np.random.randint(0, index)
        #     if i < self.batch_size:
        #         observation_reservoir[i] = self.observation_buffer[index]
        #         info_reservoir[i] = self.info_buffer[index]
        #         target_policy_reservoir[i] = self.target_policy_buffer[index]
        #         indices.append(index)
        #     index += 1

        # while index < len(self):
        #     p = float(self.batch_size) / index
        #     u = np.random.rand()
        #     g = np.floor(np.log(u) / np.log(1 - p))
        #     index = index + int(g)
        #     if index < len(self):
        #         i = np.random.randint(0, self.batch_size - 1)
        #         observation_reservoir[i] = self.observation_buffer[index]
        #         info_reservoir[i] = self.info_buffer[index]
        #         target_policy_reservoir[i] = self.target_policy_buffer[index]
        #         indices.append(index)
        #     index += 1
        # print(indices)
        # return dict(
        #     observations=observation_reservoir,
        #     infos=info_reservoir,
        #     targets=target_policy_reservoir,
        # )

    def clear(self):
        if self.compressed_observations:
            self.observation_buffer = np.zeros(self.max_size, dtype=np.object_)
        else:
            observation_buffer_shape = (self.max_size,self.observation_dimensions)
            self.observation_buffer = np.zeros(
                observation_buffer_shape, dtype=self.observation_dtype
            )
        self.info_buffer = np.zeros(self.max_size, dtype=int)
        self.target_policy_buffer = np.zeros(
            (self.max_size, self.num_actions), dtype=np.float64
        )
        self.size = 0
        self.add_calls = 0
