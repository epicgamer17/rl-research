import numpy as np
import torch

from packages.utils.utils.utils import action_mask
from replay_buffers.base_replay_buffer import BaseReplayBuffer

from utils import augment_board
import copy


class NFSPReservoirBuffer(BaseReplayBuffer):
    def __init__(
        self,
        observation_dimensions,
        observation_dtype: torch.dtype,
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

    def store(
        self,
        observation,
        info: dict,
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
            self.observation_buffer[self.add_calls] = copy.deepcopy(observation)
            self.action_mask_buffer[self.add_calls] = action_mask(
                self.num_actions, info.get("legal_actions", [])
            )
            self.target_policy_buffer[self.add_calls] = target_policy
            self.size = min(self.size + 1, self.max_size)
        else:
            idx = np.random.randint(0, self.add_calls + 1)
            if idx < self.max_size:
                self.observation_buffer[idx] = copy.deepcopy(observation)
                self.action_mask_buffer[idx] = action_mask(
                    self.num_actions, info.get("legal_actions", [])
                )
                self.target_policy_buffer[idx] = target_policy
        self.add_calls += 1

    def sample(self):
        # http://erikerlandson.github.io/blog/2015/11/20/very-fast-reservoir-sampling/
        assert len(self) >= self.batch_size
        indices = np.random.choice(len(self), self.batch_size, replace=False)
        return dict(
            observations=self.observation_buffer[indices],
            action_masks=self.action_mask_buffer[indices],
            current_players=self.current_player_buffer[indices],
            targets=self.target_policy_buffer[indices],
        )

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
        observation_buffer_shape = (self.max_size,) + self.observation_dimensions

        print(observation_buffer_shape)
        self.observation_buffer = torch.zeros(
            observation_buffer_shape, dtype=self.observation_dtype
        )
        # self.info_buffer = torch.zeros(self.max_size, dtype=torch.object)
        self.action_mask_buffer = torch.zeros(
            (self.max_size, self.num_actions), dtype=torch.bool
        )

        self.target_policy_buffer = torch.zeros(
            (self.max_size, self.num_actions), dtype=torch.float16
        )
        self.size = 0
        self.add_calls = 0
