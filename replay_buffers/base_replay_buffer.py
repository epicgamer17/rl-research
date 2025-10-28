import copy

import numpy as np
import torch
from utils import action_mask, numpy_dtype_to_torch_dtype, discounted_cumulative_sums


class BaseReplayBuffer:
    def __init__(
        self,
        max_size: int,
        batch_size: int = None,
        compressed_observations: bool = False,
    ):
        self.max_size: int = max_size
        self.batch_size: int = batch_size if batch_size is not None else max_size
        self.compressed_observations: bool = compressed_observations
        print("Max size:", max_size)

        self.clear()
        assert self.size == 0, "Replay buffer should be empty at initialization"
        assert self.max_size > 0, "Replay buffer should have a maximum size"
        assert self.batch_size > 0, "Replay buffer batch size should be greater than 0"

    def store(self, *args, **kwargs):
        raise NotImplementedError

    def sample(self, *args, **kwargs):
        raise NotImplementedError

    def sample_from_indices(self, indices: list[int]):
        raise NotImplementedError

    def clear(self):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError

    def __len__(self):
        return self.size


class Game:
    def __init__(
        self, num_players: int
    ):  # num_actions, discount=1.0, n_step=1, gamma=0.99
        self.length = 0
        self.observation_history = []
        self.rewards = []
        self.policy_history = []
        self.value_history = []
        self.action_history = []
        self.info_history = []

        self.num_players = num_players

    def append(
        self,
        observation,
        info,
        reward: int = None,
        policy=None,
        value=None,
        action=None,
    ):
        self.observation_history.append(copy.deepcopy(observation))
        self.info_history.append(copy.deepcopy(info))
        if reward is not None:
            self.rewards.append(reward)
        if policy is not None:
            self.policy_history.append(policy)
        if value is not None:
            self.value_history.append(value)
        if action is not None:
            self.action_history.append(action)
        # print("Game info history", self.info_history)

    def set_rewards(self):
        print("Initial Rewards", self.rewards)
        final_reward = self.rewards[-1]
        for i in reversed(range(self.length)):
            self.rewards[i] = (
                final_reward[i % self.num_players]
                # if i % self.num_players == (self.length - 1) % self.num_players
                # else -final_reward
            )
        print("Updated Rewards", self.rewards)

    def __len__(self):
        # SHOULD THIS BE LEN OF ACTIONS INSTEAD???
        # AS THIS ALLOWS SAMPLING THE TERMINAL STATE WHICH HAS NO FURTHER ACTIONS
        return len(self.action_history)


# class BaseGameReplayBuffer(BaseReplayBuffer):
#     def __init__(
#         self,
#         max_size: int,
#         batch_size: int,
#     ):
#         super().__init__(max_size=max_size, batch_size=batch_size)

#     def store(self, game: Game):
#         self.buffer[self.pointer] = copy.deepcopy(game)
#         self.pointer = (self.pointer + 1) % self.max_size
#         self.size = min(self.size + 1, self.max_size)

#     def sample(self):
#         move_sum = float(sum([len(game) for game in self.buffer]))
#         games: list[Game] = np.random.choice(
#             self.buffer,
#             self.batch_size,
#             p=[len(game) / move_sum for game in self.buffer],
#         )

#         return [(game, np.random.randint(len(game))) for game in games]

#     def clear(self):
#         self.buffer: list[Game] = torch.zeros(self.max_size, dtype=torch.object)
#         self.size = 0
#         self.pointer = 0


class BaseDQNReplayBuffer(BaseReplayBuffer):
    def __init__(
        self,
        observation_dimensions: tuple,
        observation_dtype: torch.dtype,
        max_size: int,
        num_actions: int,
        batch_size: int = 32,
        compressed_observations: bool = False,
    ):
        self.observation_dimensions = observation_dimensions
        self.observation_dtype = observation_dtype
        self.num_actions = num_actions
        print(observation_dtype)
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
        next_observation,
        next_info: dict,
        done: bool,
        id=None,
    ):
        # compute n-step return and store
        # self.id_buffer[self.pointer] = id
        self.observation_buffer[self.pointer] = torch.from_numpy(observation)
        self.action_buffer[self.pointer] = action
        self.reward_buffer[self.pointer] = reward
        self.next_observation_buffer[self.pointer] = torch.from_numpy(next_observation)
        self.done_buffer[self.pointer] = done
        # self.info_buffer[self.pointer] = copy.deepcopy(info)
        # self.next_info_buffer[self.pointer] = copy.deepcopy(next_info)
        # self.action_mask_buffer[self.pointer] = action_mask(
        #     self.num_actions, info.get("legal_actions", [])
        # )
        self.next_action_mask_buffer[self.pointer] = action_mask(
            self.num_actions, next_info.get("legal_actions", [])
        )

        self.pointer = (self.pointer + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def clear(self):
        observation_buffer_shape = (self.max_size,) + self.observation_dimensions
        self.observation_buffer = torch.zeros(
            observation_buffer_shape,
            dtype=numpy_dtype_to_torch_dtype(self.observation_dtype),
        )
        self.next_observation_buffer = torch.zeros(
            observation_buffer_shape,
            dtype=numpy_dtype_to_torch_dtype(self.observation_dtype),
        )

        self.action_buffer = torch.zeros(self.max_size, dtype=torch.uint8)
        self.reward_buffer = torch.zeros(self.max_size, dtype=torch.float16)
        self.done_buffer = torch.zeros(self.max_size, dtype=torch.bool)
        # self.info_buffer = torch.zeros(self.max_size, dtype=torch.object)
        # self.next_info_buffer = torch.zeros(self.max_size, dtype=torch.object)
        # self.action_mask_buffer = torch.zeros(
        #     (self.max_size, self.num_actions), dtype=torch.bool
        # )
        self.next_action_mask_buffer = torch.zeros(
            (self.max_size, self.num_actions), dtype=torch.bool
        )
        self.pointer = 0
        self.size = 0

    def sample(self):
        indices = np.random.choice(self.size, self.batch_size, replace=False)

        return dict(
            observations=self.observation_buffer[indices],
            next_observations=self.next_observation_buffer[indices],
            actions=self.action_buffer[indices],
            rewards=self.reward_buffer[indices],
            dones=self.done_buffer[indices],
            # ids=self.id_buffer[indices],
            # info=self.info_buffer[indices],
            # next_info=self.next_info_buffer[indices],
            # action_masks=self.action_mask_buffer[indices],
            next_action_masks=self.next_action_mask_buffer[indices],
        )

    def sample_from_indices(self, indices: list[int]):
        return dict(
            observations=self.observation_buffer[indices],
            next_observations=self.next_observation_buffer[indices],
            actions=self.action_buffer[indices],
            rewards=self.reward_buffer[indices],
            dones=self.done_buffer[indices],
            ids=self.id_buffer[indices],
            # infos=self.info_buffer[indices],
            # next_infos=self.next_info_buffer[indices],
            # action_masks=self.action_mask_buffer[indices],
            next_action_masks=self.next_action_mask_buffer[indices],
        )

    def __check_id__(self, index: int, id: str) -> bool:
        return self.id_buffer[index] == id


class BasePPOReplayBuffer(BaseReplayBuffer):
    def __init__(
        self,
        observation_dimensions,
        observation_dtype: torch.dtype,
        max_size: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        compressed_observations: bool = False,
    ):
        self.observation_dimensions = observation_dimensions
        self.observation_dtype = observation_dtype
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        super().__init__(
            max_size=max_size, compressed_observations=compressed_observations
        )

    def store(
        self,
        observation,
        info: dict,
        action,
        value: float,
        log_probability: float,
        reward: float,
        id=None,
    ):
        self.observation_buffer[self.pointer] = copy.deepcopy(observation)
        self.action_buffer[self.pointer] = action
        self.reward_buffer[self.pointer] = reward
        self.value_buffer[self.pointer] = value
        self.log_probability_buffer[self.pointer] = log_probability
        # self.info_buffer[self.pointer] = copy.deepcopy(info)
        self.action_mask_buffer[self.pointer] = action_mask(
            self.num_actions, info.get("legal_actions", [])
        )

        self.pointer = (self.pointer + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self):
        self.pointer, self.trajectory_start_index = 0, 0
        # advantage_mean = np.mean(self.advantage_buffer)
        # advantage_std = np.std(self.advantage_buffer)
        advantage_mean = torch.mean(
            torch.tensor(self.advantage_buffer, dtype=torch.float32)
        )
        advantage_std = torch.std(
            torch.tensor(self.advantage_buffer, dtype=torch.float32)
        )
        self.advantage_buffer = (self.advantage_buffer - advantage_mean) / (
            advantage_std + 1e-10
        )  # avoid division by zero
        return dict(
            observations=self.observation_buffer,
            actions=self.action_buffer,
            advantages=self.advantage_buffer,
            returns=self.return_buffer,
            log_probabilities=self.log_probability_buffer,
            action_masks=self.action_mask_buffer,
        )

    def clear(self):
        observation_buffer_shape = (self.max_size,) + self.observation_dimensions
        self.observation_buffer = torch.zeros(
            observation_buffer_shape,
            dtype=numpy_dtype_to_torch_dtype(self.observation_dtype),
        )
        self.next_observation_buffer = torch.zeros(
            observation_buffer_shape,
            dtype=numpy_dtype_to_torch_dtype(self.observation_dtype),
        )
        self.action_buffer = torch.zeros(self.max_size, dtype=torch.int8)
        self.reward_buffer = torch.zeros(self.max_size, dtype=torch.float16)
        self.advantage_buffer = torch.zeros(self.max_size, dtype=torch.float16)
        self.return_buffer = torch.zeros(self.max_size, dtype=torch.float16)
        self.value_buffer = torch.zeros(self.max_size, dtype=torch.float16)
        self.log_probability_buffer = torch.zeros(self.max_size, dtype=torch.float16)
        # self.info_buffer = torch.zeros(self.max_size, dtype=torch.object)
        self.action_mask_buffer = torch.zeros(
            (self.max_size, self.num_actions), dtype=torch.bool
        )

        self.pointer = 0
        self.trajectory_start_index = 0
        self.size = 0

    def finish_trajectory(self, last_value: float = 0):
        path_slice = slice(self.trajectory_start_index, self.pointer)
        rewards = torch.cat(
            (
                self.reward_buffer[path_slice],
                torch.tensor([last_value], dtype=torch.float16),
            )
        )
        values = torch.cat(
            (
                self.value_buffer[path_slice],
                torch.tensor([last_value], dtype=torch.float16),
            )
        )

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
