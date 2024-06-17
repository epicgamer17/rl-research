import copy

import numpy as np
from utils.utils import discounted_cumulative_sums


class BaseReplayBuffer:
    def __init__(
        self,
        max_size: int,
        batch_size: int = None,
        compressed_observations: bool = False,
    ):
        self.max_size = max_size
        self.batch_size = batch_size if batch_size is not None else max_size
        self.compressed_observations = compressed_observations

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
        self.legal_moves_history = []

        self.num_players = num_players

    def append(
        self,
        observation,
        reward: int,
        policy,
        value=None,
        action=None,
        legal_moves=None,
    ):
        self.observation_history.append(copy.deepcopy(observation))
        self.rewards.append(reward)
        self.policy_history.append(policy)
        self.value_history.append(value)
        self.action_history.append(action)
        self.legal_moves_history.append(legal_moves)
        self.length += 1

    def set_rewards(self):
        print("Initial Rewards", self.rewards)
        final_reward = self.rewards[-1]
        for i in reversed(range(self.length)):
            self.rewards[i] = (
                final_reward
                if i % self.num_players == (self.length - 1) % self.num_players
                else -final_reward
            )
        print("Updated Rewards", self.rewards)

    def __len__(self):
        return self.length


class BaseGameReplayBuffer(BaseReplayBuffer):
    def __init__(
        self,
        max_size: int,
        batch_size: int,
    ):
        super().__init__(max_size=max_size, batch_size=batch_size)

    def store(self, game: Game):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append(game)
        self.size += 1

    def sample(self):
        move_sum = float(sum([len(game) for game in self.buffer]))
        games: list[Game] = np.random.choice(
            self.buffer,
            self.batch_size,
            p=[len(game) / move_sum for game in self.buffer],
        )

        return [(game, np.random.randint(len(game))) for game in games]

    def clear(self):
        self.buffer: list[Game] = []
        self.size = 0


class BaseDQNReplayBuffer(BaseReplayBuffer):
    def __init__(
        self,
        observation_dimensions: tuple,
        observation_dtype: np.dtype,
        max_size: int,
        batch_size: int = 32,
        compressed_observations: bool = False,
    ):
        self.observation_dimensions = observation_dimensions
        self.observation_dtype = observation_dtype
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
        self.id_buffer[self.pointer] = id
        self.observation_buffer[self.pointer] = observation
        self.action_buffer[self.pointer] = action
        self.reward_buffer[self.pointer] = reward
        self.next_observation_buffer[self.pointer] = next_observation
        self.done_buffer[self.pointer] = done
        self.info_buffer[self.pointer] = info
        self.next_info_buffer[self.pointer] = next_info

        self.pointer = (self.pointer + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def clear(self):
        if self.compressed_observations:
            self.observation_buffer = np.zeros(self.max_size, dtype=np.object_)
            self.next_observation_buffer = np.zeros(self.max_size, dtype=np.object_)
        else:
            observation_buffer_shape = (self.max_size,) + self.observation_dimensions
            self.observation_buffer = np.zeros(
                observation_buffer_shape, self.observation_dtype
            )
            self.next_observation_buffer = np.zeros(
                observation_buffer_shape, dtype=self.observation_dtype
            )

        self.id_buffer = np.zeros(self.max_size, dtype=np.object_)
        self.action_buffer = np.zeros(self.max_size, dtype=np.uint8)
        self.reward_buffer = np.zeros(self.max_size, dtype=np.float16)
        self.done_buffer = np.zeros(self.max_size, dtype=np.bool_)
        self.info_buffer = np.zeros(self.max_size, dtype=np.object_)
        self.next_info_buffer = np.zeros(self.max_size, dtype=np.object_)
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
            ids=self.id_buffer[indices],
            info=self.info_buffer[indices],
            next_info=self.next_info_buffer[indices],
        )

    def sample_from_indices(self, indices: list[int]):
        return dict(
            observations=self.observation_buffer[indices],
            next_observations=self.next_observation_buffer[indices],
            actions=self.action_buffer[indices],
            rewards=self.reward_buffer[indices],
            dones=self.done_buffer[indices],
            ids=self.id_buffer[indices],
            infos=self.info_buffer[indices],
            next_infos=self.next_info_buffer[indices],
        )

    def __check_id__(self, index: int, id: str) -> bool:
        return self.id_buffer[index] == id


class BasePPOReplayBuffer(BaseReplayBuffer):
    def __init__(
        self,
        observation_dimensions,
        observation_dtype: np.dtype,
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
        self.observation_buffer[self.pointer] = observation
        self.action_buffer[self.pointer] = action
        self.reward_buffer[self.pointer] = reward
        self.value_buffer[self.pointer] = value
        self.log_probability_buffer[self.pointer] = log_probability
        self.info_buffer[self.pointer] = info

        self.pointer = (self.pointer + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self):
        self.pointer, self.trajectory_start_index = 0, 0
        advantage_mean = np.mean(self.advantage_buffer)
        advantage_std = np.std(self.advantage_buffer)
        self.advantage_buffer = (self.advantage_buffer - advantage_mean) / (
            advantage_std + 1e-10
        )  # avoid division by zero
        return dict(
            observations=self.observation_buffer,
            actions=self.action_buffer,
            advantages=self.advantage_buffer,
            returns=self.return_buffer,
            log_probabilities=self.log_probability_buffer,
            infos=self.info_buffer,
        )

    def clear(self):
        if self.compressed_observations:
            self.observation_buffer = np.zeros(self.max_size, dtype=np.object_)
            self.next_observation_buffer = np.zeros(self.max_size, dtype=np.object_)
        else:
            observation_buffer_shape = (self.max_size,) + self.observation_dimensions
            self.observation_buffer = np.zeros(
                observation_buffer_shape, self.observation_dtype
            )
            self.next_observation_buffer = np.zeros(
                observation_buffer_shape, dtype=self.observation_dtype
            )
        self.action_buffer = np.zeros(self.max_size, dtype=np.int8)
        self.reward_buffer = np.zeros(self.max_size, dtype=np.float16)
        self.advantage_buffer = np.zeros(self.max_size, dtype=np.float16)
        self.return_buffer = np.zeros(self.max_size, dtype=np.float16)
        self.value_buffer = np.zeros(self.max_size, dtype=np.float16)
        self.log_probability_buffer = np.zeros(self.max_size, dtype=np.float16)
        self.info_buffer = np.zeros(self.max_size, dtype=np.object_)

        self.pointer = 0
        self.trajectory_start_index = 0
        self.size = 0

    def finish_trajectory(self, last_value: float = 0):
        path_slice = slice(self.trajectory_start_index, self.pointer)
        rewards = np.append(self.reward_buffer[path_slice], last_value)
        values = np.append(self.value_buffer[path_slice], last_value)

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
