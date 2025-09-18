from ast import Tuple
from calendar import c
import numpy as np
from sympy import N
from replay_buffers.alphazero_replay_buffer import AlphaZeroReplayBuffer
from replay_buffers.base_replay_buffer import (
    BaseGameReplayBuffer,
    BaseReplayBuffer,
    Game,
)
from replay_buffers.segment_tree import MinSegmentTree, SumSegmentTree


class MuZeroReplayBuffer(BaseReplayBuffer):
    def __init__(
        self,
        observation_dimensions,
        observation_dtype: type,
        max_size: int,
        batch_size: int,
        n_step: int,
        num_unroll_steps: int,
        gamma: float,
        has_intermediate_rewards: bool,
        max_priority: float = 1.0,
        alpha: float = 0.6,
        beta: float = 0.4,
        # epsilon=0.01,
    ):
        assert alpha >= 0 and alpha <= 1
        assert beta >= 0 and beta <= 1
        assert n_step >= 1
        assert gamma >= 0 and gamma <= 1

        self.observation_dimensions = observation_dimensions
        self.observation_dtype = observation_dtype

        self.n_step = n_step
        self.unroll_steps = num_unroll_steps
        self.gamma = gamma
        self.has_intermediate_rewards = has_intermediate_rewards

        self.initial_max_priority = max_priority
        self.alpha = alpha
        self.beta = beta
        # self.epsilon = epsilon

        print("Warning: for board games it is recommnded to have n_step >= game length")

        super().__init__(max_size=max_size, batch_size=batch_size)

    def store_position(self, game: Game, position: int, priority: float = None):
        if priority is None:
            priority = self.max_priority
        self.max_priority = max(
            self.max_priority, priority
        )  # could remove and clip priorities in experience replay isntead

        self.max_priority = max(
            self.max_priority, priority
        )  # could remove and clip priorities in experience replay isntead

        # print("Storing position", position + 1, "of game of length", len(game))
        # print(self.observation_buffer.shape)
        self.observation_buffer[self.pointer] = game.observation_history[position]
        self.info_buffer[self.pointer] = game.info_history[position]

        values, policies, rewards, actions = self._get_n_step_info(
            position,
            game.value_history,
            game.policy_history,
            game.rewards,
            game.action_history,
            game.info_history,
            self.unroll_steps,
            self.n_step,
        )
        self.n_step_values_buffer[self.pointer] = values
        self.n_step_policies_buffer[self.pointer] = policies
        self.n_step_rewards_buffer[self.pointer] = rewards
        self.n_step_actions_buffer[self.pointer] = actions

        # print("Priority", priority)
        self.sum_tree[self.tree_pointer] = priority**self.alpha
        self.min_tree[self.tree_pointer] = priority**self.alpha
        self.tree_pointer = (self.tree_pointer + 1) % self.max_size
        self.pointer = (self.pointer + 1) % self.max_size

    def store(self, game: Game):
        for i in range(len(game)):
            self.store_position(game, i)
            if self.size < self.max_size:
                self.size += 1

    def sample(self):
        indices: list[int] = self._sample_proportional()
        # print("Sampled indices", indices)
        weights = np.array([self._calculate_weight(i) for i in indices])
        # print("Weights", weights)

        samples = self.sample_from_indices(indices)
        # print("Sampled n step samples", samples)
        samples.update(dict(weights=weights, indices=indices))
        return samples

    def update_priorities(self, indices: list[int], priorities: list[float], ids=None):
        # necessary for shared replay buffer
        if ids is not None:
            assert len(priorities) == len(ids) == len(indices)
            assert priorities.shape == ids.shape == indices.shape

            for index, id, priority in zip(indices, ids, priorities):
                assert (
                    priority > 0
                ), "Negative priority: {} \n All priorities {}".format(
                    priority, priorities
                )
                assert 0 <= index < len(self)

                if self.id_buffer[index] != id:
                    continue

                self.sum_tree[index] = priority**self.alpha
                self.min_tree[index] = priority**self.alpha
                self.max_priority = max(self.max_priority, priority)
        else:
            assert len(indices) == len(priorities)
            for index, priority in zip(indices, priorities):
                # print("Updating index", index, "with priority", priority)
                assert priority > 0, "Negative priority: {}".format(priority)
                assert 0 <= index < len(self)

                self.sum_tree[index] = priority**self.alpha
                self.min_tree[index] = priority**self.alpha
                self.max_priority = max(
                    self.max_priority, priority
                )  # could remove and clip priorities in experience replay isntead

        # return priorities**self.alpha

    def _sample_proportional(self):
        indices = []
        total_priority = self.sum_tree.sum(0, len(self) - 1)
        priority_segment = total_priority / self.batch_size

        for i in range(self.batch_size):
            a = priority_segment * i
            b = priority_segment * (i + 1)
            upperbound = np.random.uniform(a, b)
            index = self.sum_tree.retrieve(upperbound)
            indices.append(index)

        return indices

    def _calculate_weight(self, index: int):
        min_priority = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (min_priority * len(self)) ** (-self.beta)
        priority_sample = self.sum_tree[index] / self.sum_tree.sum()
        weight = (priority_sample * len(self)) ** (-self.beta)
        weight = weight / max_weight

        return weight

    def _get_n_step_info(
        self,
        index: int,
        values: list,
        policies: list,
        rewards: list,
        actions: list,
        infos: list,
        num_unroll_steps: int,
        n_step: int,
    ):
        n_step_values = []
        n_step_rewards = []
        n_step_policies = []
        n_step_actions = []
        for current_index in range(index, index + num_unroll_steps + 1):
            bootstrap_index = current_index + n_step
            # print("bootstrapping")
            # value of current position is the value at the position n_steps away + rewards to get to the n_step position
            if bootstrap_index < len(values):
                if infos[current_index]["player"] == infos[bootstrap_index]["player"]:
                    value = values[bootstrap_index] * self.gamma**n_step
                else:
                    value = -values[bootstrap_index] * self.gamma**n_step
            else:
                value = 0

            # the rewards at this index to the bootstrap index should be added to the value
            for i, reward in enumerate(rewards[current_index:bootstrap_index]):
                # WHAT IS current_index + i + 1 when current index is the last frame?? IS THIS AN ERROR?
                if (
                    infos[current_index]["player"]
                    == infos[current_index + i + 1]["player"]
                ):
                    value += (
                        reward * self.gamma**i
                    )  # pytype: disable=unsupported-operands
                else:
                    value -= (
                        reward * self.gamma**i
                    )  # pytype: disable=unsupported-operands

            # target reward is the reward before the ones added to the value
            if current_index > 0 and current_index <= len(rewards):
                if self.has_intermediate_rewards:
                    last_reward = rewards[current_index - 1]
                else:
                    # print(
                    #     "Warning: for games with no intermediate rewards n_step should be >= game length"
                    # )
                    value += rewards[current_index - 1]
                    last_reward = None
            else:
                last_reward = None  # self absorbing state

            if current_index < len(values):
                n_step_values.append(value)
                n_step_rewards.append(last_reward)
                n_step_policies.append(policies[current_index])
                n_step_actions.append(actions[current_index])
            else:
                n_step_values.append(0)
                n_step_rewards.append(last_reward)
                n_step_policies.append([])  # self absorbing state
                n_step_actions.append(None)  # self absorbing state

        # print(actions[index : index + num_unroll_steps])
        # print(n_step_actions[:-1])
        return (
            n_step_values,  # [initial value, recurrent values]
            n_step_policies,  # [initial policy, recurrent policies]
            n_step_rewards,  # [initial reward (0), recurrent rewards] initial reward is useless like the first last action, but we ignore it in the learn function
            n_step_actions[:-1],  # [recurrent actions, extra action]
        )  # remove the last actions, as there should be one less action than other stuff

    def set_beta(self, beta: float):
        self.beta = beta

    def clear(self):
        self.size = 0
        self.pointer = 0
        self.max_priority = self.initial_max_priority  # (initial) priority
        self.tree_pointer = 0

        self.observation_buffer = np.zeros(
            (self.max_size,) + self.observation_dimensions, dtype=self.observation_dtype
        )
        self.info_buffer = np.zeros((self.max_size,), dtype=np.object_)
        # TODO MAKE THESE WORK TO BE THE CORRECT DTYPE INSTEAD OF STORING LISTS AND THEY WOULD BE OF LENGTH unroll_steps
        self.n_step_rewards_buffer = np.zeros(
            (self.max_size,),
            dtype=np.object_,
        )
        self.n_step_policies_buffer = np.zeros(
            (self.max_size,),
            dtype=np.object_,
        )
        self.n_step_values_buffer = np.zeros((self.max_size,), dtype=np.object_)
        self.n_step_actions_buffer = np.zeros(
            (self.max_size,),
            dtype=np.object_,
        )

        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def sample_from_indices(self, indices: list[int]):
        return dict(
            observations=self.observation_buffer[indices],
            rewards=self.n_step_rewards_buffer[indices],
            policy=self.n_step_policies_buffer[indices],
            values=self.n_step_values_buffer[indices],
            actions=self.n_step_actions_buffer[indices],
            infos=self.info_buffer[indices],
        )
