from time import time
import numpy as np
from replay_buffers.base_replay_buffer import BaseDQNReplayBuffer
from replay_buffers.segment_tree import SumSegmentTree, MinSegmentTree
from replay_buffers.fast_sum_tree import FastSumTree
from replay_buffers.n_step_replay_buffer import NStepReplayBuffer


class PrioritizedReplayBuffer(BaseDQNReplayBuffer):
    def __init__(
        self,
        observation_dimensions: tuple,
        observation_dtype: np.dtype,
        max_size: int,
        batch_size: int = 32,
        max_priority: float = 1.0,
        alpha: float = 0.6,
        beta: float = 0.4,
        # epsilon=0.01,
    ):
        assert alpha >= 0 and alpha <= 1
        assert beta >= 0 and beta <= 1

        self.initial_max_priority = max_priority
        super().__init__(
            observation_dimensions, observation_dtype, max_size, batch_size
        )

        self.alpha = alpha  # Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly
        self.beta = beta
        # self.epsilon = epsilon

    def store(
        self,
        observation,
        action,
        reward: float,
        next_observation,
        done: bool,
        id=None,
        priority: float = None,
    ):
        super().store(observation, action, reward, next_observation, done, id)

        if priority is None:
            priority = self.max_priority**self.alpha
            self.max_priority = max(
                self.max_priority, priority
            )  # could remove and clip priorities in experience replay isntead

        self.sum_tree[self.tree_pointer] = priority**self.alpha
        self.min_tree[self.tree_pointer] = priority**self.alpha
        self.tree_pointer = (self.tree_pointer + 1) % self.max_size

    def set_beta(self, beta: float):
        self.beta = beta

    def sample(self):
        assert (
            len(self) >= self.batch_size
        ), "Only {} elements in buffer expected at least {}".format(
            len(self), self.batch_size
        )

        indices = self._sample_proportional()
        weights = np.array([self._calculate_weight(i) for i in indices])

        samples = self.sample_from_indices(indices)
        samples.update(dict(weights=weights, indices=indices))

        return samples

    def clear(self):
        super().clear()
        self.max_priority = self.initial_max_priority  # (initial) priority
        self.tree_pointer = 0

        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def update_priorities(self, indices: list[int], priorities: list[float], ids=None):
        # necessary for shared replay buffer
        if ids is not None:
            assert len(priorities) == len(ids) == len(indices)

            for index, id, priority in zip(indices, ids, priorities):
                assert priority > 0, "Negative priority: {}".format(priority)
                assert 0 <= index < len(self)

                if self.id_buffer[index] != id:
                    continue

                self.sum_tree[index] = priority**self.alpha
                self.min_tree[index] = priority**self.alpha
                self.max_priority = max(self.max_priority, priority)
        else:
            assert len(indices) == len(priorities)
            for index, priority in zip(indices, priorities):
                assert priority > 0, "Negative priority: {}".format(priority)
                assert 0 <= index < len(self)

                self.sum_tree[index] = priority**self.alpha
                self.min_tree[index] = priority**self.alpha
                self.max_priority = max(
                    self.max_priority, priority
                )  # could remove and clip priorities in experience replay isntead

        return priorities**self.alpha

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
