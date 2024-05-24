from time import time
import numpy as np
from replay_buffers.segment_tree import SumSegmentTree, MinSegmentTree
from replay_buffers.fast_sum_tree import FastSumTree
from replay_buffers.n_step_replay_buffer import NStepReplayBuffer


class PrioritizedNStepReplayBuffer(NStepReplayBuffer):
    def __init__(
        self,
        observation_dimensions,
        max_size: int,
        batch_size: int = 32,
        max_priority: float = 1.0,
        alpha: float = 0.6,
        beta: float = 0.4,
        # epsilon=0.01,
        n_step: float = 1,
        gamma: float = 0.99,
    ):
        assert alpha >= 0 and alpha <= 1
        assert beta >= 0 and beta <= 1
        assert n_step >= 1
        assert gamma > 0 and gamma <= 1

        super(PrioritizedNStepReplayBuffer, self).__init__(
            observation_dimensions, max_size, batch_size, n_step=n_step, gamma=gamma
        )

        self.max_priority = max_priority  # (initial) priority
        self.tree_pointer = 0

        self.alpha = alpha  # Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly
        self.beta = beta
        # self.epsilon = epsilon
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def store(
        self,
        observation,
        action,
        reward: float,
        next_observation,
        done: bool,
        priority: float = None,
        id=None,
    ):
        transition = super().store(
            observation, action, reward, next_observation, done, id
        )

        if priority is None:
            priority = self.max_priority**self.alpha
            self.max_priority = max(
                self.max_priority, priority
            )  # could remove and clip priorities in experience replay isntead

        if transition:
            self.sum_tree[self.tree_pointer] = priority**self.alpha
            self.min_tree[self.tree_pointer] = priority**self.alpha
            self.tree_pointer = (self.tree_pointer + 1) % self.max_size

        return transition

    def sample(self):
        assert (
            len(self) >= self.batch_size
        ), "Only {} elements in buffer expected at least {}".format(
            len(self), self.batch_size
        )

        indices = self._sample_proportional()

        observations = self.observation_buffer[indices]
        next_observations = self.next_observation_buffer[indices]
        actions = self.action_buffer[indices]
        rewards = self.reward_buffer[indices]
        dones = self.done_buffer[indices]
        ids = self.id_buffer[indices]
        weights = np.array([self._calculate_weight(i) for i in indices])

        return dict(
            observations=observations,
            next_observations=next_observations,
            actions=actions,
            rewards=rewards,
            dones=dones,
            weights=weights,
            indices=indices,
            ids=ids,
        )

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
            # priorities += self.self.epsilon
            for index, priority in zip(indices, priorities):
                # print("Priority", priority)
                assert priority > 0, "Negative priority: {}".format(priority)
                assert 0 <= index < len(self)

                self.sum_tree[index] = priority**self.alpha
                self.min_tree[index] = priority**self.alpha
                self.max_priority = max(
                    self.max_priority, priority
                )  # could remove and clip priorities in experience replay isntead

        return priorities**self.alpha

    def _sample_proportional(self):
        # print("Getting Indices from PrioritizedReplayBuffer Sum Tree")
        # time1 = 0
        # time1 = time()
        indices = []
        total_priority = self.sum_tree.sum(0, len(self) - 1)
        priority_segment = total_priority / self.batch_size

        for i in range(self.batch_size):
            a = priority_segment * i
            b = priority_segment * (i + 1)
            upperbound = np.random.uniform(a, b)
            index = self.sum_tree.retrieve(upperbound)
            indices.append(index)

        # print("Getting Indices from PrioritizedReplayBuffer Sum Tree Time ", time() - time1)
        return indices

    def _calculate_weight(self, index: int):
        min_priority = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (min_priority * len(self)) ** (-self.beta)
        priority_sample = self.sum_tree[index] / self.sum_tree.sum()
        weight = (priority_sample * len(self)) ** (-self.beta)
        weight = weight / max_weight

        # print("Min Tree Min ", self.min_tree.min())
        # print("Max Weight ", max_weight)
        # print("Min Priority ", min_priority)
        # print("Length Self ", len(self))
        return weight

    def __check_id__(self, index: int, id: str) -> bool:
        return self.id_buffer[index] == id


class FastPrioritizedReplayBuffer(NStepReplayBuffer):
    def __init__(
        self,
        observation_dimensions,
        max_size: int,
        batch_size: int = 32,
        max_priority: float = 1.0,
        alpha: float = 0.6,
        beta: float = 0.4,
        # epsilon=0.01,
        n_step: int = 1,
        gamma: float = 0.99,
    ):
        assert alpha >= 0 and alpha <= 1
        assert beta >= 0 and beta <= 1
        assert n_step >= 1
        assert gamma > 0 and gamma <= 1

        super(FastPrioritizedReplayBuffer, self).__init__(
            observation_dimensions, max_size, batch_size, n_step=n_step, gamma=gamma
        )

        self.max_priority = max_priority  # (initial) priority
        self.min_priority = max_priority
        self.tree_pointer = 0

        self.alpha = alpha  # Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly
        self.beta = beta
        # self.epsilon = epsilon

        self.tree = FastSumTree(self.max_size)

    def store(
        self,
        observation,
        action,
        reward: float,
        next_observation,
        done: bool,
    ):
        transition = super().store(observation, action, reward, next_observation, done)

        # max_priority = np.max(self.tree.tree[-self.tree.capacity :])
        # if max_priority == 0:
        #     max_priority = self.max_priority

        if transition:
            self.tree.add(self.tree_pointer, self.max_priority)
            self.tree_pointer = (self.tree_pointer + 1) % self.max_size

        return transition

    def sample(self):
        # print("Sampling from PrioritizedReplayBuffer")
        # time1 = 0
        # time1 = time()
        assert len(self) >= self.batch_size

        # indices = self._sample_proportional()
        # print("Getting Indices from PrioritizedReplayBuffer Sum Tree")
        # time1 = 0
        # time1 = time()
        priority_segment = self.tree.total_priority / self.batch_size
        indices, weights = np.empty((self.batch_size,), dtype=np.int32), np.empty(
            (self.batch_size, 1), dtype=np.float32
        )
        # print("Total Priority",self.tree.total_priority)
        for i in range(self.batch_size):
            a, b = priority_segment * i, priority_segment * (i + 1)
            # print(a, b)
            # print("a, b", a, b)
            value = np.random.uniform(a, b)
            index, priority = self.tree.retrieve(value)
            sampling_probabilities = priority / self.tree.total_priority
            # weights[i, 0] = (self.batch_size * sampling_probabilities) ** -beta
            weights[i, 0] = (len(self) * sampling_probabilities) ** -self.beta
            indices[i] = index - self.tree.capacity + 1
            indices[i] = index - self.tree.capacity + 1

        # max_weight = max(weights)
        max_weight = (
            len(self) * self.min_priority / self.tree.total_priority
        ) ** -self.beta
        weights = weights / max_weight

        # print(weights)
        # print("Getting Indices from PrioritizedReplayBuffer Sum Tree Time ", time() - time1)
        # print("Retrieving Data from PrioritizedReplayBuffer Data Arrays")
        # time2 = 0
        # time2 = time()
        observations = self.observation_buffer[indices]
        next_observations = self.next_observation_buffer[indices]
        actions = self.action_buffer[indices]
        rewards = self.reward_buffer[indices]
        dones = self.done_buffer[indices]
        # weights = np.array([self._calculate_weight(i, beta) for i in indices])
        # print("Retrieving Data from PrioritizedReplayBuffer Data Arrays Time ", time() - time2)

        # print("Sampling from PrioritizedReplayBuffer Time ", time() - time1)
        return dict(
            observations=observations,
            next_observations=next_observations,
            actions=actions,
            rewards=rewards,
            dones=dones,
            weights=weights,
            indices=indices,
        )

    def update_priorities(self, indices: list[int], priorities: list[float]):
        assert len(indices) == len(priorities)
        # priorities += self.epsilon

        for index, priority in zip(indices, priorities):
            assert priority > 0, "Negative priority: {}".format(priority)
            # assert 0 <= index < len(self)
            # self.tree[index] = priority ** self.alpha
            self.max_priority = max(self.max_priority, priority**self.alpha)
            self.min_priority = min(self.min_priority, priority**self.alpha)
            # priority = np.clip(priority, self.epsilon, self.max_priority)
            self.tree.update(index + self.tree.capacity - 1, priority**self.alpha)
