import typing
from typing import NamedTuple
import numpy as np
import numpy.typing as npt
from .replay_buffer import Storable, Sampleable, NStepable, WithId
from .sample_functions import sample_tree_proportional
from collections import deque
from .segment_tree import SumSegmentTree, MinSegmentTree


class Transition(NamedTuple):
    """
    Represents a transition t := (s, a, r, s', done) in the replay buffer
    """

    observation: np.ndarray
    action: int
    reward: float
    next_observation: np.ndarray
    done: bool


class TransitionBatch(NamedTuple):
    """
    "struct of arrays" representation of a batch of transitions:

    indices: np.ndarray[typing.Any, np.int64]
    observations: np.ndarray[typing.Any, np.float32]
    actions: np.ndarray[typing.Any, np.int32]
    rewards: np.ndarray[typing.Any, np.float32]
    next_observations: np.ndarray[typing.Any, np.float32]
    dones: np.ndarray[typing.Any, np.bool_]
    """

    ids: np.ndarray[typing.Any, np.object_]
    indices: np.ndarray[typing.Any, np.int64]
    observations: npt.NDArray[typing.Any]
    actions: np.ndarray[typing.Any, np.int32]
    rewards: np.ndarray[typing.Any, np.float32]
    next_observations: npt.NDArray[typing.Any]
    dones: np.ndarray[typing.Any, np.bool_]
    weights: np.ndarray[typing.Any, np.float32]


class ReplayBuffer(
    Storable[Transition],
    Sampleable[TransitionBatch],
    NStepable[Transition],
    WithId,
):
    """
    Stores n-step transitions t := (s, a, r, s', done) in an n-step replay buffer
    Samples a batch of n-step transitions list[t] from the buffer
    """

    def __init__(
        self,
        observation_dimensions,
        max_size: int,
        batch_size=1,
        n_step=1,
        gamma=0.99,
        alpha=0.6,
        max_priority=1.0,
        min_size=None,
    ):

        assert n_step > 0
        assert gamma > 0 and gamma <= 1
        assert max_size > 0
        assert batch_size >= 0
        assert alpha >= 0
        if min_size == None:
            min_size = batch_size
        assert min_size >= batch_size

        # self.observation_buffer = np.zeros((max_size,) + observation_dimensions, dtype=np.float32)
        # self.next_observation_buffer = np.zeros((max_size,) + observation_dimensions, dtype=np.float32)
        observation_buffer_shape = []
        observation_buffer_shape += [max_size]
        observation_buffer_shape += list(observation_dimensions)
        observation_buffer_shape = list(observation_buffer_shape)

        # transition storage
        self.observation_buffer = np.zeros(observation_buffer_shape, dtype=np.float32)
        self.next_observation_buffer = np.zeros(
            observation_buffer_shape, dtype=np.float32
        )
        self.action_buffer = np.zeros(max_size, dtype=np.int32)
        self.reward_buffer = np.zeros(max_size, dtype=np.float32)
        self.done_buffer = np.zeros(max_size)

        # id storage
        self.id_buffer = np.zeros(max_size, dtype=np.object_)

        self.max_size = max_size
        self.min_size = min_size
        self.batch_size = batch_size if batch_size > 0 else max_size
        self.pointer = 0
        self.size = 0

        # n-step learning
        self.n_step_buffer = deque[Transition](maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

        # experience replay
        self.alpha = alpha
        self.tree_capacity = 2 ** (int(np.ceil(np.log2(self.max_size))) + 1)
        self.sum_tree = SumSegmentTree(self.tree_capacity)
        self.min_tree = MinSegmentTree(self.tree_capacity)
        self.tree_pointer = 0
        self.max_priority = max_priority

    def __store__id__(self, index: int, id: str):
        self.id_buffer[index] = id

    def __check_id__(self, index: int, id: str) -> bool:
        return self.id_buffer[index] == id

    def __store__(self, t: Transition, priority=None):
        # store n-step transition in buffers
        self.observation_buffer[self.pointer] = t.observation
        self.action_buffer[self.pointer] = t.action
        self.reward_buffer[self.pointer] = t.reward
        self.next_observation_buffer[self.pointer] = t.next_observation
        self.done_buffer[self.pointer] = t.done

        # update pointer and size
        self.pointer = (self.pointer + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        # update trees
        if priority is None:
            priority = self.max_priority**self.alpha

        self.sum_tree[self.tree_pointer] = priority**self.alpha
        self.min_tree[self.tree_pointer] = priority**self.alpha

        self.tree_pointer = (self.tree_pointer + 1) % self.max_size
        return t

    def store(
        self,
        observation,
        action,
        reward,
        next_observation,
        done,
        id=None,
        priority=None,
    ):
        # print("Storing in Buffer")
        # time1 = 0
        # time1 = time()
        self.n_step_buffer.append(
            Transition(observation, action, reward, next_observation, done)
        )

        if len(self.n_step_buffer) < self.n_step:
            return None

        # compute n-step return and store
        t = self.__get_n_step_info__()
        self.__store__id__(self.pointer, id)
        self.__store__(t, priority)

        # print("Buffer Storage Time ", time() - time1)
        return t

    def __sample__(self, beta=0.4) -> TransitionBatch:
        assert self.__len__() >= self.min_size
        assert beta > 0

        indices = sample_tree_proportional(self.sum_tree, self.batch_size, self.size)

        return TransitionBatch(
            indices=indices,
            ids=self.id_buffer[indices],
            observations=self.observation_buffer[indices],
            actions=self.action_buffer[indices],
            rewards=self.reward_buffer[indices],
            next_observations=self.next_observation_buffer[indices],
            dones=self.done_buffer[indices],
            weights=np.array([self._calculate_weight(i, beta) for i in indices]),
        )

    def __get_n_step_info__(self) -> Transition:
        reward, next_observation, done = self.n_step_buffer[-1][-3:]

        for transition in reversed(list(self.n_step_buffer)[:-1]):
            r, new_obs, d = transition[-3:]
            reward = r + self.gamma * reward * (1 - d)
            next_observation, done = (new_obs, d) if d else (next_observation, done)

        return Transition(
            observation=self.n_step_buffer[0][0],
            action=self.n_step_buffer[0][1],
            reward=reward,
            next_observation=next_observation,
            done=done,
        )

    def __len__(self):
        return self.size

    def _calculate_weight(self, index, beta):
        min_priority = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (min_priority * len(self)) ** (-beta)
        priority_sample = self.sum_tree[index] / self.sum_tree.sum()
        weight = (priority_sample * len(self)) ** (-beta)
        weight = weight / max_weight

        # print(beta, len(self), self.min_tree.min(), min_priority)
        # print(max_weight)
        return weight

    def update_priorities(self, indices, priorities, ids=None):
        # necessary for shared replay buffer
        if ids is not None:
            # ids_updated = 0
            # ids_skipped = 0
            assert len(priorities) == len(ids) == len(indices)

            for index, id, priority in zip(indices, ids, priorities):
                assert priority > 0, "Negative priority: {}".format(priority)
                assert 0 <= index < len(self)

                if self.id_buffer[index] != id:
                    # ids_skipped += 1
                    continue

                new_priority = priority**self.alpha
                self.sum_tree[index] = new_priority
                self.min_tree[index] = new_priority
                self.max_priority = max(self.max_priority, priority)
                # ids_updated += 1

            # print("updated: ", ids_updated, "skipped:", ids_skipped)

        else:
            assert len(indices) == len(priorities)
            # priorities += self.self.epsilon
            for index, priority in zip(indices, priorities):
                # print("Priority", priority)
                assert priority > 0, "Negative priority: {}".format(priority)
                assert 0 <= index < len(self)

                new_priority = priority**self.alpha
                self.sum_tree[index] = new_priority
                self.min_tree[index] = new_priority
                self.max_priority = max(self.max_priority, priority)
                # self.max_priority = max(
                #     self.max_priority, priority
                # )  # could remove and clip priorities in experience replay isntead
