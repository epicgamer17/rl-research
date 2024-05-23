import numpy as np
from collections import deque

from utils import calculate_observation_buffer_shape

from replay_buffers.base_replay_buffer import BaseReplayBuffer


class NStepReplayBuffer(BaseReplayBuffer):
    def __init__(
        self, observation_dimensions, max_size: int, batch_size=32, n_step=1, gamma=0.99
    ):
        self.n_step = n_step
        self.gamma = gamma
        self.observation_dimensions = observation_dimensions
        super().__init__(max_size=max_size, batch_size=batch_size)

    def store(self, observation, action, reward, next_observation, done, id=None):
        transition = (observation, action, reward, next_observation, done)
        self.n_step_buffer.append(transition)

        if len(self.n_step_buffer) < self.n_step:
            return ()

        # compute n-step return and store
        reward, next_observation, done = self._get_n_step_info()
        observation, action = self.n_step_buffer[0][:2]
        self.id_buffer[self.pointer] = id
        self.observation_buffer[self.pointer] = observation
        self.action_buffer[self.pointer] = action
        self.reward_buffer[self.pointer] = reward
        self.next_observation_buffer[self.pointer] = next_observation
        self.done_buffer[self.pointer] = done

        self.pointer = (self.pointer + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        return self.n_step_buffer[0]

    def clear(self):
        observation_buffer_shape = calculate_observation_buffer_shape(
            self.max_size, self.observation_dimensions
        )
        self.observation_buffer = np.zeros(observation_buffer_shape, dtype=np.float32)
        self.next_observation_buffer = np.zeros(
            observation_buffer_shape, dtype=np.float32
        )

        self.id_buffer = np.zeros(self.max_size, dtype=np.object_)
        self.action_buffer = np.zeros(self.max_size, dtype=np.int32)
        self.reward_buffer = np.zeros(self.max_size, dtype=np.float32)
        self.done_buffer = np.zeros(self.max_size)
        self.pointer, self.trajectory_start_index = 0, 0

        self.pointer = 0
        self.size = 0
        self.n_step_buffer = deque(maxlen=self.n_step)

    def sample(self):
        indices = np.random.choice(self.size, self.batch_size, replace=False)

        return dict(
            observations=self.observation_buffer[indices],
            next_observations=self.next_observation_buffer[indices],
            actions=self.action_buffer[indices],
            rewards=self.reward_buffer[indices],
            dones=self.done_buffer[indices],
        )

    def sample_from_indices(self, indices):
        return dict(
            observations=self.observation_buffer[indices],
            next_observations=self.next_observation_buffer[indices],
            actions=self.action_buffer[indices],
            rewards=self.reward_buffer[indices],
            dones=self.done_buffer[indices],
            ids=self.id_buffer[indices],
        )

    def _get_n_step_info(self):
        reward, next_observation, done = self.n_step_buffer[-1][-3:]

        for transition in reversed(list(self.n_step_buffer)[:-1]):
            r, n_o, d = transition[-3:]
            reward = r + self.gamma * reward * (1 - d)
            next_observation, done = (n_o, d) if d else (next_observation, done)

        return reward, next_observation, done

    def __len__(self):
        return self.size
