import numpy as np
from collections import deque


class ReplayBuffer:
    def __init__(
        self, observation_dimensions, max_size: int, batch_size=0, n_step=1, gamma=0.99
    ):
        # self.observation_buffer = np.zeros((max_size,) + observation_dimensions, dtype=np.float32)
        # self.next_observation_buffer = np.zeros((max_size,) + observation_dimensions, dtype=np.float32)
        observation_buffer_shape = []
        observation_buffer_shape += [max_size]
        observation_buffer_shape += list(observation_dimensions)
        observation_buffer_shape = list(observation_buffer_shape)
        self.observation_buffer = np.zeros(observation_buffer_shape, dtype=np.float32)
        self.next_observation_buffer = np.zeros(
            observation_buffer_shape, dtype=np.float32
        )

        self.id_buffer = np.zeros(max_size, dtype=np.str_)
        self.action_buffer = np.zeros(max_size, dtype=np.int32)
        self.reward_buffer = np.zeros(max_size, dtype=np.float32)
        self.done_buffer = np.zeros(max_size)
        self.pointer, self.trajectory_start_index = 0, 0

        self.max_size = max_size
        self.batch_size = batch_size if batch_size > 0 else max_size
        self.pointer = 0
        self.size = 0

        # n-step learning
        self.n_step_buffer = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

    def store(self, observation, action, reward, next_observation, done, id=None):
        # print("Storing in Buffer")
        # time1 = 0
        # time1 = time()
        transition = (observation, action, reward, next_observation, done)
        self.n_step_buffer.append(transition)

        if len(self.n_step_buffer) < self.n_step:
            # print("Buffer Storage Time ", time() - time1)
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

        # print("Buffer Storage Time ", time() - time1)
        return self.n_step_buffer[0]

    def sample(self):
        # print("Sampling From Buffer")
        # time1 = time()
        indices = np.random.choice(self.size, self.batch_size, replace=False)

        # print("Buffer Sampling Time ", time() - time1)
        return dict(
            observations=self.observation_buffer[indices],
            next_observations=self.next_observation_buffer[indices],
            actions=self.action_buffer[indices],
            rewards=self.reward_buffer[indices],
            dones=self.done_buffer[indices],
        )

    def sample_from_indices(self, indices):
        # print("Sampling From Indices")
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
