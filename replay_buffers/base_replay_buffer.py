import copy


class BaseReplayBuffer:
    def __init__(
        self,
        max_size: int,
        batch_size: int = None,
    ):
        self.max_size = max_size
        self.batch_size = batch_size if batch_size is not None else max_size
        self.clear()
        assert self.size == 0, "Replay buffer should be empty at initialization"

    def store(self, *args, **kwargs):
        raise NotImplementedError

    def sample(self):
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

        self.num_players = num_players

    def append(self, observation, reward: int, policy, value=None, action=None):
        self.observation_history.append(copy.deepcopy(observation))
        self.rewards.append(reward)
        self.policy_history.append(policy)
        self.value_history.append(value)
        self.action_history.append(action)
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
