import copy


class BaseReplayBuffer:
    def __init__(
        self,
        observation_dimensions,
        max_size,
        batch_size=32,
    ):
        self.observation_dimensions = observation_dimensions
        self.max_size = max_size
        self.batch_size = batch_size
        self.size = 0

    def store(self, *args, **kwargs):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError

    def __len__(self):
        return self.size


class Game:
    def __init__(self, num_players=2, num_actions=9, discount=1.0):
        self.length = 0
        self.observation_history = []
        self.rewards = []
        self.policy_history = []
        self.num_players = num_players

    def append(self, observation, reward, policy):
        # print("Observation", observation)
        # print("Reward", reward)
        # print("Policy", policy)
        self.observation_history.append(copy.deepcopy(observation))
        self.rewards.append(reward)
        self.policy_history.append(policy)
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
