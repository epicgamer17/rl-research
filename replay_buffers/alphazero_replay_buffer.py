import numpy as np


class Game:
    def __init__(self, num_players=2):
        self.length = 0
        self.observation_history = []
        self.rewards = []
        self.policy_history = []
        self.num_players = num_players

    def append(self, observation, reward, policy):
        self.observation_history.append(observation)
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


class ReplayBuffer:
    def __init__(
        self,
        max_size: int,
        batch_size: int,
    ):
        self.max_size = max_size
        self.batch_size = batch_size
        self.buffer = []

    def store(self, game):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append(game)

    def sample(self):
        move_sum = float(sum([len(game) for game in self.buffer]))
        games = np.random.choice(
            self.buffer,
            self.batch_size,
            p=[len(game) / move_sum for game in self.buffer],
        )

        game_indices = [(game, np.random.randint(len(game))) for game in games]

        return dict(
            observations=[game.observation_history[i] for game, i in game_indices],
            rewards=[game.rewards[i] for game, i in game_indices],
            policy=[game.policy_history[i] for game, i in game_indices],
        )

    def __len__(self):
        return self.size
