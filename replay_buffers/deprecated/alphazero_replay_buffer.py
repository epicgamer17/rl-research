import numpy as np
import sys

sys.path.ap
from replay_buffers.deprecated.base_replay_buffer import Game, BaseReplayBuffer


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
        self.info_history = []

        self.num_players = num_players

    def append(
        self,
        observation,
        info,
        reward: int = None,
        policy=None,
        value=None,
        action=None,
    ):
        self.observation_history.append(copy.deepcopy(observation))
        self.info_history.append(copy.deepcopy(info))
        if reward is not None:
            self.rewards.append(reward)
        if policy is not None:
            self.policy_history.append(policy)
        if value is not None:
            self.value_history.append(value)
        if action is not None:
            self.action_history.append(action)
        # print("Game info history", self.info_history)

    def set_rewards(self):
        print("Initial Rewards", self.rewards)
        final_reward = self.rewards[-1]
        for i in reversed(range(self.length)):
            self.rewards[i] = (
                final_reward[i % self.num_players]
                # if i % self.num_players == (self.length - 1) % self.num_players
                # else -final_reward
            )
        print("Updated Rewards", self.rewards)

    def __len__(self):
        # SHOULD THIS BE LEN OF ACTIONS INSTEAD???
        # AS THIS ALLOWS SAMPLING THE TERMINAL STATE WHICH HAS NO FURTHER ACTIONS
        return len(self.observation_history)


class BaseGameReplayBuffer(BaseReplayBuffer):
    def __init__(
        self,
        max_size: int,
        batch_size: int,
    ):
        super().__init__(max_size=max_size, batch_size=batch_size)

    def store(self, game: Game):
        self.buffer[self.pointer] = copy.deepcopy(game)
        self.pointer = (self.pointer + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self):
        move_sum = float(sum([len(game) for game in self.buffer]))
        games: list[Game] = np.random.choice(
            self.buffer,
            self.batch_size,
            p=[len(game) / move_sum for game in self.buffer],
        )

        return [(game, np.random.randint(len(game))) for game in games]

    def clear(self):
        self.buffer: list[Game] = torch.zeros(self.max_size, dtype=torch.object)
        self.size = 0
        self.pointer = 0


class AlphaZeroReplayBuffer(BaseGameReplayBuffer):
    def sample(self):
        game_indices = super().sample()
        return dict(
            observations=np.array(
                [game.observation_history[i] for game, i in game_indices]
            ),
            policies=np.array([game.policy_history[i] for game, i in game_indices]),
            rewards=np.array([[game.rewards[i]] for game, i in game_indices]),
            infos=[game.info_history[i] for game, i in game_indices],
        )
