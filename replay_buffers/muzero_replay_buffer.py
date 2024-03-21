from collections import deque
import numpy as np
import copy


class Game:
    def __init__(
        self, num_players=2, num_actions=9, discount=1.0, n_step=1, gamma=0.99
    ):
        self.length = 0
        self.observation_history = []
        self.rewards = []
        self.policy_history = []
        self.value_history = []
        self.action_history = []

        self.num_players = num_players

    def append(self, observation, reward, policy, value, action):
        self.observation_history.append(copy.deepcopy(observation))
        self.rewards.append(reward)
        self.policy_history.append(policy)
        self.value_history.append(value)
        self.action_history.append(action)
        self.length += 1

    def __len__(self):
        return self.length


class ReplayBuffer:
    def __init__(
        self,
        max_size: int,
        batch_size: int,
        n_step: int,
        gamma: float,
    ):
        self.max_size = max_size
        self.batch_size = batch_size
        self.buffer = []

        self.n_step = n_step
        self.gamma = gamma

    def store(self, game):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append(game)

    def sample(self, num_unroll_steps, n_step):
        move_sum = float(sum([len(game) for game in self.buffer]))
        games = np.random.choice(
            self.buffer,
            self.batch_size,
            p=[len(game) / move_sum for game in self.buffer],
        )

        game_indices = [(game, np.random.randint(len(game))) for game in games]

        values, policies, rewards = [
            self._get_n_step_info(
                i,
                game.value_history,
                game.policy_history,
                game.rewards,
                num_unroll_steps,
                n_step,
            )
            for game, i in game_indices
        ]

        return dict(
            observations=[game.observation_history[i] for game, i in game_indices],
            rewards=rewards,
            policy=policies,
            values=values,
            actions=[
                game.action_history[i : i + num_unroll_steps]
                for game, i in game_indices
            ],
        )

    def _get_n_step_info(
        self, index, values, policies, rewards, num_unroll_steps, n_step
    ):
        n_step_values = []
        n_step_rewards = []
        n_step_policies = []
        for current_index in range(index, index + num_unroll_steps + 1):
            bootstrap_index = current_index + n_step
            if bootstrap_index < len(values):
                value = values[bootstrap_index] * self.gamma**n_step
            else:
                value = 0

            for i, reward in enumerate(rewards[current_index:bootstrap_index]):
                value += reward * self.gamma**i  # pytype: disable=unsupported-operands

            if current_index < len(values):
                n_step_values.append(value)
                n_step_rewards.append(rewards[current_index])
                n_step_policies.append(policies[current_index])
            else:
                # States past the end of games are treated as absorbing states.
                n_step_values.append(0)
                n_step_rewards.append(0)
                n_step_policies.append([])

        return n_step_values, n_step_policies, n_step_rewards

    def __len__(self):
        return self.size
