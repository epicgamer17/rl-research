from calendar import c
import numpy as np
from sympy import N
from replay_buffers.deprecated.alphazero_replay_buffer import AlphaZeroReplayBuffer
from replay_buffers.base_replay_buffer import BaseGameReplayBuffer, Game


class MuZeroReplayBuffer(
    BaseGameReplayBuffer
):  # does not inherit from AlphaZeroReplayBuffer but maybe should?
    def __init__(
        self,
        max_size: int,
        batch_size: int,
        n_step: int,
        num_unroll_steps: int,
        gamma: float,
        has_intermediate_rewards: bool,
    ):
        self.n_step = n_step
        self.unroll_steps = num_unroll_steps
        self.gamma = gamma
        self.has_intermediate_rewards = has_intermediate_rewards

        print("Warning: for board games it is recommnded to have n_step >= game length")

        super().__init__(max_size=max_size, batch_size=batch_size)

    def sample(self):
        game_indices = super().sample()

        observations = [game.observation_history[i] for game, i in game_indices]
        infos = [game.info_history[i] for game, i in game_indices]
        actions = [
            game.action_history[i : i + self.unroll_steps] for game, i in game_indices
        ]
        print("Old actions", actions)

        n_step_info = [
            self._get_n_step_info(
                i,
                game.value_history,
                game.policy_history,
                game.rewards,
                game.action_history,
                game.info_history,
                self.unroll_steps,
                self.n_step,
            )
            for game, i in game_indices
        ]

        values, policies, rewards, actions = zip(*n_step_info)
        return dict(
            observations=observations,
            rewards=rewards,
            policy=policies,
            values=values,
            actions=actions,
            infos=infos,
        )

    def _get_n_step_info(
        self,
        index: int,
        values: list,
        policies: list,
        rewards: list,
        actions: list,
        infos: list,
        num_unroll_steps: int,
        n_step: int,
    ):
        n_step_values = []
        n_step_rewards = []
        n_step_policies = []
        n_step_actions = []
        for current_index in range(index, index + num_unroll_steps + 1):
            bootstrap_index = current_index + n_step
            # print("bootstrapping")
            # value of current position is the value at the position n_steps away + rewards to get to the n_step position
            if bootstrap_index < len(values):
                if infos[current_index]["player"] == infos[bootstrap_index]["player"]:
                    value = values[bootstrap_index] * self.gamma**n_step
                else:
                    value = -values[bootstrap_index] * self.gamma**n_step
            else:
                value = 0

            # the rewards at this index to the bootstrap index should be added to the value
            for i, reward in enumerate(rewards[current_index:bootstrap_index]):
                # WHAT IS current_index + i + 1 when current index is the last frame?? IS THIS AN ERROR?
                if (
                    infos[current_index]["player"]
                    == infos[current_index + i + 1]["player"]
                ):
                    value += (
                        reward * self.gamma**i
                    )  # pytype: disable=unsupported-operands
                else:
                    value -= (
                        reward * self.gamma**i
                    )  # pytype: disable=unsupported-operands

            # target reward is the reward before the ones added to the value
            if current_index > 0 and current_index <= len(rewards):
                if self.has_intermediate_rewards:
                    last_reward = rewards[current_index - 1]
                else:
                    print(
                        "Warning: for games with no intermediate rewards n_step should be >= game length"
                    )
                    value += rewards[current_index - 1]
                    last_reward = None
            else:
                last_reward = None  # self absorbing state

            if current_index < len(values):
                n_step_values.append(value)
                n_step_rewards.append(last_reward)
                n_step_policies.append(policies[current_index])
                n_step_actions.append(actions[current_index])
            else:
                n_step_values.append(0)
                n_step_rewards.append(last_reward)
                n_step_policies.append([])  # self absorbing state
                n_step_actions.append(None)  # self absorbing state

        print(actions[index : index + num_unroll_steps])
        print(n_step_actions[:-1])
        return (
            n_step_values,  # [initial value, recurrent values]
            n_step_policies,  # [initial policy, recurrent policies]
            n_step_rewards,  # [initial reward (0), recurrent rewards] initial reward is useless like the first last action, but we ignore it in the learn function
            n_step_actions[:-1],  # [recurrent actions, extra action]
        )  # remove the last actions, as there should be one less action than other stuff
