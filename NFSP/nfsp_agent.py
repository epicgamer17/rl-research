# baseline for NFSP, should work for any model type (paper uses DQN)
# Initialize game Γ and execute an agent via RUNAGENT for each player in the game
# function RUNAGENT(Γ)
# Initialize replay memories MRL (circular buffer) and MSL (reservoir)
# Initialize average-policy network Π(s, a | θΠ) with random parameters θΠ
# Initialize action-value network Q(s, a | θQ) with random parameters θQ
# Initialize target network parameters θQ0 ← θQ
# Initialize anticipatory parameter η
# for each episode do
# Set policy σ ← {
# epsilon-greedy (Q), with probability η
# Π, with probability 1 − η
# }
# Observe initial information state s1 and reward r1
# for t = 1, T do
# Sample action at from policy σ
# Execute action at in game and observe reward rt+1 and next information state st+1
# Store transition (st, at, rt+1, st+1) in reinforcement learning memory MRL
# if agent follows best response policy σ = epsilon-greedy (Q) then
# Store behaviour tuple (st, at) in supervised learning memory MSL
# end if
# Update θΠ with stochastic gradient descent on loss
# L(θΠ) = E(s,a) from MSL [log Π(s, a | θΠ)]
# Update θQ with stochastic gradient descent on loss
# LθQ = E(s,a,r,s0) from MRL [r + maxa0 Q(s0, a0| θQ0) − Q(s, a | θQ)^2]
# Periodically update target network parameters θ   Q0 ← θQ
# end for
# end for
# end function

from collections import deque
import copy
import sys
from time import time

from agent_configs import NFSPDQNConfig
import numpy as np

sys.path.append("../")

import os

os.environ["OMP_NUM_THREADS"] = f"{8}"
os.environ["MKL_NUM_THREADS"] = f"{8}"
os.environ["TF_NUM_INTEROP_THREADS"] = f"{8}"
os.environ["TF_NUM_INTRAOP_THREADS"] = f"{8}"

import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(0)
tf.config.threading.set_inter_op_parallelism_threads(0)

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # memory growth must be set before GPUs have been initialized
        print(e)

import random
from supervised_agent import AverageStrategyAgent
from base_agent.agent import BaseAgent
from configs.agent_configs.agent_configs.nfsp_config import NFSPDQNConfig


class NFSPDQN(BaseAgent):
    def __init__(self, env, config: NFSPDQNConfig, agent_type) -> None:
        super().__init__(env, config, "NFSPDQN")
        self.agent_type = agent_type
        rl_configs = self.config.rl_configs
        sl_configs = self.config.sl_configs
        self.nfsp_agents = [
            NFSPDQNAgent(
                env, rl_configs[i], sl_configs[i], f"NFSPAgent_{i}", agent_type
            )
            for i in range(self.config.num_players)
        ]
        self.checkpoint_interval = 50

    def train(self):
        training_time = time()
        self.is_test = False
        stats = {
            "rl_loss": [],
            "sl_loss": [],
            "test_score": [],
        }
        targets = {
            "test_score": self.env.spec.reward_threshold,
        }

        state_p1, info = self.env.reset()

        for p in range(self.config.num_players):
            self.nfsp_agents[p].select_policy(self.config.anticipatory_param)

        for training_step in range(self.training_steps):
            for _ in range(self.config.replay_interval):
                action_p1 = self.nfsp_agents[0].select_action(
                    state_p1,
                    (info["legal_moves"] if self.config.game.has_legal_moves else None),
                )
                self.nfsp_agents[0].rl_agent.transition = [state_p1, action_p1]
                next_state_p2, reward_p1, terminated, truncated, info = self.step(
                    action_p1
                )
                done = terminated or truncated

                if done:
                    reward_p1 = reward_p1[0]
                    reward_p2 = reward_p1[1]
                    # next_state_p1 is irrelivent but could be the terminal state from p1 persepctive (instead of p2 which is what it currently is)
                    # rewards from list
                    # state_p1 is as normal
                    # action_p1 is as normal
                    # done is True
                    # state_p2 may not exist (if it is the first move of the game and p1 folds) otherwise as normal
                    # next_state_p2 is irrelevant but it will always exist but could be wrong (if it is the first move of the game) otherwise it is as normal
                    # reward from list
                    # action_p2 may not exist (first move) but if it does it is as normal
                    # done is True
                    state_p1, info = self.env.reset()

                # dont do twice if done is true maybe should exit in done check
                if len(self.nfsp_agents[1].rl_agent.transition) == 2:
                    self.nfsp_agents[1].rl_agent.transition += [
                        reward_p2,
                        next_state_p2,
                        done,
                    ]

                one_step_transition = self.nfsp_agents[
                    1
                ].rl_agent.n_step_replay_buffer.store(
                    *self.nfsp_agents[1].rl_agent.transition
                )
                if one_step_transition is not None:
                    self.nfsp_agents[1].rl_agent.replay_buffer.store(
                        *one_step_transition
                    )

                state_p2 = next_state_p2
                action_p2 = self.nfsp_agents[1].select_action(
                    state_p2,
                    (info["legal_moves"] if self.config.game.has_legal_moves else None),
                )

                self.nfsp_agents[1].rl_agent.transition = [state_p2, action_p2]

                next_state_p1, reward_p2, terminated, truncated, info = self.step(
                    action_p2
                )
                done = terminated or truncated

                if done:
                    reward_p1 = reward_p2[1]
                    reward_p2 = reward_p2[0]
                    # rewards from list
                    # state_p1 and next_state_p1 (next_state_p1 is irrelevant) are as normal
                    # action_p1 is as normal
                    # done is True
                    # next_state_p2 is irrelevant (but could be terminal state from p2 persepctive (instead of p1 which is what it currently is))
                    # state_p2 as normal
                    # rewards from list
                    # action_p2 as normal
                    # done is True
                    state_p1, info = self.env.reset()

                self.nfsp_agents[0].rl_agent.transition += [
                    reward_p1,
                    next_state_p1,
                    done,
                ]

                one_step_transition = self.nfsp_agents[
                    0
                ].rl_agent.n_step_replay_buffer.store(
                    *self.nfsp_agents[0].rl_agent.transition
                )
                if one_step_transition is not None:
                    self.nfsp_agents[0].rl_agent.replay_buffer.store(
                        *one_step_transition
                    )

            self.nfsp_agents[0].rl_agent.config.per_beta = min(
                1.0,
                self.nfsp_agents[0].rl_agent.config.per_beta
                + (1 - self.nfsp_agents[0].rl_agent.config.per_beta)
                / self.training_steps,  # per beta increase
            )
            self.nfsp_agents[1].rl_agent.config.per_beta = min(
                1.0,
                self.nfsp_agents[1].rl_agent.config.per_beta
                + (1 - self.nfsp_agents[1].rl_agent.config.per_beta)
                / self.training_steps,  # per beta increase
            )

            for p in range(self.config.num_players):
                for minibatch in range(self.config.num_minibatches):
                    rl_loss, sl_loss = self.nfsp_agents[p].experience_replay()
                    if p == 0:
                        stats["rl_loss"].append(rl_loss)
                        stats["sl_loss"].append(sl_loss)

                if training_step % self.config.rl_configs[p].transfer_interval == 0:
                    self.nfsp_agents[p].rl_agent.target_model.set_weights(
                        self.nfsp_agents[p].rl_agent.model.get_weights()
                    )

            if training_step % self.checkpoint_interval == 0 and training_step > 0:
                self.save_checkpoint(
                    stats,
                    targets,
                    50,
                    training_step,
                    training_step * self.config.replay_interval,
                    time() - training_time,
                )

        self.save_checkpoint(
            stats,
            targets,
            50,
            training_step,
            training_step * self.config.replay_interval,
            time() - training_time,
        )
        self.env.close()

    def save_checkpoint(
        self, stats, targets, num_trials, training_step, frames_seen, time_taken
    ):
        # save the model weights
        if not os.path.exists("./model_weights"):
            os.makedirs("./model_weights")
        if not os.path.exists("./model_weights/{}".format(self.model_name)):
            os.makedirs("./model_weights/{}".format(self.model_name))

        for p in range(self.config.num_players):
            self.nfsp_agents[p].rl_agent.model.save(
                f"./model_weights/{self.model_name}/rl_agent_{p}_episode_{training_step}.keras"
            )
            self.nfsp_agents[p].sl_agent.model.save(
                f"./model_weights/{self.model_name}/sl_agent_{p}_episode_{training_step}.keras"
            )

        # save replay buffer
        # save optimizer

        # test model
        test_score = self.test(num_trials)
        stats["test_score"].append(test_score)
        # plot the graphs
        self.plot_graph(stats, targets, training_step, frames_seen, time_taken)

    def step(self, action):
        if self.is_test:
            return self.test_env.step(action)
        else:
            return self.env.step(action)

    def test(self, num_trials):
        print("Testing")
        self.is_test = True
        test_score = 0
        for _ in range(num_trials):
            print("Trial ", _)
            state, info = self.test_env.reset()
            done = False
            policies = [
                self.nfsp_agents[p].policy for p in range(self.config.num_players)
            ]
            for p in range(self.config.num_players):
                self.nfsp_agents[p].policy = "best_response"
            self.nfsp_agents[0].policy = "average_strategy"
            while not done:
                for p in range(self.config.num_players):
                    action = self.nfsp_agents[p].select_action(
                        state,
                        (
                            info["legal_moves"]
                            if self.config.game.has_legal_moves
                            else None
                        ),
                    )
                    next_state, reward, terminated, truncated, info = self.step(action)
                    done = terminated or truncated
                    state = next_state
                    if done:
                        break
            test_score += 1 if reward[0] >= 0 else 0
        self.is_test = False
        for p in range(self.config.num_players):
            self.nfsp_agents[p].policy = policies[p]
        return 1 - (test_score / num_trials)


class NFSPDQNAgent(BaseAgent):
    def __init__(self, env, rl_config, sl_config, name, agent_type) -> None:
        super().__init__(env, rl_config, name)
        self.rl_agent = agent_type(env, rl_config, name)
        self.sl_agent = AverageStrategyAgent(env, sl_config, name)
        self.policy = "best_response"  # "average_strategy" or "best_response

    def select_policy(self, anticipatory_param):
        if random.random() < anticipatory_param and not self.is_test:
            return "best_response"
        else:
            return "average_strategy"

    def select_action(self, state, legal_moves=None):
        if self.policy == "average_strategy":
            action = self.sl_agent.select_action(state, legal_moves)
        else:
            action = self.rl_agent.select_action(state, legal_moves)
            if not self.is_test:
                self.sl_agent.replay_buffer.store(
                    state, action
                )  # Store best moves in SL Memory
            return action

        return action

    def experience_replay(self):
        rl_loss = 0
        sl_loss = 0
        print("RL Buffer Size ", self.rl_agent.replay_buffer.size)
        print("Min RL Buffer Size", self.sl_agent.config.min_replay_buffer_size)
        if (
            self.rl_agent.replay_buffer.size
            > self.rl_agent.config.min_replay_buffer_size
        ):
            print("Experience Replay for RL Agent")
            rl_loss = self.rl_agent.experience_replay()
        if (
            self.sl_agent.replay_buffer.size
            > self.sl_agent.config.min_replay_buffer_size
        ):
            print("Experience Replay for SL Agent")
            sl_loss = self.sl_agent.experience_replay()
        print("RL Loss", rl_loss)
        print("SL Loss", sl_loss)
        return rl_loss, sl_loss
