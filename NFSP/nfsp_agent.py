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

import copy
import sys
from time import time

from agent_configs import NFSPConfig
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
from configs.agent_configs.agent_configs.nfsp_config import NFSPConfig


class NFSP:
    def __init__(self, env, config: NFSPConfig, agent_type) -> None:
        self.env = env
        self.config = config
        self.agent_type = agent_type
        rl_configs = self.config.rl_configs
        sl_configs = self.config.sl_configs
        self.nfsp_agents = [
            NFSPAgent(env, rl_configs[i], sl_configs[i], f"NFSPAgent_{i}", agent_type)
            for i in range(self.config.num_players)
        ]
        self.training_steps = self.config.training_steps

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

        state, info = self.env.reset()

        for p in range(self.config.num_players):
            self.nfsp_agents[p].select_policy(self.config.anticipatory_param)

        for training_step in range(self.training_steps):
            for _ in range(self.config.replay_interval):
                states = [None] * self.config.num_players
                next_states = [None] * self.config.num_players
                actions = [None] * self.config.num_players
                rewards = [0] * self.config.num_players
                dones = [0] * self.config.num_players
                transitions = [None] * self.config.num_players

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
                    # STORE AFTER ONE ROUND FOR ALL AGENTS!!!
                    next_states[p] = state
                    if states[p] is not None:
                        transitions = [
                            states[p],
                            actions[p],
                            rewards[p],
                            next_states[p],
                            dones[p],
                        ]

                    self.nfsp_agents[p].rl_agent.config.per_beta = min(
                        1.0,
                        self.nfsp_agents[p].rl_agent.config.per_beta
                        + (1 - self.nfsp_agents[p].rl_agent.config.per_beta)
                        / self.training_steps,  # per beta increase
                    )

                    if done:
                        # assume sparse rewards/zero sum (so if game is done, all players are done and rewards should be updated accordingly)
                        # game should return a list or dictionary of rewards for each player
                        rewards = reward  # will be a list (for leduc holdem at least)
                        state, info = self.env.reset()
                        transitions = [None] * self.config.num_players
                    else:
                        rewards[p] = reward
                    states[p] = state
                    actions[p] = action
                    dones[p] = done

                    state = next_state

                for p in range(self.config.num_players):
                    if transitions[p] is not None:
                        one_step_transition = self.nfsp_agents[
                            p
                        ].rl_agent.n_step_replay_buffer.store(*transitions[p])
                        print(one_step_transition)
                        if one_step_transition:
                            self.nfsp_agents[p].rl_agent.replay_buffer.store(
                                *one_step_transition
                            )

                    for minibatch in range(self.config.num_minibatches):
                        self.nfsp_agents[p].experience_replay()
                        # rl_loss, sl_loss = self.experience_replay()
                        # stats["rl_loss"].append(
                        #     rl_loss
                        # )  # may want to average since it could be noisy between the different agents
                        # stats["sl_loss"].append(sl_loss)

                    if training_step % self.config.rl_configs[p].transfer_interval == 0:
                        self.nfsp_agents[p].rl_agent.target_model.set_weights(
                            self.nfsp_agents[p].rl_agent.model.get_weights()
                        )

        #     if training_step % self.checkpoint_interval == 0 and training_step > 0:
        #         self.save_checkpoint(
        #             stats,
        #             targets,
        #             5,
        #             training_step,
        #             training_step * self.config.replay_interval,
        #             time() - training_time,
        #         )

        # self.save_checkpoint(
        #     stats,
        #     targets,
        #     5,
        #     training_step,
        #     training_step * self.config.replay_interval,
        #     time() - training_time,
        # )
        self.env.close()

    def save_checkpoint(
        self, stats, targets, num_trials, training_step, frames_seen, time_taken
    ):
        # save the model weights
        if not os.path.exists("./model_weights"):
            os.makedirs("./model_weights")
        if not os.path.exists("./model_weights/{}".format(self.model_name)):
            os.makedirs("./model_weights/{}".format(self.model_name))

        path = "./model_weights/{}/episode_{}.keras".format(
            self.model_name, training_step
        )

        # for agent in self.rl_agents:
        #     agent.model.save(path)
        # for agent in self.sl_agents:
        #     agent.model.save(path)

        self.rl_agents[0].model.save(path)
        self.sl_agents[0].model.save(path)

        # save replay buffer
        # save optimizer

        # test model
        test_score = self.test(num_trials, training_step)
        stats["test_score"].append(test_score)
        # plot the graphs
        self.plot_graph(stats, targets, training_step, frames_seen, time_taken)

    def step(self, action):
        return self.env.step(action)


class NFSPAgent(BaseAgent):
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
            self.sl_agent.replay_buffer.store(
                state, action
            )  # Store best moves in SL Memory
            return action

        return action

    def experience_replay(self):
        rl_loss = 0
        sl_loss = 0
        if (
            self.rl_agent.replay_buffer.size
            > self.rl_agent.config.min_replay_buffer_size
        ):
            rl_loss = self.rl_agent.experience_replay()
        if (
            self.sl_agent.replay_buffer.size
            > self.sl_agent.config.min_replay_buffer_size
        ):
            sl_loss = self.sl_agent.experience_replay()
        return rl_loss, sl_loss
