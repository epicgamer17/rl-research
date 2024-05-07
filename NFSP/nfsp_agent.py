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


class NFSPAgent(BaseAgent):
    def __init__(self, env, config: NFSPConfig, name, agent_type) -> None:
        super().__init__(env, config, name)
        rl_configs = self.config.rl_configs
        self.rl_agents = [
            agent_type(env, rl_configs[player], name)
            for player in range(self.config.num_players)
        ]

        sl_configs = self.config.sl_configs
        self.sl_agents = [
            AverageStrategyAgent(env, sl_configs[player], name)
            for player in range(self.config.num_players)
        ]

        self.current_agent = 0
        self.checkpoint_interval = 10000

    def train(self):
        training_time = time()
        self.is_test = False
        stats = {
            "score": [],
            "rl_loss": [],
            "sl_loss": [],
            "test_score": [],
        }
        targets = {
            "score": self.env.spec.reward_threshold,
            "test_score": self.env.spec.reward_threshold,
        }

        state, info = self.env.reset()
        score = 0

        rewards = np.zeros(self.config.num_players)
        states = np.zeros((self.config.num_players,) + self.observation_dimensions)
        next_states = np.zeros((self.config.num_players,) + self.observation_dimensions)
        dones = np.zeros(self.config.num_players)

        for training_step in range(self.training_steps):
            for p in range(self.config.num_players):
                for _ in range(self.config.replay_interval):
                    states[self.current_agent] = state
                    action = self.select_action(
                        state,
                        (
                            info["legal_moves"]
                            if self.config.game.has_legal_moves
                            else None
                        ),
                    )

                    (
                        next_states[self.current_agent],
                        rewards[self.current_agent],
                        terminated,
                        truncated,
                        info,
                    ) = self.step(
                        action
                    )  # Stores RL Experiences in step function

                    one_step_transition = self.rl_agents[
                        self.current_agent
                    ].n_step_replay_buffer.store(
                        *self.rl_agents[self.current_agent].transition
                    )

                    if one_step_transition:
                        self.rl_agents[self.current_agent].replay_buffer.store(
                            *one_step_transition
                        )

                    dones[self.current_agent] = terminated or truncated
                    state = next_states[self.current_agent]
                    score += rewards[self.current_agent]
                    self.config.rl_configs[self.current_agent].per_beta = min(
                        1.0,
                        self.config.rl_configs[self.current_agent].per_beta
                        + (1 - self.config.rl_configs[self.current_agent].per_beta)
                        / self.training_steps,  # per beta increase
                    )

                    if dones[self.current_agent]:
                        dones[self.current_agent :] = 1
                        state, info = self.env.reset()
                        stats["score"].append(score)  # might be irrelevant for NFSP
                        score = 0

                for minibatch in range(self.config.num_minibatches):
                    rl_loss, sl_loss = self.experience_replay()
                    stats["rl_loss"].append(
                        rl_loss
                    )  # may want to average since it could be noisy between the different agents
                    stats["sl_loss"].append(sl_loss)

                if (
                    training_step
                    % self.config.rl_configs[self.current_agent].transfer_interval
                    == 0
                ):
                    self.rl_agents[self.current_agent].update_target_model(
                        training_step
                    )  # Update target model for the current RL agent
                self.current_agent = info["player"] if "player" in info else 0

            if training_step % self.checkpoint_interval == 0 and training_step > 0:
                self.save_checkpoint(
                    stats,
                    targets,
                    5,
                    training_step,
                    training_step * self.config.replay_interval,
                    time() - training_time,
                )

        self.save_checkpoint(
            stats,
            targets,
            5,
            training_step,
            training_step * self.config.replay_interval,
            time() - training_time,
        )
        self.env.close()

    def select_action(self, state, legal_moves=None):
        if random.random() < self.config.anticipatory_param and not self.is_test:
            print("Selected Action from RL Agent")
            action = self.rl_agents[self.current_agent].select_action(
                state, legal_moves
            )
            self.sl_agents[self.current_agent].replay_buffer.store(
                state, action
            )  # Store best moves in SL Memory
        else:
            print("Selected Action from SL Agent")
            action = self.sl_agents[self.current_agent].select_action(
                state, legal_moves
            )
            self.rl_agents[self.current_agent].transition = [state, action]

        return action

    def experience_replay(self):
        rl_loss = 0
        sl_loss = 0
        if (
            self.rl_agents[self.current_agent].replay_buffer.size
            > self.config.rl_configs[self.current_agent].min_replay_buffer_size
        ):
            rl_loss = self.rl_agents[self.current_agent].experience_replay()
        if (
            self.sl_agents[self.current_agent].replay_buffer.size
            > self.config.sl_configs[self.current_agent].min_replay_buffer_size
        ):
            sl_loss = self.sl_agents[self.current_agent].experience_replay()
        return rl_loss, sl_loss

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
        if not self.is_test:
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            self.rl_agents[self.current_agent].transition += [reward, next_state, done]
        else:
            next_state, reward, terminated, truncated, info = self.test_env.step(action)

        return next_state, reward, terminated, truncated, info
