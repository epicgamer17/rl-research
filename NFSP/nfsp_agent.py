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
                env, rl_configs[i], sl_configs[i], f"NFSPAgent_{i}", agent_type, i
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

        training_step = 0
        while (
            training_step < self.training_steps
        ):  # change to training steps and just make it so it ends a game even if the training steps are done
            for p in range(self.config.num_players):
                self.nfsp_agents[p].select_policy(self.config.anticipatory_param)
            state, info = self.env.reset()
            done = False
            while not done:
                # for _ in range(self.config.replay_interval):
                # for current_player in range(self.config.num_players):
                current_player = state["current_player"]
                current_agent = self.nfsp_agents[current_player]
                action = current_agent.select_action(
                    state,
                    (info["legal_moves"] if self.config.game.has_legal_moves else None),
                )

                current_agent.config.per_beta = min(
                    1.0,
                    current_agent.rl_agent.config.per_beta
                    + (1 - current_agent.rl_agent.config.per_beta)
                    / self.training_steps,  # per beta increase
                )

                # self.nfsp_agents[0].rl_agent.transition = [state_p1, action_p1]
                next_state, rewards, terminated, truncated, info = self.step(action)
                state = next_state
                done = terminated or truncated

                training_step += 1
                for minibatch in range(self.config.num_minibatches):
                    rl_loss, sl_loss = self.experience_replay()

                if training_step % self.config.transfer_interval == 0:
                    self.target_model.set_weights(self.rl_agent.model.get_weights())

                if training_step % self.checkpoint_interval == 0 and training_step > 0:
                    for p in range(self.config.num_players):
                        self.nfsp_agents[p].save_checkpoint(
                            stats,
                            targets,
                            5,
                            training_step,
                            training_step * self.config.replay_interval,
                            time() - training_time,
                        )
                    # CALL CHECKPOINTING ON THE INDIVIDUAL AGENTS

                if not done:
                    current_agent.step(
                        action, state, rewards[current_player], done
                    )  # stores experiences

            for p in range(self.config.num_players):
                self.nfsp_agents[p].step(action, state, rewards[p], done)
        self.env.close()

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
    def __init__(self, env, rl_config, sl_config, name, agent_type, player) -> None:
        super().__init__(env, rl_config, name)
        self.rl_agent = agent_type(env, rl_config, name)
        self.sl_agent = AverageStrategyAgent(env, sl_config, name)
        self.policy = "best_response"  # "average_strategy" or "best_response
        # transition = [state, action, reward, next_state, done]
        self.previous_state = None
        self.previous_action = None
        self.previous_reward = None

    def step(self, action, state, reward, done):
        self.transition = [self.previous_state, self.previous_action]
        if (
            self.previous_action is not None
            and self.previous_state is not None
            and self.previous_reward is not None
        ):
            self.transition += [self.previous_reward, state, done]
            # only do this if it is average policy? (open spiel)
            self.rl_agent.replay_buffer.store(*self.transition)
        if not done:
            self.previous_state = state
            self.previous_action = action
            self.previous_reward = reward
        else:
            self.previous_state = None
            self.previous_action = None
            self.previous_reward = None

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
        if (
            self.rl_agent.replay_buffer.size
            > self.rl_agent.config.min_replay_buffer_size
        ):  # only do if not in rl agent mode? (open spiel)
            rl_loss = self.rl_agent.experience_replay()
        if (
            self.sl_agent.replay_buffer.size
            > self.sl_agent.config.min_replay_buffer_size
        ):
            sl_loss = self.sl_agent.experience_replay()
        return rl_loss, sl_loss
