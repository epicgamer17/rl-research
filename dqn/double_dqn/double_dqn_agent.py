import os
from agent_configs import RainbowConfig
from utils import update_per_beta, action_mask, get_legal_moves

import sys

sys.path.append("../")

from base_agent.agent import BaseAgent

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
        # Memory growth must be set before GPUs have been initialized
        print(e)

import datetime
import numpy as np
import copy
import matplotlib.pyplot as plt
from IPython.display import clear_output
from typing import NamedTuple

# import search
from typing import Deque, Dict, List, Tuple
import gymnasium as gym
from time import time

# import moviepy
from replay_buffers.n_step_replay_buffer import ReplayBuffer
from replay_buffers.prioritized_replay_buffer import (
    PrioritizedReplayBuffer,
    FastPrioritizedReplayBuffer,
)
from dqn.dqn_network import Network


class Sample(NamedTuple):
    ids: np.ndarray
    indices: np.ndarray
    actions: np.ndarray
    observations: np.ndarray
    weights: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray


class DQNAgent(BaseAgent):
    def __init__(
        self,
        env,
        config: DQNConfig,
        name=datetime.datetime.now().timestamp(),
    ):
        super(DQNAgent, self).__init__(env, config, name)
        self.config = config
        self.model = Network(
            config, self.num_actions, input_shape=self.observation_dimensions
        )
        self.target_model = Network(
            config, self.num_actions, input_shape=self.observation_dimensions
        )

        self.model(np.zeros((1,) + self.observation_dimensions))
        self.target_model(np.zeros((1,) + self.observation_dimensions))

        self.target_model.set_weights(self.model.get_weights())

        self.replay_buffer = ReplayBuffer(
            observation_dimensions=self.observation_dimensions,
            max_size=self.config.replay_buffer_size,
            batch_size=self.config.minibatch_size,
            n_step=1,
            gamma=self.config.discount_factor,
        )

        self.transition = list()

    def predict_single(self, state, legal_moves=None):
        state_input = self.prepare_states(state)
        q_values = self.model(inputs=state_input).numpy()
        q_values = action_mask(
            q_values, legal_moves, self.num_actions, mask_value=-np.inf
        )
        return q_values

    def select_action(self, state, legal_moves=None):
        q_values = self.predict_single(state, legal_moves)
        selected_action = np.argmax(q_values)
        if not self.is_test:
            self.transition = [state, selected_action]
        return selected_action

    def step(self, action):
        if not self.is_test:
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            self.transition += [reward, next_state, done]
            self.replay_buffer.store(*self.transition)
        else:
            next_state, reward, terminated, truncated, info = self.test_env.step(action)

        return next_state, reward, terminated, truncated, info

    def update_target_model(self, step):

        if self.config.soft_update:
            new_weights = self.target_model.get_weights()

            counter = 0
            for wt, wp in zip(
                self.target_model.get_weights(),
                self.model.get_weights(),
            ):
                wt = (self.config.ema_beta * wt) + ((1 - self.config.ema_beta) * wp)
                new_weights[counter] = wt
                counter += 1
            self.target_model.set_weights(new_weights)
        else:
            self.target_model.set_weights(self.model.get_weights())

    def learn(self):
        for training_iteration in range(self.config.training_iterations):
            with tf.GradientTape() as tape:
                samples = self.replay_buffer.sample(self.config.per_beta)
                discount_factor = self.config.discount_factor
                actions = samples["actions"]
                observations = samples["observations"]
                inputs = self.prepare_states(observations)
                next_observations = samples["next_observations"]
                next_inputs = self.prepare_states(next_observations)
                rewards = samples["rewards"].reshape(-1, 1)
                dones = samples["dones"].reshape(-1, 1)
                next_q_values = self.model(next_inputs)
                initial_q_values = self.model(inputs)
                targets = rewards + (1 - dones) * discount_factor * np.max(
                    next_q_values
                )
                loss = tf.keras.losses.MSE.call(
                    y_pred=initial_q_values[actions],
                    y_true=targets,
                )

            # TRAINING WITH GRADIENT TAPE
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.config.optimizer.apply_gradients(
                grads_and_vars=zip(gradients, self.model.trainable_variables)
            )
            loss = loss.numpy()
        # should return a loss for each iteration
        return loss

    def fill_replay_buffer(self):
        state, _ = self.env.reset()
        for experience in range(self.config.min_replay_buffer_size):
            action = self.env.action_space.sample()
            self.transition = [state, action]

            next_state, reward, terminated, truncated, info = self.step(action)
            done = terminated or truncated
            state = next_state
            if done:
                state, _ = self.env.reset()

    def train(self):
        training_time = time()
        self.is_test = False
        stats = {
            "score": [],
            "loss": [],
            "test_score": [],
        }
        targets = {
            "score": self.env.spec.reward_threshold,
            "test_score": self.env.spec.reward_threshold,
        }

        self.fill_replay_buffer()
        state, info = self.env.reset()
        score = 0
        target_model_updated = (False, False)  # (score, loss)

        for training_step in range(self.training_steps):
            for _ in range(self.config.replay_interval):
                action = self.select_action(
                    state,
                    get_legal_moves(info),
                )

                next_state, reward, terminated, truncated, info = self.step(action)
                done = terminated or truncated
                state = next_state
                score += reward
                self.config.per_beta = update_per_beta(
                    self.config.per_beta, 1.0, self.training_steps
                )

                if done:
                    state, info = self.env.reset()
                    stats["score"].append(
                        {
                            "score": score,
                            "target_model_updated": target_model_updated[0],
                        }
                    )
                    target_model_updated = (False, target_model_updated[1])
                    score = 0

            for minibatch in range(self.config.num_minibatches):
                loss = self.learn()
                stats["loss"].append(
                    {"loss": loss, "target_model_updated": target_model_updated[1]}
                )
                target_model_updated = (target_model_updated[0], False)

            if training_step % self.config.transfer_interval == 0:
                target_model_updated = (True, True)
                # stats["test_score"].append(
                #     {"target_model_weight_update": training_step}
                # )
                self.update_target_model(training_step)

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
