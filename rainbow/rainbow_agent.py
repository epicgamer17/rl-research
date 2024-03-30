import sys

from base_agent.agent import BaseAgent
from configs.agent_configs.rainbow_config import RainbowConfig

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
from rainbow.rainbow_network import Network


class Sample(NamedTuple):
    ids: np.ndarray
    indices: np.ndarray
    actions: np.ndarray
    observations: np.ndarray
    weights: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray


class RainbowAgent(BaseAgent):
    def __init__(
        self,
        env,
        config: RainbowConfig,
        name=datetime.datetime.now().timestamp(),
    ):
        super(RainbowAgent, self).__init__(env, config, name)
        self.model = Network(
            config, self.num_actions, input_shape=self.observation_dimensions
        )

        self.target_model = Network(
            config, self.num_actions, input_shape=self.observation_dimensions
        )

        self.model.compile(
            optimizer=self.config.optimizer(
                learning_rate=self.config.learning_rate,
                epsilon=self.config.adam_epsilon,
                clipnorm=self.config.clipnorm,
            ),
            loss=self.config.loss_function,
        )

        self.target_model.compile(
            optimizer=self.config.optimizer(
                learning_rate=self.config.learning_rate,
                epsilon=self.config.adam_epsilon,
                clipnorm=self.config.clipnorm,
            ),
            loss=self.config.loss_function,
        )

        self.target_model.set_weights(self.model.get_weights())

        self.replay_buffer = PrioritizedReplayBuffer(
            observation_dimensions=self.observation_dimensions,
            max_size=self.config.replay_buffer_size,
            batch_size=self.config.minibatch_size,
            max_priority=1.0,
            alpha=self.config.per_alpha,
            # epsilon=config["per_epsilon"],
            n_step=self.config.n_step,
            gamma=self.config.discount_factor,
        )

        self.n_step_replay_buffer = ReplayBuffer(
            observation_dimensions=self.observation_dimensions,
            max_size=self.config.replay_buffer_size,
            batch_size=self.config.minibatch_size,
            n_step=self.config.n_step,
            gamma=self.config.discount_factor,
        )

        # could use a MuZero min-max config and just constantly update the suport size (would this break the model?)
        # self.v_min = self.config.v_min
        # self.v_max = self.config.v_max

        self.support = np.linspace(
            self.config.v_min, self.config.v_max, self.config.atom_size
        )

        self.transition = list()
        self.is_test = True
        # self.search = search.Search(
        #     scoring_function=self.score_state,
        #     max_depth=config["search_max_depth"],
        #     max_time=config["search_max_time"],
        #     transposition_table=search.TranspositionTable(
        #         buckets=config["search_transposition_table_buckets"],
        #         bucket_size=config["search_transposition_table_bucket_size"],
        #         replacement_strategy=search.TranspositionTable.replacement_strategies[
        #             config["search_transposition_table_replacement_strategy"]
        #         ],
        #     ),
        #     debug=False,
        # )

    def predict_single(self, state):
        state_input = self.prepare_states(state)
        q_values = self.model(inputs=state_input).numpy()
        return q_values

    def select_action(self, state, legal_moves=None):
        q_values = np.sum(
            np.multiply(self.predict_single(state), np.array(self.support)), axis=2
        )
        selected_action = np.argmax(q_values)
        if not self.is_test:
            self.transition = [state, selected_action]
        return selected_action

    def step(self, action):
        if not self.is_test:
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            self.transition += [reward, next_state, done]
            # if self.use_n_step:
            one_step_transition = self.n_step_replay_buffer.store(*self.transition)
            # else:
            #     one_step_transition = self.transition

            if one_step_transition:
                self.replay_buffer.store(*one_step_transition)
        else:
            next_state, reward, terminated, truncated, info = self.test_env.step(action)

        return next_state, reward, terminated, truncated, info

    def experience_replay(self):
        for training_iteration in range(self.config.training_iterations):
            with tf.GradientTape() as tape:
                elementwise_loss = 0
                samples = self.replay_buffer.sample(self.config.per_beta)
                # actions = samples["actions"]
                # observations = samples["observations"]
                # inputs = self.prepare_states(observations)
                weights = samples["weights"].reshape(-1, 1)
                indices = samples["indices"]
                # discount_factor = self.config.discount_factor
                # target_ditributions = self.compute_target_distributions(
                #     samples, discount_factor
                # )
                # initial_distributions = self.model(inputs)
                # distributions_to_train = tf.gather_nd(
                #     initial_distributions,
                #     list(zip(range(initial_distributions.shape[0]), actions)),
                # )
                # changed this from self.model.loss.call to self.config.loss_function.call
                # elementwise_loss = self.config.loss_function.call(
                #     y_pred=distributions_to_train,
                #     y_true=tf.convert_to_tensor(target_ditributions),
                # )
                # assert np.all(elementwise_loss) >= 0, "Elementwise Loss: {}".format(
                #     elementwise_loss
                # )
                # if self.use_n_step:
                discount_factor = self.config.discount_factor**self.config.n_step
                n_step_samples = self.n_step_replay_buffer.sample_from_indices(indices)
                actions = n_step_samples["actions"]
                n_step_observations = n_step_samples["observations"]
                observations = n_step_observations
                inputs = self.prepare_states(observations)
                target_ditributions = self.compute_target_distributions(
                    n_step_samples, discount_factor
                )
                initial_distributions = self.model(inputs)
                distributions_to_train = tf.gather_nd(
                    initial_distributions,
                    list(zip(range(initial_distributions.shape[0]), actions)),
                )
                # changed this from self.model.loss.call to self.config.loss_function.call
                elementwise_loss_n_step = self.config.loss_function.call(
                    y_pred=distributions_to_train,
                    y_true=tf.convert_to_tensor(target_ditributions),
                )
                # add the losses together to reduce variance (original paper just uses n_step loss)
                # elementwise_loss += elementwise_loss_n_step
                elementwise_loss = elementwise_loss_n_step
                assert np.all(elementwise_loss) >= 0, "Elementwise Loss: {}".format(
                    elementwise_loss
                )

                loss = tf.reduce_mean(elementwise_loss * weights)

            # TRAINING WITH GRADIENT TAPE
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.config.optimizer(
                learning_rate=self.config.learning_rate,
                epsilon=self.config.adam_epsilon,
                clipnorm=self.config.clipnorm,
            ).apply_gradients(
                grads_and_vars=zip(gradients, self.model.trainable_variables)
            )
            # TRAINING WITH tf.train_on_batch
            # loss = self.model.train_on_batch(samples["observations"], target_ditributions, sample_weight=weights)

            prioritized_loss = elementwise_loss + self.config.per_epsilon
            # CLIPPING PRIORITIZED LOSS FOR ROUNDING ERRORS OR NEGATIVE LOSSES (IDK HOW WE ARE GETTING NEGATIVE LSOSES)
            prioritized_loss = np.clip(
                prioritized_loss, 0.01, tf.reduce_max(prioritized_loss)
            )
            self.replay_buffer.update_priorities(indices, prioritized_loss)
            self.model.reset_noise()
            self.target_model.reset_noise()
            loss = loss.numpy()
        # should return a loss for each iteration
        return loss

    def compute_target_distributions(self, samples, discount_factor):
        observations = samples["observations"]
        inputs = self.prepare_states(observations)
        next_observations = samples["next_observations"]
        next_inputs = self.prepare_states(next_observations)
        rewards = samples["rewards"].reshape(-1, 1)
        dones = samples["dones"].reshape(-1, 1)

        next_actions = np.argmax(np.sum(self.model(inputs).numpy(), axis=2), axis=1)
        target_network_distributions = self.target_model(next_inputs).numpy()

        target_distributions = target_network_distributions[
            range(self.config.minibatch_size), next_actions
        ]
        target_z = rewards + (1 - dones) * (discount_factor) * self.support
        target_z = np.clip(target_z, self.config.v_min, self.config.v_max)

        b = (
            (target_z - self.config.v_min) / (self.config.v_max - self.config.v_min)
        ) * (self.config.atom_size - 1)
        l, u = tf.cast(tf.math.floor(b), tf.int32), tf.cast(tf.math.ceil(b), tf.int32)
        m = np.zeros_like(target_distributions)
        assert m.shape == l.shape
        lower_distributions = target_distributions * (tf.cast(u, tf.float64) - b)
        upper_distributions = target_distributions * (b - tf.cast(l, tf.float64))

        for i in range(self.config.minibatch_size):
            np.add.at(m[i], np.asarray(l)[i], lower_distributions[i])
            np.add.at(m[i], np.asarray(u)[i], upper_distributions[i])
        target_distributions = m
        return target_distributions

    def compute_target_distributions_np(self, samples, discount_factor):
        observations = samples.observations
        inputs = self.prepare_states(observations)
        next_observations = samples.next_observations
        next_inputs = self.prepare_states(next_observations)
        rewards = samples.rewards.reshape(-1, 1)
        dones = samples.dones.reshape(-1, 1)

        next_actions = np.argmax(np.sum(self.model(inputs).numpy(), axis=2), axis=1)
        target_network_distributions = self.target_model(next_inputs).numpy()

        target_distributions = target_network_distributions[
            range(self.config.minibatch_size), next_actions
        ]
        target_z = rewards + (1 - dones) * (discount_factor) * self.support
        target_z = np.clip(target_z, self.config.v_min, self.config.v_max)

        b = (
            (target_z - self.config.v_min) / (self.config.v_max - self.config.v_min)
        ) * (self.config.atom_size - 1)
        l, u = tf.cast(tf.math.floor(b), tf.int32), tf.cast(tf.math.ceil(b), tf.int32)
        m = np.zeros_like(target_distributions)
        assert m.shape == l.shape
        lower_distributions = target_distributions * (tf.cast(u, tf.float64) - b)
        upper_distributions = target_distributions * (b - tf.cast(l, tf.float64))

        for i in range(self.config.minibatch_size):
            np.add.at(m[i], np.asarray(l)[i], lower_distributions[i])
            np.add.at(m[i], np.asarray(u)[i], upper_distributions[i])
        target_distributions = m
        return target_distributions

    def compute_target_distributions_np(self, samples: Sample, discount_factor):
        observations = samples.observations
        inputs = self.prepare_states(observations)
        next_observations = samples.next_observations
        next_inputs = self.prepare_states(next_observations)
        rewards = samples.rewards.reshape(-1, 1)
        dones = samples.dones.reshape(-1, 1)

        next_actions = np.argmax(np.sum(self.model(inputs).numpy(), axis=2), axis=1)
        target_network_distributions = self.target_model(next_inputs).numpy()

        target_distributions = target_network_distributions[
            range(self.replay_batch_size), next_actions
        ]
        target_z = rewards + (1 - dones) * (discount_factor) * self.support
        target_z = np.clip(target_z, self.v_min, self.v_max)

        b = ((target_z - self.v_min) / (self.v_max - self.v_min)) * (self.atom_size - 1)
        l, u = tf.cast(tf.math.floor(b), tf.int32), tf.cast(tf.math.ceil(b), tf.int32)
        m = np.zeros_like(target_distributions)
        assert m.shape == l.shape
        lower_distributions = target_distributions * (tf.cast(u, tf.float64) - b)
        upper_distributions = target_distributions * (b - tf.cast(l, tf.float64))

        for i in range(self.replay_batch_size):
            np.add.at(m[i], np.asarray(l)[i], lower_distributions[i])
            np.add.at(m[i], np.asarray(u)[i], upper_distributions[i])
        target_distributions = m
        return target_distributions

    # def score_state(self, state, turn):
    #     state_input = self.prepare_state(state)
    #     q = self.predict(state_input)

    #     if (turn % 2) == 0:
    #         return q.max(), q.argmax()

    #     return q.min(), q.argmin()

    # def play_optimal_move(
    #     self, state: bb.Bitboard, turn: int, max_depth: int, with_output=True
    # ):
    #     # q_value, action = self.alpha_beta_pruning(state, turn, max_depth=max_depth)
    #     q_value, action = self.search.iterative_deepening(state, turn, max_depth)
    #     if with_output:
    #         print("Evaluation: {}".format(q_value))
    #         print("Action: {}".format(action + 1))
    #     state.move(turn % 2, action)
    #     winner, _ = state.check_victory()

    #     if winner == 0:
    #         return False
    #     else:
    #         return True

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
            if step % self.config.transfer_interval == 0 and (
                len(self.replay_buffer) >= self.config.minibatch_size
            ):
                self.target_model.set_weights(self.model.get_weights())

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
        state, _ = self.env.reset()
        score = 0
        for training_step in range(self.config.training_steps):
            for _ in range(self.config.replay_interval):
                action = self.select_action(state)

                next_state, reward, terminated, truncated, info = self.step(action)
                done = terminated or truncated
                state = next_state
                score += reward
                self.config.per_beta = min(
                    1.0,
                    self.config.per_beta
                    + (1 - self.config.per_beta)
                    / self.config.training_steps,  # per beta increase
                )

                if done:
                    state, _ = self.env.reset()
                    stats["score"].append(score)
                    score = 0
            for minibatch in range(self.config.num_minibatches):
                loss = self.experience_replay()
                stats["loss"].append(loss)

            if training_step % self.config.transfer_interval == 0:
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
