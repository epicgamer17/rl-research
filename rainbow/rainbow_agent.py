import sys

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


class RainbowAgent:
    def __init__(
        self,
        env,
        model_name=datetime.datetime.now().timestamp(),
        config=None,
    ):
        self.config = config
        self.model_name = model_name
        self.env = env
        self.test_env = copy.deepcopy(env)
        self.observation_dimensions = env.observation_space.shape
        self.num_actions = env.action_space.n

        self.start_episode = 0

        self.model = Network(
            config, self.num_actions, input_shape=self.observation_dimensions
        )

        self.target_model = Network(
            config, self.num_actions, input_shape=self.observation_dimensions
        )

        self.optimizer = config["optimizer"]
        self.adam_epsilon = config["adam_epsilon"]
        self.learning_rate = config["learning_rate"]
        self.loss_function = config["loss_function"]
        self.clipnorm = config["clipnorm"]

        self.model.compile(
            optimizer=self.optimizer(
                learning_rate=self.learning_rate,
                epsilon=self.adam_epsilon,
                clipnorm=self.clipnorm,
            ),
            loss=config["loss_function"],
        )

        self.target_model.compile(
            optimizer=self.optimizer(
                learning_rate=self.learning_rate,
                epsilon=self.adam_epsilon,
                clipnorm=self.clipnorm,
            ),
            loss=config["loss_function"],
        )

        self.target_model.set_weights(self.model.get_weights())

        self.num_training_steps = int(config["num_training_steps"])

        self.discount_factor = config["discount_factor"]

        self.replay_batch_size = int(config["replay_batch_size"])
        self.replay_period = int(config["replay_period"])
        self.replay_buffer_size = max(
            int(config["replay_buffer_size"]), self.replay_batch_size
        )
        self.min_replay_buffer_size = int(config["min_replay_buffer_size"])

        self.soft_update = config["soft_update"]
        self.transfer_frequency = int(config["transfer_frequency"])
        self.ema_beta = config["ema_beta"]

        self.per_beta = config["per_beta"]
        # self.per_beta_increase = config["per_beta_increase"]
        self.per_beta_increase = (1 - self.per_beta) / self.num_training_steps
        self.per_epsilon = config["per_epsilon"]

        self.replay_buffer = PrioritizedReplayBuffer(
            observation_dimensions=self.observation_dimensions,
            max_size=self.replay_buffer_size,
            batch_size=self.replay_batch_size,
            max_priority=1.0,
            alpha=config["per_alpha"],
            # epsilon=config["per_epsilon"],
            n_step=config["n_step"],
            gamma=config["discount_factor"],
        )

        self.use_n_step = config["n_step"] > 1

        self.n_step = config["n_step"]

        if self.use_n_step:
            self.n_step_replay_buffer = ReplayBuffer(
                observation_dimensions=self.observation_dimensions,
                max_size=self.replay_buffer_size,
                batch_size=self.replay_batch_size,
                n_step=self.n_step,
                gamma=config["discount_factor"],
            )

        self.v_min = config["v_min"]
        self.v_max = config["v_max"]

        self.atom_size = config["atom_size"]
        self.support = np.linspace(self.v_min, self.v_max, self.atom_size)

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

    def save_checkpoint(self, episode=-1, best_model=False):
        if episode != -1:
            path = "./{}_{}_episodes.keras".format(
                self.model_name, episode + self.start_episode
            )
        else:
            path = "./{}.keras".format(self.model_name)

        if best_model:
            path = "./best_model.keras"

        self.model.save(path)

    def prepare_states(self, state):
        state = np.array(state)
        if (self.env.observation_space.high == 255).all():
            state = state / 255
        if state.shape == self.observation_dimensions:
            new_shape = (1,) + state.shape
            state_input = state.reshape(new_shape)
        else:
            state_input = state
        return state_input

    def predict_single(self, state):
        state_input = self.prepare_states(state)
        q_values = self.model(inputs=state_input).numpy()
        return q_values

    def select_action(self, state):
        q_values = np.sum(
            np.multiply(self.predict_single(state), np.array(self.support)), axis=2
        )
        selected_action = np.argmax(q_values)
        if not self.is_test:
            self.transition = [state, selected_action]
        return selected_action

    def step(self, action):
        if not self.is_test:
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            self.transition += [reward, next_state, done]
            if self.use_n_step:
                one_step_transition = self.n_step_replay_buffer.store(*self.transition)
            else:
                one_step_transition = self.transition

            if one_step_transition:
                self.replay_buffer.store(*one_step_transition)
        else:
            next_state, reward, terminated, truncated, _ = self.test_env.step(action)

        return next_state, reward, terminated, truncated

    def experience_replay(self):
        with tf.GradientTape() as tape:
            elementwise_loss = 0
            samples = self.replay_buffer.sample(self.per_beta)
            actions = samples["actions"]
            observations = samples["observations"]
            inputs = self.prepare_states(observations)
            weights = samples["weights"].reshape(-1, 1)
            indices = samples["indices"]
            discount_factor = self.discount_factor
            target_ditributions = self.compute_target_distributions(
                samples, discount_factor
            )
            initial_distributions = self.model(inputs)
            distributions_to_train = tf.gather_nd(
                initial_distributions,
                list(zip(range(initial_distributions.shape[0]), actions)),
            )
            elementwise_loss = self.model.loss.call(
                y_pred=distributions_to_train,
                y_true=tf.convert_to_tensor(target_ditributions),
            )
            assert np.all(elementwise_loss) >= 0, "Elementwise Loss: {}".format(
                elementwise_loss
            )
            if self.use_n_step:
                discount_factor = self.discount_factor**self.n_step
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
                elementwise_loss_n_step = self.model.loss.call(
                    y_pred=distributions_to_train,
                    y_true=tf.convert_to_tensor(target_ditributions),
                )
                # add the losses together to reduce variance (original paper just uses n_step loss)
                elementwise_loss += elementwise_loss_n_step
                assert np.all(elementwise_loss) >= 0, "Elementwise Loss: {}".format(
                    elementwise_loss
                )

            loss = tf.reduce_mean(elementwise_loss * weights)

        # TRAINING WITH GRADIENT TAPE
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer(
            learning_rate=self.learning_rate,
            epsilon=self.adam_epsilon,
            clipnorm=self.clipnorm,
        ).apply_gradients(grads_and_vars=zip(gradients, self.model.trainable_variables))
        # TRAINING WITH tf.train_on_batch
        # loss = self.model.train_on_batch(samples["observations"], target_ditributions, sample_weight=weights)

        prioritized_loss = elementwise_loss + self.per_epsilon
        # CLIPPING PRIORITIZED LOSS FOR ROUNDING ERRORS OR NEGATIVE LOSSES (IDK HOW WE ARE GETTING NEGATIVE LSOSES)
        prioritized_loss = np.clip(
            prioritized_loss, 0.01, tf.reduce_max(prioritized_loss)
        )
        self.replay_buffer.update_priorities(indices, prioritized_loss)
        self.model.reset_noise()
        self.target_model.reset_noise()
        loss = loss.numpy()
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

    def action_mask(self, q, state, turn):
        q_copy = copy.deepcopy(q)
        for i in range(len(q_copy)):
            if not state.is_valid_move(i):
                if turn % 2 == 0:
                    q_copy[i] = float("-inf")
                else:
                    q_copy[i] = float("inf")
        return q_copy

    def fill_replay_buffer(self):
        state, _ = self.env.reset()
        # print(state)
        for experience in range(self.min_replay_buffer_size):
            # clear_output(wait=False)
            # print("Filling replay_buffer")
            # print("replay_buffer Size: {}/{}".format(experience, self.min_replay_buffer_size))
            # state_input = self.prepare_state(state)
            action = self.env.action_space.sample()
            self.transition = [state, action]

            next_state, reward, terminated, truncated = self.step(action)
            done = terminated or truncated
            state = next_state
            if done:
                state, _ = self.env.reset()

    def update_target_model(self, step):
        # print("Updating Target Model")
        # time1 = 0
        # time1 = time()
        if self.soft_update:
            new_weights = self.target_model.get_weights()

            counter = 0
            for wt, wp in zip(
                self.target_model.get_weights(),
                self.model.get_weights(),
            ):
                wt = (self.ema_beta * wt) + ((1 - self.ema_beta) * wp)
                new_weights[counter] = wt
                counter += 1
            self.target_model.set_weights(new_weights)
        else:
            if step % self.transfer_frequency == 0 and (
                len(self.replay_buffer) >= self.replay_batch_size
            ):
                self.target_model.set_weights(self.model.get_weights())
        # print("Updating Target Model Time ", time() - time1)

    def train(self, graph_interval=200):
        self.is_test = False
        stat_score = (
            []
        )  # make these num trials divided by graph interval so i dont need to append (to make it faster?)
        stat_test_score = []
        stat_loss = []
        self.fill_replay_buffer()
        num_trials_truncated = 0
        state, _ = self.env.reset()
        model_update_count = 0
        score = 0
        training_step = 0
        step = 0
        while training_step < self.num_training_steps:
            # state_input = self.prepare_state(state)
            # clear_output(wait=False)
            # print("Last Training Score: ", stat_score[-1] if len(stat_score) > 0 else 0)
            # print("Last Training Loss: ", stat_loss[-1] if len(stat_loss) > 0 else 0)
            action = self.select_action(state)

            next_state, reward, terminated, truncated = self.step(action)
            done = terminated or truncated
            state = next_state
            score += reward
            self.per_beta = min(1.0, self.per_beta + self.per_beta_increase)

            if done:
                state, _ = self.env.reset()
                stat_score.append(score)
                score = 0

            if (step % self.replay_period) == 0 and (
                len(self.replay_buffer) >= self.replay_batch_size
            ):
                model_update_count += 1
                loss = self.experience_replay()
                training_step += 1
                stat_loss.append(loss)

                self.update_target_model(model_update_count)

            if training_step % graph_interval == 0 and training_step > 0:
                self.save_checkpoint()
                # stat_test_score.append(self.test())
                self.plot_graph(stat_score, stat_loss, stat_test_score, training_step)
                print(
                    "{} Training Step: {}/{}".format(
                        self.model_name, training_step, self.num_training_steps
                    )
                )
            step += 1

        self.plot_graph(stat_score, stat_loss, stat_test_score, training_step)
        self.save_checkpoint()
        self.env.close()
        return num_trials_truncated / self.num_training_steps

    def plot_graph(self, score, loss, test_score, step):
        fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(30, 5))
        ax1.plot(score, linestyle="solid")
        ax1.set_title("Frame {}. Score: {}".format(step, np.mean(score[-10:])))
        ax2.plot(loss, linestyle="solid")
        ax2.set_title("Frame {}. Loss: {}".format(step, np.mean(loss[-10:])))
        ax3.plot(test_score, linestyle="solid")
        if self.env.spec.reward_threshold is not None:
            ax3.axhline(y=self.env.spec.reward_threshold, color="r", linestyle="-")
        ax3.set_title(
            "Frame {}. Test Score: {}".format(step, np.mean(test_score[-10:]))
        )
        plt.savefig("./{}.png".format(self.model_name))
        plt.close(fig)

    def test(self, num_trials=100, video_folder="") -> None:
        """Test the agent."""
        self.is_test = True
        average_score = 0
        for trials in range(num_trials - 1):
            state, _ = self.test_env.reset()
            done = False
            score = 0

            while not done:
                action = self.select_action(state)
                next_state, reward, terminated, truncated = self.step(action)
                done = terminated or truncated
                state = next_state

                score += reward
            average_score += score
            print("score: ", score)

        if video_folder == "":
            video_folder = "./videos/{}".format(self.model_name)
        # for recording a video
        self.test_env = gym.wrappers.RecordVideo(self.test_env, video_folder)
        state, _ = self.test_env.reset()
        done = False
        score = 0

        while not done:
            action = self.select_action(state)
            next_state, reward, terminated, truncated = self.step(action)
            done = terminated or truncated
            state = next_state

            score += reward

        print("score: ", score)
        average_score += score
        self.test_env.close()

        # reset
        self.is_test = False
        average_score /= num_trials
        return average_score
