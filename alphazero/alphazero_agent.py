import sys

sys.path.append("../")

import os

# os.environ["OMP_NUM_THREADS"] = f"{1}"
# os.environ['TF_NUM_INTEROP_THREADS'] = f"{1}"
# os.environ['TF_NUM_INTRAOP_THREADS'] = f"{1}"

import tensorflow as tf

# tf.config.threading.set_intra_op_parallelism_threads(1)
# tf.config.threading.set_inter_op_parallelism_threads(1)

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
import copy
import numpy as np
from alphazero.alphazero_network import Network

from memory.alphazero_replay_buffer import ReplayBuffer
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import gymnasium as gym

import alphazero.MTCS_alphazero as MCTS


class AlphaZeroAgent:
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
        self.model = Network(
            input_shape=self.observation_dimensions,
            output_shape=self.num_actions,
            config=config,
        )

        self.optimizer = config["optimizer_function"]
        self.adam_epsilon = config["adam_epsilon"]
        self.learning_rate = config["learning_rate"]
        # self.clipnorm = config["clipnorm"]
        self.clipnorm = None

        self.num_epochs = int(config["num_epochs"])
        # self.replay_batch_size = int(config["replay_batch_size"])
        self.memory_size = config["memory_size"]
        self.max_game_length = config["max_game_length"]

        self.memory = ReplayBuffer(
            observation_dimensions=self.observation_dimensions,
            max_size=self.memory_size,
            gamma=config["discount_factor"],
        )

        self.dirichlet_concentration = config["dirichlet_concentration"]
        self.dirichlet_epsilon = config["dirichlet_epsilon"]

        self.transition = list()
        self.is_test = True

    def export(self, episode=-1, best_model=False):
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
        if (self.env.observation_space.high == 255).all():
            state = np.array(state) / 255
        # print(state.shape)
        if state.shape == self.observation_dimensions:
            new_shape = (1,) + state.shape
            state_input = state.reshape(new_shape)
        else:
            state_input = state
        return state_input

    def predict_single(self, state):
        state_input = self.prepare_states(state)
        value, probabilities = self.model(inputs=state_input).numpy()
        return probabilities, value

    def select_action(self, state):
        probabilities, value = self.predict_single(state)
        distribution = tfp.distributions.Categorical(probs=probabilities)
        selected_action = distribution.sample().numpy()
        if len(selected_action) == 1:
            selected_action = selected_action[0]
        value = value[0][0]
        if not self.is_test:
            self.transition = [state]
        return selected_action

    def step(self, action):
        if not self.is_test:
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            self.transition += [reward]
            self.memory.store(*self.transition)
        else:
            next_state, reward, terminated, truncated, _ = self.test_env.step(action)

        return next_state, reward, terminated, truncated

    def experience_replay(self):
        samples = self.memory.sample()
        observations = samples["observations"]
        action_probabilities = samples["action_probabilities"]
        rewards = samples["rewards"]
        print(rewards)
        inputs = self.prepare_states(observations)
        with tf.GradientTape() as tape:
            value, probabilities = self.model(inputs)
            loss = (
                (rewards - value) ** 2
                - action_probabilities * tf.math.log(probabilities)
                + self.weight_decay * tf.reduce_sum(self.model.trainable_variables**2)
            )

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer(
            learning_rate=self.learning_rate,
            epsilon=self.adam_epsilon,
            clipnorm=self.clipnorm,
        ).apply_gradients(grads_and_vars=zip(gradients, self.model.trainable_variables))

        return loss

    def action_mask(self, q, state, turn):
        q_copy = copy.deepcopy(q)
        for i in range(len(q_copy)):
            if not state.is_valid_move(i):
                if turn % 2 == 0:
                    q_copy[i] = float("-inf")
                else:
                    q_copy[i] = float("inf")
        return q_copy

    def train(self):
        self.is_test = False
        stat_score = (
            []
        )  # make these num trials divided by graph interval so i dont need to append (to make it faster?)
        stat_test_score = []
        stat_loss = []
        num_trials_truncated = 0
        state, _ = self.env.reset()
        epoch = 0
        step = 0
        temp_env = gym.make("environments/TicTacToe")
        while epoch < self.num_epochs:
            num_episodes = 0
            total_score = 0
            score = 0
            step += 1
            # play a game and learn from it
            # MONTE CARLO MONTE CARLO MONTE CARLO (PICK ACTION WITH MONTE CARLO) LOOK FOR 800 MOVES
            # action = self.select_action(state)
            temp_env = copy.deepcopy(self.env)
            info = temp_env._get_info()
            action_probabilities = self.MCTS(state, info["possible_actions"], 800)
            # MONTE CARLO MONTE CARLO MONTE CARLO (PICK ACTION WITH MONTE CARLO)
            self.transition += action_probabilities  # MONTE CARLO PROBABILITY MONTE CARLO PROBABILITY MONTE CARLO PROBABILITY
            next_state, reward, terminated, truncated = self.step(
                np.argmax(action_probabilities)
            )
            done = terminated or truncated
            state = next_state
            score += reward

            if done:
                num_episodes += 1
                state, _ = self.env.reset()
                self.store_game()
                total_score += score
                score = 0
                if self.memory.size >= self.replay_batch_size:
                    epoch += 1
                    loss = self.experience_replay()
                    stat_loss.append(loss)
                    stat_score.append(total_score / num_episodes)
                    # stat_test_score.append(self.test())
                    self.plot_graph(
                        stat_score,
                        stat_loss,
                        stat_test_score,
                        step,
                    )
                    self.export()

        self.plot_graph(
            stat_score,
            stat_loss,
            stat_test_score,
            self.num_epochs * self.steps_per_epoch,
        )
        self.export()
        self.env.close()
        return num_trials_truncated / self.num_epochs

    def plot_graph(self, score, loss, test_score, step):
        fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(30, 5))
        ax1.plot(score, linestyle="solid")
        ax1.set_title("Frame {}. Score: {}".format(step, np.mean(score[-10:])))
        ax2.plot(loss, linestyle="solid")
        ax2.set_title("Frame {}. Loss: {}".format(step, np.mean(loss[-10:])))
        ax3.plot(test_score, linestyle="solid")
        ax3.axhline(y=self.env.spec.reward_threshold, color="r", linestyle="-")
        ax3.set_title(
            "Frame {}. Test Score: {}".format(step, np.mean(test_score[-10:]))
        )
        plt.savefig("./{}.png".format(self.model_name))
        plt.close(fig)

    def test(self, video_folder="", num_trials=100) -> None:
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
