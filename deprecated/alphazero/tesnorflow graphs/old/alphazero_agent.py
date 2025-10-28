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

from replay_buffers.deprecated.alphazero_replay_buffer import ReplayBuffer
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import gymnasium as gym

import alphazero.old.MCTS_alphazero as MCTS
import random
import gc


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
        self.batch_size = int(config["batch_size"])
        self.memory_size = int(config["memory_size"])  # times number of agents
        self.max_game_length = int(config["max_game_length"])

        self.dirichlet_alpha = config["dirichlet_alpha"]
        self.dirichlet_epsilon = config["dirichlet_epsilon"]
        self.exploration_moves = config["exploration_moves"]

        self.exploration_tau = config["exploration_tau"]
        self.exploitation_tau = config["exploitation_tau"]

        self.memory = ReplayBuffer(
            observation_dimensions=self.observation_dimensions,
            max_size=self.memory_size,
            batch_size=self.batch_size,
            max_game_length=self.max_game_length,
            num_actions=self.num_actions,
            two_player=config["two_player"],
        )

        self.c_puct = config["c_puct"]
        self.weight_decay = config["weight_decay"]
        self.monte_carlo_simulations = config["monte_carlo_simulations"]
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
        state = np.array(state)
        if (self.env.observation_space.high == 255).all():
            state = state / 255
        # print(state.shape)
        if state.shape == self.observation_dimensions:
            new_shape = (1,) + state.shape
            state_input = state.reshape(new_shape)
        else:
            state_input = state
        return state_input

    def predict_single(self, state):
        state_input = self.prepare_states(state)
        # print(state_input.shape)
        value, probabilities = self.model(inputs=state_input)
        print("Probabilities ", probabilities.numpy()[0], "Value", value.numpy()[0])
        return probabilities.numpy()[0].reshape(self.num_actions), value.numpy()[0]

    def step(self, action):
        if not self.is_test:
            next_state, reward, terminated, truncated, info = self.env.step(action)
            self.transition += [reward]
            # print(self.transition)
            self.memory.store(*self.transition)
        else:
            next_state, reward, terminated, truncated, info = self.test_env.step(action)

        return next_state, reward, terminated, truncated, info

    def experience_replay(self):
        samples = self.memory.sample()
        observations = samples["observations"]
        action_probabilities = samples["action_probabilities"]
        rewards = samples["rewards"]
        print("Rewards", rewards)
        inputs = self.prepare_states(observations)
        with tf.GradientTape() as tape:
            value, probabilities = self.model(inputs)
            loss = (
                (rewards - value) ** 2
                - action_probabilities * tf.math.log(probabilities)
                # + self.weight_decay
                # * tf.reduce_sum(tf.math.square(self.model.trainable_variables))
                + sum(self.model.losses)
            )
            loss = tf.reduce_mean(loss)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer(
            learning_rate=self.learning_rate,
            epsilon=self.adam_epsilon,
            clipnorm=self.clipnorm,
        ).apply_gradients(grads_and_vars=zip(gradients, self.model.trainable_variables))
        print("Loss", loss)
        return loss

    def monte_carlo_search(
        self, env, observation, legal_moves, num_simulations, game_turn=None
    ):
        root = MCTS(env, observation, False, None, None, legal_moves)
        for i in range(num_simulations):
            print("MCTS Simulation", i)
            self.explore(root, game_turn)
        mcts_policy = np.zeros((9), dtype=np.float32)
        if game_turn != None and game_turn < self.exploration_moves:
            tau = self.exploration_tau
        else:
            tau = self.exploitation_tau
        for node in root.children:
            mcts_policy[node.parent_action] = (node.visits / (root.visits - 1)) ** (
                1 / tau
            )
        # for action, node in root.children.items():
        # mcts_policy[action] = node.visits / (root.visits - 1)
        # print(node.visits, root.visits - 1)

        for child in root.children:
            child.parent = None
            child = None
        root.children = None
        root = None
        gc.collect()
        return mcts_policy

    def explore(self, root, game_turn):
        current_node = root
        while current_node.children.any():
            puct_scores = [c.return_score() for c in current_node.children]
            max_puct_index = np.argmax(puct_scores)
            # best_actions = [
            #     action
            #     for action, child in children.items()
            #     if child.return_score() == max_puct
            # ]
            # assert (
            #     len(best_actions) > 0
            # ), "MCTS Best Actions: {}, Legal Moves: {}".format(
            #     best_actions, current_node.legal_moves
            # )
            # action_selected = random.choice(best_actions)
            # current_node = children[action_selected]
            current_node = current_node.children[max_puct_index]
        if not current_node.done:
            probabilities, value = self.predict_single(current_node.observation)
            probabilities = (
                1 - self.dirichlet_epsilon
            ) * probabilities + self.dirichlet_epsilon * np.random.dirichlet(
                [self.dirichlet_alpha] * self.num_actions
            )
            current_node.create_children()
            print("Legal Moves", current_node.legal_moves)
            print("Children", current_node.children)
            for child in current_node.children:
                print("Child", child)
                print("Actoin to reach child", child.parent_action)
                print("Child Legal Moves", child.legal_moves)
                print("Child Done", child.done)
                if child.done:
                    value = child.reward
                puct_score = value + self.c_puct * probabilities[
                    current_node.parent_action
                ] * np.sqrt(child.parent.visits) / (child.visits + 1)
                # print("MCTS PUCT", puct_score)
                child.set_score(puct_score)

        current_node.visits += 1
        parent = current_node

        while parent.parent:
            parent = parent.parent
            parent.visits += 1
            # parent.score += puct_score

    def train(self):
        super().train()
        self.is_test = False
        stat_score = (
            []
        )  # make these num trials divided by graph interval so i dont need to append (to make it faster?)
        stat_test_score = []
        stat_loss = []
        num_trials_truncated = 0
        state, info = self.env.reset()
        epoch = 0
        step = 0
        game_turn = 0
        while epoch < self.num_epochs:
            self.env.render()
            num_episodes = 0
            total_score = 0
            score = 0
            step += 1
            game_turn += 1
            print("Step", step)
            legal_moves = (
                info["legal_moves"]
                if "legal_moves" in info
                else range(self.num_actions)
            )
            action_probabilities = self.monte_carlo_search(
                self.env, state, legal_moves, self.monte_carlo_simulations, game_turn
            )
            self.transition = [state, action_probabilities]
            next_state, reward, terminated, truncated, info = self.step(
                np.random.choice(np.arange(self.num_actions), p=action_probabilities)
            )
            done = terminated or truncated  # or max(action_probabilities) < 0.05
            state = next_state
            score += reward

            if done:
                num_episodes += 1
                state, info = self.env.reset()
                game_turn = 0
                self.memory.store_game()
                # if score >= self.env.spec.reward_threshold:
                #     print("Your agent has achieved the env's reward threshold.")
                total_score += score
                score = 0
                if self.memory.size >= self.batch_size:
                    epoch += 1
                    loss = self.experience_replay()
                    stat_loss.append(loss)
                    # self.memory.clear()
                    stat_score.append(total_score / num_episodes)
                    stat_test_score.append(self.test(num_trials=5))
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
        # ax3.axhline(y=self.env.spec.reward_threshold, color="r", linestyle="-")
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
            state, info = self.test_env.reset()
            done = False
            score = 0

            while not done:
                legal_moves = (
                    info["legal_moves"]
                    if "legal_moves" in info
                    else range(self.num_actions)
                )
                action = np.argmax(
                    self.monte_carlo_search(
                        self.env, state, legal_moves, self.monte_carlo_simulations
                    )
                )
                next_state, reward, terminated, truncated, info = self.step(action)
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
            legal_moves = (
                info["legal_moves"]
                if "legal_moves" in info
                else range(self.num_actions)
            )
            action = np.argmax(
                self.monte_carlo_search(
                    self.env, state, legal_moves, self.monte_carlo_simulations
                )
            )
            next_state, reward, terminated, truncated, info = self.step(action)
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
