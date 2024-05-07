import math
import os
import gymnasium as gym
from agent_configs import Config
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import copy

# Every model should have:
# 1. A network
# 2. An optimizer
# 3. A loss function
# 4. A training method
#       this method should have training iterations, minibatches, and training steps
# 5. A step method
# 6. A select_action method
# 7. A predict method


class BaseAgent:
    def __init__(self, env: gym.Env, config: Config, name):
        self.model_name = name
        self.config = config

        self.env = env
        # self.test_env = copy.deepcopy(env)
        if self.env.render_mode == "rgb_array":
            self.test_env = gym.wrappers.RecordVideo(
                copy.deepcopy(env),
                "./videos/{}".format(self.model_name),
                name_prefix="{}".format(self.model_name),
            )
        else:
            self.test_env = copy.deepcopy(env)

        if isinstance(env.observation_space, gym.spaces.Box):
            self.observation_dimensions = env.observation_space.shape
        elif isinstance(env.observation_space, gym.spaces.Discrete):
            self.observation_dimensions = (env.observation_space.n,)
        else:
            raise ValueError("Observation space not supported")

        print("observation_dimensions: ", self.observation_dimensions)
        if isinstance(env.action_space, gym.spaces.Discrete):
            self.num_actions = env.action_space.n
            self.discrete_action_space = True
        else:
            self.num_actions = env.action_space.shape[0]
            self.discrete_action_space = False

        print("num_actions: ", self.num_actions)

        self.training_steps = self.config.training_steps
        self.checkpoint_interval = 10

        self.is_test = False

    def step(self, action):
        if not self.is_test:
            next_state, reward, terminated, truncated, info = self.env.step(action)
        else:
            next_state, reward, terminated, truncated, info = self.test_env.step(action)

        return next_state, reward, terminated, truncated, info

    def train(self):
        raise NotImplementedError

    def prepare_states(self, state):
        state_copy = np.array(state)
        if self.config.game.is_image:
            state_copy = state_copy / 255.0
        if state_copy.shape == self.observation_dimensions:
            new_shape = (1,) + state_copy.shape
            state_input = state_copy.reshape(new_shape)
        else:
            state_input = state_copy
        return state_input

    def predict_single(self, state):
        raise NotImplementedError

    def select_action(self, state, legal_moves=None):
        raise NotImplementedError

    def action_mask(self, actions, legal_moves, mask_value=0):
        if self.config.game.has_legal_moves and self.config.game.is_discrete:
            print("masking actions")
            mask = np.zeros(self.num_actions, dtype=np.int8)
            mask[legal_moves] = 1
            print("mask", mask)
            print("legal_moves", legal_moves)
            actions[mask == 0] = mask_value
            print(actions)
        return actions

    def calculate_loss(self, batch):
        pass

    def learn(self):
        # experience replay
        # raise NotImplementedError
        pass

    def collect_experience(self):
        # raise NotImplementedError
        pass

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

        self.model.save(path)
        # save replay buffer
        # save optimizer

        # test model
        test_score = self.test(num_trials, training_step)
        stats["test_score"].append(test_score)
        # plot the graphs
        self.plot_graph(stats, targets, training_step, frames_seen, time_taken)

    def plot_graph(self, stats, targets, step, frames_seen, time_taken):
        num_plots = len(stats)
        sqrt_num_plots = math.ceil(np.sqrt(num_plots))
        fig, axs = plt.subplots(
            sqrt_num_plots,
            sqrt_num_plots,
            figsize=(10 * sqrt_num_plots, 5 * sqrt_num_plots),
            squeeze=False,
        )

        hours = int(time_taken // 3600)
        minutes = int((time_taken % 3600) // 60)
        seconds = int(time_taken % 60)

        fig.suptitle(
            "training stats | training step {} | frames seen {} | time taken {} hours {} minutes {} seconds".format(
                step, frames_seen, hours, minutes, seconds
            )
        )

        for i, (key, value) in enumerate(stats.items()):
            x = np.arange(0, len(value))
            row = i // sqrt_num_plots
            col = i % sqrt_num_plots
            axs[row, col].plot(x, value)
            axs[row, col].set_title(
                "{} | rolling average: {}".format(key, np.mean(value[-10:]))
            )
            if key in targets and targets[key] is not None:
                axs[row, col].axhline(y=targets[key], color="r", linestyle="--")

        for i in range(num_plots, sqrt_num_plots**2):
            row = i // sqrt_num_plots
            col = i % sqrt_num_plots
            fig.delaxes(axs[row, col])

        # plt.show()
        if not os.path.exists("./training_graphs"):
            os.makedirs("./training_graphs")
        if not os.path.exists("./training_graphs/{}".format(self.model_name)):
            os.makedirs("./training_graphs/{}".format(self.model_name))
        plt.savefig(
            "./training_graphs/{}/{}.png".format(self.model_name, self.model_name)
        )

        plt.close(fig)

    def test(self, num_trials, step) -> None:
        """Test the agent."""
        self.is_test = True
        average_score = 0
        if self.test_env.render_mode == "rgb_array":
            self.test_env.episode_trigger = lambda x: (x + 1) % num_trials == 0
            self.test_env.video_folder = "./videos/{}/{}".format(self.model_name, step)
            if not os.path.exists(self.test_env.video_folder):
                os.makedirs(self.test_env.video_folder)
        for trials in range(num_trials):
            state, info = self.test_env.reset()
            # self.test_env.render()
            legal_moves = (
                info["legal_moves"] if self.config.game.has_legal_moves else None
            )

            done = False
            score = 0

            while not done:
                action = self.select_action(state, legal_moves)
                next_state, reward, terminated, truncated, info = self.step(action)
                # self.test_env.render()
                done = terminated or truncated
                legal_moves = (
                    info["legal_moves"] if self.config.game.has_legal_moves else None
                )

                state = next_state
                score += reward
            average_score += score
            print("score: ", score)

        # reset
        if self.test_env.render_mode != "rgb_array":
            self.test_env.render()
        self.test_env.close()
        self.is_test = False
        average_score /= num_trials
        return average_score
