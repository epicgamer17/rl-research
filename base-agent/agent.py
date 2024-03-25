import gc
import sys

sys.path.append("../")

import copy
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import gymnasium as gym


class Agent:
    def __init__(self, env, name, config):
        self.model_name = name

        self.env = env
        self.test_env = copy.deepcopy(env)
        self.num_actions = env.action_space.n
        self.observation_dimensions = env.observation_space.shape

        self.training_steps = config["training_steps"]
        self.checkpoint_interval = 1

        self.is_test = False

    def step(self, action):
        if not self.is_test:
            next_state, reward, terminated, truncated, info = self.env.step(action)
        else:
            next_state, reward, terminated, truncated, info = self.test_env.step(action)

        return next_state, reward, terminated, truncated, info

    def train(self):
        pass

    def prepare_states(self, state):
        state = np.array(state)
        if state.shape == self.observation_dimensions:
            new_shape = (1,) + state.shape
            state_input = state.reshape(new_shape)
        else:
            state_input = state
        return state_input

    def predict_single(self, state, illegal_moves=None):
        pass

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

    def plot_graph(
        self, score, policy_loss, value_loss, l2_loss, loss, test_score, step
    ):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 10))
        ax1.plot(score, linestyle="solid")
        ax1.set_title("Frame {}. Score: {}".format(step, np.mean(score[-10:])))
        ax2.plot(policy_loss, linestyle="solid")
        ax2.set_title(
            "Frame {}. Policy Loss: {}".format(step, np.mean(policy_loss[-10:]))
        )
        ax3.plot(value_loss, linestyle="solid")
        ax3.set_title(
            "Frame {}. Value Loss: {}".format(step, np.mean(value_loss[-10:]))
        )
        ax4.plot(test_score, linestyle="solid")
        # ax3.axhline(y=self.env.spec.reward_threshold, color="r", linestyle="-")
        ax4.set_title(
            "Frame {}. Test Score: {}".format(step, np.mean(test_score[-10:]))
        )
        plt.savefig("./{}.png".format(self.model_name))
        plt.close(fig)

    def test(self, num_trials=100, video_folder="") -> None:
        """Test the agent."""
        self.is_test = True
        average_score = 0

        state, info = self.test_env.reset()
        legal_moves = (
            info["legal_moves"] if "legal_moves" in info else range(self.num_actions)
        )
        for trials in range(num_trials):
            done = False
            score = 0
            test_game_moves = []
            while not done:
                visit_counts = self.monte_carlo_tree_search(
                    self.test_env, state, legal_moves
                )
                actions = [action for _, action in visit_counts]
                visit_counts = np.array(
                    [count for count, _ in visit_counts], dtype=np.float32
                )
                print("MCTS Policy ", visit_counts / np.sum(visit_counts))
                action = actions[np.argmax(visit_counts)]
                test_game_moves.append(action)
                next_state, reward, terminated, truncated, info = self.step(action)
                done = terminated or truncated
                legal_moves = (
                    info["legal_moves"]
                    if "legal_moves" in info
                    else range(self.num_actions)
                )
                state = next_state
                score += reward
            state, info = self.test_env.reset()
            legal_moves = (
                info["legal_moves"]
                if "legal_moves" in info
                else range(self.num_actions)
            )
            average_score += score
            print("score: ", score)

        if video_folder == "":
            video_folder = "./videos/{}".format(self.model_name)

        video_test_env = copy.deepcopy(self.test_env)
        video_test_env.reset()
        video_test_env = gym.wrappers.RecordVideo(video_test_env, video_folder)
        for move in test_game_moves:
            video_test_env.step(move)
        video_test_env.close()

        # reset
        self.is_test = False
        average_score /= num_trials
        return average_score
