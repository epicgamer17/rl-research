import time
import gymnasium as gym
import environments.environments_.envs as envs
import tensorflow as tf
import numpy as np

env = gym.make("environments/TicTacToe")
config = {
    "clip_param": 0.2,
    "activation": "relu",
    "kernel_initializer": "orthogonal",
    "actor_optimizer": tf.keras.optimizers.legacy.Adam,
    "critic_optimizer": tf.keras.optimizers.legacy.Adam,
    "actor_learning_rate": 0.005,
    "critic_learning_rate": 0.005,
    "actor_epsilon": 1e-7,
    "critic_epsilon": 1e-7,
    "actor_clipnorm": 0.5,
    "critic_clipnorm": 0.5,
    # NORMALIZATION?
    # 'n_step': 3,
    "discount_factor": 0.99,
    "gae_lambda": 0.98,
    "conv_layers": [],
    "conv_layers_noisy": False,
    "critic_width": 64,
    "critic_dense_layers": 2,
    "critic_dense_layers_noisy": False,
    "actor_width": 64,
    "actor_dense_layers": 2,
    "actor_dense_layers_noisy": False,
    # REWARD CLIPPING
    "noisy_sigma": 0.5,  #
    "loss_function": tf.keras.losses.KLDivergence(),
    "num_epochs": 30,
    "steps_per_epoch": 4800,
    "train_policy_iterations": 5,
    "train_value_iterations": 5,
    # 'per_epsilon': 0.001,
    # 'per_alpha': 0.5,
    # 'per_beta': 0.5,
    "target_kl": 0.02,
    # 'per_beta_increase': hp.uniform('per_beta_increase', 0, 0.015),
    # 'v_min': -500.0, # MIN GAME SCORE
    # 'v_max': 500.0, # MAX GAME SCORE
    # 'search_max_depth': 5,
    # 'search_max_time': 10,
}
GAME_ACTIONS = env.action_space.n
GAME_OBS = env.observation_space.shape[0]
from copy import deepcopy
from math import log, sqrt, inf
import random


class Node:
    def __init__(self, observation, done, parent, action_to_get_here, actions_possible):
        self.observation = observation
        self.done = done
        self.parent = parent
        self.action__to_get_here = action_to_get_here
        self.children = {}
        self.total_rollouts = 0
        self.visits = 0
        self.action_possible = actions_possible

    def get_UCB_score(self):
        if self.visits == 0:
            return inf
        return self.total_rollouts / self.visits + sqrt(
            log(self.parent.visits) / self.visits
        )

    def create_children(self):
        if self.done:
            return None
        dico = self.game.get_info()
        self.actions_to_try = dico["possible_actions"]
        for action in range(self.actions_to_try):
            new_game = deepcopy(self.game)
            observation, reward, terminated, truncated, info = new_game.step(action)
            done = terminated or truncated
            new_possible_actions = self.action_to_try
            new_possible_actions.remove(action)
            self.children[action] = Node(
                observation, done, self, action, new_possible_actions
            )

    def explore(self):
        current = self
        while current.children:
            child = current.children
            max_U = max(c.get_UCB_score() for c in child.values())
            actions = [a for a, c in child.items() if c.get_UCB_score() == max_U]
            action_selected = random.choice(actions, self.game)
            current = child[action_selected]

        if current.visits == 0:
            current.total_rollouts += current.rollout()
        else:
            current.create_children()
            if current.children:
                current = random.choice(current.children)
            current.total_rollouts += current.rollout()

        current.visits += 1
        dad = current
        rollouts_rn = current.total_rollouts

        while dad.parent:
            dad = dad.parent
            dad.visits += 1
            dad.total_rollouts += rollouts_rn

    # useless for alphago zero
    def rollout(self):
        if self.done:
            return 0
        v = 0
        done = False
        new_game = deepcopy(self.game)
        while not done:
            action = new_game.action_space.sample()
            observation, reward, terminated, truncated, info = new_game.step(action)
            done = terminated or truncated
            v += reward
            if done:
                new_game.reset()
                new_game.close()
                return v

    # not useful for alphazaero
    def choose_next(self):
        if self.done:
            raise ValueError("game has ended")
        if not self.children:
            raise ValueError("no children found and game has not ended")
        max_N = max(node.N for node in self.children.values())
        max_children = [c for a, c in self.children.items() if c.N == max_N]
        max_child = random.choice(max_children)
        return max_child, max_child.action_to_get_here


def monte_carlo_search(observation, actions_possible, num_of_iterations):
    root = Node(observation, False, None, None, actions_possible)
    for i in range(num_of_iterations):
        root.explore()
    tot = 0
    for action, node in root.children.items():
        tot += node.get_USB_score()
    prob_array = np.zeros((1, 9))
    for action, node in root.children.items():
        prob_array[0][action] = node.visits / tot
    return prob_array
