from copy import deepcopy
from math import log, sqrt, inf
import copy
import math
import numpy as np


class Node:
    def __init__(self, prior_policy, parent=None):
        self.visits = 0
        self.to_play = -1
        self.prior_policy = prior_policy
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0
        self.parent = parent

    def expand(self, legal_moves, to_play, policy, hidden_state, reward):
        self.to_play = to_play
        self.reward = reward
        self.hidden_state = hidden_state
        # print(legal_moves)
        policy = {a: policy[a] for a in legal_moves}
        policy_sum = sum(policy.values())

        for action, p in policy.items():
            self.children[action] = Node((p / (policy_sum + 1e-10)).item(), self)

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visits == 0:
            return 0
        return self.value_sum / self.visits

    def add_noise(self, dirichlet_alpha, exploration_fraction):
        actions = list(self.children.keys())
        noise = np.random.dirichlet([dirichlet_alpha] * len(actions))
        frac = exploration_fraction
        for a, n in zip(actions, noise):
            self.children[a].prior_policy = (1 - frac) * self.children[
                a
            ].prior_policy + frac * n

    def select_child(self, min_max_stats, pb_c_base, pb_c_init, discount, num_players):
        # Select the child with the highest UCB
        child_ucbs = [
            self.child_ucb_score(
                child, min_max_stats, pb_c_base, pb_c_init, discount, num_players
            )
            for action, child in self.children.items()
        ]
        # print("Child UCBs", child_ucbs)
        action_index = np.random.choice(
            np.where(np.isclose(child_ucbs, max(child_ucbs)))[0]
        )
        action = list(self.children.keys())[action_index]
        return action, self.children[action]

    def child_ucb_score(
        self, child, min_max_stats, pb_c_base, pb_c_init, discount, num_players
    ):
        pb_c = log((self.visits + pb_c_base + 1) / pb_c_base) + pb_c_init
        pb_c *= sqrt(self.visits) / (child.visits + 1)

        prior_score = pb_c * child.prior_policy
        if child.visits > 0:
            if num_players == 1:
                sign = 1.0
            else:
                sign = 1.0 if child.to_play == self.to_play else -1.0

            value_score = min_max_stats.normalize(
                child.reward + discount * (sign * child.value())
            )
        else:
            value_score = 0.0

        # check if value_score is nan
        assert (
            value_score == value_score
        ), "value_score is nan, child value is {}, and reward is {},".format(
            child.value(),
            child.reward,
        )
        assert prior_score == prior_score, "prior_score is nan"
        return prior_score + value_score
