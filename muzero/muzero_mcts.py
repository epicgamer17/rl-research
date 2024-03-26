from copy import deepcopy
from math import log, sqrt, inf
import copy
import numpy as np


class Node:
    def __init__(self, prior_policy):
        self.visits = 0
        self.to_play = -1
        self.prior_policy = prior_policy
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0

    def expand(self, legal_moves, to_play, policy, hidden_state, reward):
        self.to_play = to_play
        self.reward = reward
        self.hidden_state = hidden_state

        policy = {a: policy[a] for a in legal_moves}
        policy_sum = sum(policy.values())

        for action, p in policy.items():
            self.children[action] = Node(p / policy_sum)

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

    def select_child(self, min_max_stats, pb_c_base, pb_c_init):
        # Select the child with the highest UCB
        _, action, child = max(
            [
                (
                    self.child_ucb_score(child, min_max_stats, pb_c_base, pb_c_init),
                    action,
                    child,
                )
                for action, child in self.children.items()
            ]
        )
        return action, child

    def child_ucb_score(self, child, min_max_stats, pb_c_base, pb_c_init):
        pb_c = log((self.visits + pb_c_base + 1) / pb_c_base) + pb_c_init
        pb_c *= sqrt(self.visits) / (child.visits + 1)

        prior_score = pb_c * child.prior_policy * sqrt(self.visits) / (child.visits + 1)
        value_score = min_max_stats.normalize(child.value())
        return prior_score + value_score
