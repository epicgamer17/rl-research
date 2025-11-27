from copy import deepcopy
from math import log, sqrt, inf
import copy
import math
import numpy as np
import torch
import torch.nn.functional as F


class ChanceNode:
    """
    Represents the Afterstate (s, a).
    According to the paper: Expanded by querying stochastic model.
    Model returns:
      1. Value (afterstate value)
      2. Prior distribution over codes P(c|as)
    """

    estimation_method = None
    discount = None
    value_prefix = None

    def __init__(self, prior_policy, parent):
        assert isinstance(parent, DecisionNode)
        self.parent = parent  # DecisionNode
        self.prior_policy = prior_policy  # P(a|s) from Policy

        self.visits = 0
        self.value_sum = 0.0

        # NEW: The Value of the afterstate predicted by the Dynamics Network
        self.network_value = None

        # NEW: The distribution P(c|as) predicted by the Dynamics Network
        self.code_priors = {}

        # Children are DecisionNodes, indexed by code
        self.children = {}

    def expand(
        self,
        to_play,
        afterstate,
        network_value,
        code_priors,
        reward_h_state=None,
        reward_c_state=None,
    ):
        """
        Called when the Dynamics Network is run on (parent_state, action).
        """
        self.to_play = to_play
        self.afterstate = afterstate
        self.reward_h_state = reward_h_state
        self.reward_c_state = reward_c_state

        self.network_value = network_value
        # code_priors should be a dict or array mapping code_index -> probability
        num_chance = len(code_priors)
        self.code_priors = {
            # Warning non differentiable
            F.one_hot(torch.tensor(a), num_classes=num_chance): code_priors[a]
            for a in range(num_chance)
        }

        for code, p in self.code_priors.items():
            self.children[code] = DecisionNode(
                p.item(),  # TODO: IS THIS RIGHT?
                self,
            )

    def expanded(self):
        # We are expanded if we have populated our priors/value
        assert (
            (len(self.children) > 0) == (self.visits > 0) == (len(self.code_priors) > 0)
        )

        return len(self.code_priors) > 0

    def value(self):
        """
        Returns Q(s,a). initial value set to zt
        If unvisited, use the Network's predicted Afterstate Value
        (Bootstrap from the dynamics model, as per the text).
        """
        if self.visits == 0:
            assert self.network_value is not None
            if self.estimation_method == "v_mix":
                value = self.parent.get_v_mix()
            elif self.estimation_method == "mcts_value":
                value = self.parent.value()
            elif self.estimation_method == "network_value":
                value = self.parent.network_value
            else:
                value = 0.0
            return value
        return self.value_sum / self.visits

    def select_child(self):
        """
        Text says: "a code is selected by sampling the prior distribution".

        In practice, MCTS often balances Prior vs Visits for codes too (UCT on codes),
        but strictly following the 'sampling' description:
        """
        # Get all potential codes and their probs from the network output
        codes = list(self.code_priors.keys())
        probs = list(self.code_priors.values())

        # Normalize probs just in case
        probs = np.array(probs)
        probs /= probs.sum()

        # Sample a code
        idx = np.random.choice(len(codes), p=probs)
        selected_code = codes[idx]

        # Check if we have a node for this code yet
        child_node = self.children.get(selected_code)
        # print("selected code", selected_code)
        return selected_code, child_node

    def child_reward(self, child):
        assert isinstance(child, DecisionNode)
        if self.value_prefix:
            if child.is_reset:
                return child.reward
            else:
                return child.reward - self.parent.reward
        else:
            true_reward = child.reward

        assert true_reward is not None
        return true_reward

    def get_child_q_from_parent(self, child):
        assert isinstance(child, DecisionNode)
        r = float(self.child_reward(child))
        # child.value() if visited else v_mix
        v = float(child.value())
        # sign = +1 if child.to_play == self.to_play else -1 (multi-agent).
        sign = 1.0 if child.to_play == self.to_play else -1.0
        q_from_parent = r + self.discount * (sign * v)

        assert q_from_parent is not None
        return q_from_parent


class DecisionNode:
    estimation_method = None
    discount = None
    value_prefix = None
    pb_c_init = None
    pb_c_base = None
    gumbel = None
    cvisit = None
    cscale = None
    stochastic = None

    def __init__(self, prior_policy, parent=None):
        assert (
            (self.stochastic and isinstance(parent, ChanceNode))
            or (parent is None)
            or (not self.stochastic and isinstance(parent, DecisionNode))
        )
        self.visits = 0
        self.to_play = -1
        self.prior_policy = prior_policy
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward_h_state = None
        self.reward_c_state = None

        self.reward = 0
        self.parent = parent

        self.root_score = None
        self.network_policy = None  # dense policy vector (numpy or torch)
        self.network_value = None  # network scalar value estimate (float)

    def expand(
        self,
        allowed_actions,
        to_play,
        policy,
        hidden_state,
        reward,
        value=None,
        reward_h_state=None,
        reward_c_state=None,
        is_reset=True,
    ):
        # print(allowed_actions)
        # print(to_play)
        # print(policy)
        # print(hidden_state)
        # print(reward)
        # print(value)
        self.to_play = to_play
        self.reward = reward
        self.hidden_state = hidden_state
        self.reward_h_state = reward_h_state
        self.reward_c_state = reward_c_state
        self.is_reset = is_reset

        self.network_policy = policy.detach().cpu()

        self.network_value = value
        allowed_policy = {a: policy[a] for a in allowed_actions}
        allowed_policy_sum = sum(allowed_policy.values())

        for action, p in allowed_policy.items():
            if self.stochastic:
                self.children[action] = ChanceNode(
                    (p / (allowed_policy_sum + 1e-10)).item(),
                    self,
                )
            else:
                self.children[action] = DecisionNode(
                    (p / (allowed_policy_sum + 1e-10)).item(),
                    self,
                )

    def expanded(self):
        assert (len(self.children) > 0) == (self.visits > 0)
        return len(self.children) > 0

    def value(self):
        if self.visits == 0:
            if self.estimation_method == "v_mix":
                value = self.parent.get_v_mix()
            elif self.estimation_method == "mcts_value":
                value = self.parent.value()
            elif self.estimation_method == "network_value":
                value = self.parent.network_value
            else:
                value = 0.0
        else:
            value = self.value_sum / self.visits
        assert value is not None
        return value

    def child_reward(self, child):
        assert isinstance(child, DecisionNode)
        if self.value_prefix:
            if child.is_reset:
                return child.reward
            else:
                return child.reward - self.reward
        else:
            true_reward = child.reward

        assert true_reward is not None
        return true_reward

    def add_noise(self, dirichlet_alpha, exploration_fraction):
        actions = list(self.children.keys())
        noise = np.random.dirichlet([dirichlet_alpha] * len(actions))
        frac = exploration_fraction
        for a, n in zip(actions, noise):
            self.children[a].prior_policy = (1 - frac) * self.children[
                a
            ].prior_policy + frac * n

    def select_child(
        self,
        min_max_stats,
        allowed_actions=None,
    ):
        assert self.expanded(), "node must be expanded to select a child"
        actions = list(self.children.keys())
        if allowed_actions is not None and self.parent is None:
            actions = [a for a in allowed_actions]
        # print("actions", actions)
        if len(actions) == 1:
            # print("1 action only - defaulting")
            return actions[0], self.children[actions[0]]

        if self.gumbel and self.parent != None:
            # print("selecting action on non root gumbel node")
            pi0 = self.get_gumbel_improved_policy(
                min_max_stats=min_max_stats,
            )
            # compute selection metric: pi0(a) - N(a) / (1 + sum_N)
            visits = torch.tensor([float(self.children[a].visits) for a in actions])
            sum_N = float(visits.sum())
            denom = 1.0 + sum_N
            selection_scores = pi0[actions] - (visits / denom)
            # pick argmax (random tie-break among maxima)
            max_score = float(selection_scores.max())
            candidate_indices = np.where(np.isclose(selection_scores, max_score))[0]
            chosen_idx = np.random.choice(candidate_indices)
            action = actions[int(chosen_idx)]
        else:
            assert (
                self.gumbel == False
            ), "gumbel should not be used when using uct search"
            child_ucbs = [
                self.child_uct_score(
                    self.children[action],
                    min_max_stats,
                )
                for action in actions
            ]
            action_index = np.random.choice(
                np.where(np.isclose(child_ucbs, max(child_ucbs)))[0]
            )
            action = list(actions)[action_index]
        return action, self.children[action]

    def child_uct_score(
        self,
        child,
        min_max_stats,
    ):

        pb_c = log((self.visits + self.pb_c_base + 1) / self.pb_c_base) + self.pb_c_init
        pb_c *= sqrt(self.visits) / (child.visits + 1)

        prior_score = pb_c * child.prior_policy
        if child.expanded():
            value_score = min_max_stats.normalize(self.get_child_q_from_parent(child))
        else:
            value_score = min_max_stats.normalize(self.value())

        # check if value_score is nan
        assert value_score == value_score, "value_score is nan"
        assert prior_score == prior_score, "prior_score is nan"
        return prior_score + value_score

    def get_v_mix(self):
        # # grab probabilities for candidate actions
        visits = torch.tensor(
            [float(child.visits) for (action, child) in self.children.items()]
        )
        visited_actions = [
            action for action, child in self.children.items() if child.expanded()
        ]
        sum_N = float(visits.sum())

        # # candidate policy mass
        q_vals = torch.zeros(len(self.network_policy))
        # # visited action list indices
        if sum_N > 0:
            # q(a) for visited actions (use child.value() empirical)
            for action, child in self.children.items():
                if child.expanded():  # visits ? 0
                    q_vals[action] = self.get_child_q_from_parent(child)

            p_vis_sum = float(
                self.network_policy[visited_actions].sum()
            )  # pi mass on visited actions
            expected_q_vis = float(
                (self.network_policy * q_vals).sum()
            )  # sum_pi(a) * q(a) but q(a)=0 for unvisited
            term = sum_N * (expected_q_vis / p_vis_sum)
        else:
            term = 0.0
        v_mix = (self.value() + term) / (1.0 + sum_N)
        assert v_mix is not None
        return v_mix

    def get_completed_q(self, min_max_stats):
        v_mix = self.get_v_mix()
        # completedQ: visited keep q(a), unvisited get v_mix
        completedQ = torch.full(
            (len(self.network_policy),), min_max_stats.normalize(v_mix)
        )
        for action, child in self.children.items():
            if child.expanded():
                completedQ[action] = min_max_stats.normalize(
                    self.get_child_q_from_parent(child)
                )

        return completedQ

    def get_gumbel_improved_policy(
        self,
        min_max_stats,
    ):
        completedQ = self.get_completed_q(min_max_stats)

        # compute sigma: (cvisit + max_visits) * cscale * normalized_parent
        max_visits = (
            max([ch.visits for ch in self.children.values()])
            if len(self.children) > 0
            else 0
        )
        sigma = (self.cvisit + max_visits) * self.cscale * completedQ

        # combine logits (network) and sigma to get pi0 (only inside candidate_actions)
        eps = 1e-12
        logits = torch.log(self.network_policy + eps)

        pi0_logits = logits + sigma
        # softmax
        pi0 = torch.softmax(pi0_logits, dim=0)
        return pi0

    def get_gumbel_root_child_score(self, child, min_max_stats):
        if child.expanded():
            normalized_q = min_max_stats.normalize(self.get_child_q_from_parent(child))
        else:
            normalized_q = 0
        max_visits = (
            max([ch.visits for ch in self.children.values()])
            if len(self.children) > 0
            else 0
        )
        sigma = (self.cvisit + max_visits) * self.cscale * normalized_q
        return float(child.root_score + sigma)

    def get_child_q_from_parent(self, child):
        if isinstance(child, DecisionNode):
            r = float(self.child_reward(child))
            # child.value() if visited else v_mix
            v = float(child.value())
            # sign = +1 if child.to_play == self.to_play else -1 (multi-agent).
            sign = 1.0 if child.to_play == self.to_play else -1.0
            q_from_parent = r + self.discount * (sign * v)
        elif isinstance(child, ChanceNode):
            # TODO: should this have a discount??
            assert (
                child.to_play == self.to_play
            ), "chance nodes should be the same player as their parent"
            q_from_parent = child.value()

        return q_from_parent
