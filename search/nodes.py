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
        # assert isinstance(parent, DecisionNode)
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
            a: code_priors[a]
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
            return self._get_bootstrap_value()
        return self.value_sum / self.visits

    def _get_bootstrap_value(self):
        """Helper to determine value when visits are 0 based on estimation method."""
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

    # def select_child(self):
    #     """
    #     Text says: "a code is selected by sampling the prior distribution".
    #     """
    #     # Get all potential codes and their probs from the network output
    #     codes = list(self.code_priors.keys())
    #     probs = list(self.code_priors.values())

    #     code = self._sample_code(codes, probs)
    #     selected_code = F.one_hot(torch.tensor(code), num_classes=len(codes))

    #     # Check if we have a node for this code yet
    #     child_node = self.children[code]
    #     return selected_code, child_node

    def _sample_code(self, codes, probs):
        """Helper to sample a single code index from probabilities."""
        # Normalize probs just in case
        probs = np.array(probs)
        probs /= probs.sum()
        return np.random.choice(len(codes), p=probs)

    def child_reward(self, child):
        # assert isinstance(child, DecisionNode)
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
        # assert isinstance(child, DecisionNode)
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
        self.to_play = to_play
        self.reward = reward
        self.hidden_state = hidden_state
        self.reward_h_state = reward_h_state
        self.reward_c_state = reward_c_state
        self.is_reset = is_reset

        self.network_policy = policy.detach().cpu()
        self.network_value = value

        self._populate_children(allowed_actions, policy)

    def _populate_children(self, allowed_actions, policy):
        """Helper to create child nodes based on policy and allowed actions."""
        allowed_policy = {a: policy[a] for a in allowed_actions}
        allowed_policy_sum = sum(allowed_policy.values())

        NodeType = ChanceNode if self.stochastic else DecisionNode

        for action, p in allowed_policy.items():
            normalized_p = (p / (allowed_policy_sum + 1e-10)).item()
            self.children[action] = NodeType(normalized_p, self)

    def expanded(self):
        assert (len(self.children) > 0) == (self.visits > 0)
        return len(self.children) > 0

    def value(self):
        if self.visits == 0:
            return self._get_bootstrap_value()
        else:
            value = self.value_sum / self.visits
        assert value is not None
        return value

    def _get_bootstrap_value(self):
        """Helper to determine value when visits are 0."""
        if self.estimation_method == "v_mix":
            value = self.parent.get_v_mix()
        elif self.estimation_method == "mcts_value":
            value = self.parent.value()
        elif self.estimation_method == "network_value":
            value = self.parent.network_value
        else:
            value = 0.0
        return value

    def child_reward(self, child):
        # assert isinstance(child, DecisionNode)
        if self.value_prefix:
            if child.is_reset:
                return child.reward
            else:
                return child.reward - self.reward
        else:
            true_reward = child.reward

        assert true_reward is not None
        return true_reward

    def get_v_mix(self):
        # # grab probabilities for candidate actions
        visits = torch.tensor(
            [float(child.visits) for (action, child) in self.children.items()]
        )
        visited_actions = [
            action for action, child in self.children.items() if child.expanded()
        ]
        sum_N = float(visits.sum())

        if sum_N > 0:
            term = self._calculate_visited_policy_mass(visited_actions, sum_N)
        else:
            term = 0.0

        v_mix = (self.value() + term) / (1.0 + sum_N)
        assert v_mix is not None
        return v_mix

    def _calculate_visited_policy_mass(self, visited_actions, sum_N):
        """Calculates the weighted value term for v_mix based on visited actions."""
        q_vals = torch.zeros(len(self.network_policy))
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

        return sum_N * (expected_q_vis / p_vis_sum)

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
