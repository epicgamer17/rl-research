import torch
import torch.nn.functional as F
import numpy as np
from math import sqrt, log
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass

from search.nodes import ChanceNode, DecisionNode


class SelectionStrategy(ABC):
    @abstractmethod
    def select_child(self, node, allowed_actions, min_max_stats):
        pass


class UCBSelectionStrategy(SelectionStrategy):
    """
    Standard UCB Selection Strategy.
    Moves the logic that was previously in SearchAlgorithm.select_child and _child_uct_scores.
    """

    def select_child(self, node, allowed_actions, min_max_stats):
        assert isinstance(node, DecisionNode)
        assert node.expanded(), "node must be expanded to select a child"
        actions = list(node.children.keys())
        if allowed_actions is not None and node.parent is None:
            actions = [a for a in allowed_actions]

        if len(actions) == 1:
            return actions[0], node.children[actions[0]]

        """Selects action using standard UCB score."""
        child_ucts = self._child_uct_scores(node, actions, min_max_stats)
        action_index = np.random.choice(
            np.where(np.isclose(child_ucts, max(child_ucts)))[0]
        )
        action = actions[action_index]

        assert isinstance(action, int)
        return action, node.children[action]

    def _child_uct_scores(
        self,
        node,
        actions,
        min_max_stats,
    ):
        uct_scores = []
        for action in actions:
            child = node.children[action]
            pb_c = (
                log((node.visits + node.pb_c_base + 1) / node.pb_c_base)
                + node.pb_c_init
            )
            pb_c *= sqrt(node.visits) / (child.visits + 1)
            prior_score = pb_c * child.prior_policy

            if child.expanded():
                value_score = min_max_stats.normalize(
                    node.get_child_q_from_parent(child)
                )
            else:
                value_score = min_max_stats.normalize(node.value())

            # check if value_score is nan
            assert value_score == value_score, "value_score is nan"
            assert prior_score == prior_score, "prior_score is nan"
            uct_scores.append(prior_score + value_score)
        return uct_scores


class GumbelMuZeroSelectionStrategy(SelectionStrategy):
    """
    Gumbel MuZero Selection Strategy.
    Selects actions based on the improvement policy pi0 and visit counts.
    """

    def __init__(self, config):
        self.config = config

    def select_child(self, node, allowed_actions, min_max_stats):
        assert isinstance(node, DecisionNode)
        assert node.expanded(), "node must be expanded to select a child"
        actions = list(node.children.keys())
        if allowed_actions is not None:
            # If we have restricted allowed actions, filter 'actions'
            # Note: This strategy is usually for internal selection or
            # sequential halving where allowed_actions might be relevant.
            actions = [a for a in allowed_actions if a in node.children]

        if len(actions) == 1:
            return actions[0], node.children[actions[0]]

        # Use helper methods from GumbelStrategy logic attached to DecisionNode
        # OR replicate the logic here. Since get_gumbel_improved_policy relies on node state
        # and min_max_stats, we can implement it here or reuse if available.
        # Below implements the requested snippet.

        pi0 = self._get_gumbel_improved_policy(node, min_max_stats)

        # compute selection metric: pi0(a) - N(a) / (1 + sum_N)
        visits = torch.tensor([float(node.children[a].visits) for a in actions])
        sum_N = float(visits.sum())
        denom = 1.0 + sum_N
        selection_scores = pi0[actions] - (visits / denom)

        # pick argmax (random tie-break among maxima)
        max_score = float(selection_scores.max())
        candidate_indices = np.where(np.isclose(selection_scores, max_score))[0]
        chosen_idx = np.random.choice(candidate_indices)
        action = actions[int(chosen_idx)]

        assert isinstance(action, int)
        return action, node.children[action]

    # --- Reusing Gumbel Helper Logic (Self-contained) ---
    def _get_gumbel_improved_policy(self, node, min_max_stats):
        completedQ = self._get_completed_q(node, min_max_stats)
        sigma = self._calculate_gumbel_sigma(node, completedQ)
        eps = 1e-12
        logits = torch.log(node.network_policy + eps)
        pi0_logits = logits + sigma
        pi0 = torch.softmax(pi0_logits, dim=0)
        return pi0

    def _get_completed_q(self, node, min_max_stats):
        v_mix = node.get_v_mix()
        completedQ = torch.full(
            (len(node.network_policy),), min_max_stats.normalize(v_mix)
        )
        for action, child in node.children.items():
            if child.expanded():
                completedQ[action] = min_max_stats.normalize(
                    node.get_child_q_from_parent(child)
                )
        return completedQ

    def _calculate_gumbel_sigma(self, node, completedQ):
        max_visits = (
            max([ch.visits for ch in node.children.values()])
            if len(node.children) > 0
            else 0
        )
        return (
            (self.config.gumbel_cvisit + max_visits)
            * self.config.gumbel_cscale
            * completedQ
        )


class SamplingSelectionStrategy(SelectionStrategy):
    """
    Pure Sampling Strategy.
    For DecisionNodes: Samples action based on visit counts (simulating temperature) or raw priors.
    For ChanceNodes: Samples codes based on probabilities.
    """

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def select_child(self, node, allowed_actions, min_max_stats):
        if isinstance(node, DecisionNode):
            actions = list(node.children.keys())
            if allowed_actions is not None:
                actions = [a for a in allowed_actions if a in node.children]

            if not actions:
                raise ValueError("No actions available to sample from.")

            # Calculate probabilities based on visit counts
            visits = np.array([node.children[a].visits for a in actions])

            if self.temperature == 0:
                # Argmax behavior for temp=0
                action_idx = np.argmax(visits)
            else:
                # Apply temperature
                visits_temp = visits ** (1 / self.temperature)
                probs = visits_temp / np.sum(visits_temp)
                action_idx = np.random.choice(len(actions), p=probs)

            action = actions[action_idx]
            return action, node.children[action]

        elif isinstance(node, ChanceNode):
            codes = list(node.code_priors.keys())
            probs = list(node.code_priors.values())
            code = node._sample_code(codes, probs)
            selected_code = F.one_hot(torch.tensor(code), num_classes=len(codes))
            child_node = node.children[code]
            return selected_code, child_node


class LeastVisitedSelectionStrategy(SelectionStrategy):
    """
    Selects the child that has been visited the least (BFS-style exploration).
    Supports a custom heuristic for tie-breaking or sorting.
    Can effectively be used as a uniform search.
    """

    def __init__(self, heuristic: Callable[[Any], float] = None):
        # Default heuristic is just the visit count.
        # You can pass a lambda to sort by other properties (e.g. prioritizing specific actions)
        self.heuristic = heuristic if heuristic else lambda child: child.root_score

    def select_child(self, node, allowed_actions, min_max_stats):
        # if isinstance(node, DecisionNode):
        assert isinstance(node, DecisionNode)
        actions = list(node.children.keys())
        if allowed_actions is not None:
            actions = [a for a in allowed_actions if a in node.children]

        # Sort actions based on the heuristic (default: visits ascending)
        # We look for the child with the MINIMUM score returned by the heuristic
        best_action = min(actions, key=lambda a: self.heuristic(node.children[a]))

        return best_action, node.children[best_action]

    # elif isinstance(node, ChanceNode):
    #     # Fallback to sampling for chance nodes as BFS is ambiguous for stochastic transitions
    #     codes = list(node.code_priors.keys())
    #     probs = list(node.code_priors.values())
    #     code = node._sample_code(codes, probs)
    #     selected_code = F.one_hot(torch.tensor(code), num_classes=len(codes))
    #     child_node = node.children[code]
    #     return selected_code, child_node


class MaxQSelectionStrategy(SelectionStrategy):
    """
    Selects the child with the highest Q-value.
    Requires the network to output explicit Q-values (NetworkOutput.q_values)
    OR relies on the MCTS tree statistics.
    """

    def __init__(self, use_network_q: bool = True):
        self.use_network_q = use_network_q

    def select_child(self, node, allowed_actions, min_max_stats):
        assert isinstance(node, DecisionNode)
        actions = list(node.children.keys())
        if allowed_actions is not None:
            actions = [a for a in allowed_actions if a in node.children]

        if self.use_network_q:
            # Use explicit Q-values from the network output if available
            if (
                not hasattr(node, "network_output")
                or node.network_output.q_values is None
            ):
                raise ValueError(
                    "NetworkOutput does not contain q_values required for MaxQSelectionStrategy"
                )

            # Get Q-values for valid actions
            q_vals = node.network_output.q_values[actions]
            # Argmax
            best_idx = torch.argmax(q_vals).item()
            action = actions[best_idx]
        else:
            # Use Tree Q-values (mean value of child)
            # We use min_max_stats to normalize if necessary, or just raw Q
            best_action = max(
                actions,
                key=lambda a: node.get_child_q_from_parent(node.children[a]),
            )
            action = best_action

        return action, node.children[action]
