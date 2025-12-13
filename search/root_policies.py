from abc import ABC, abstractmethod
import torch
import numpy as np


class RootPolicyStrategy(ABC):
    def __init__(self, config, device, num_actions):
        self.config = config
        self.device = device
        self.num_actions = num_actions

    @abstractmethod
    def get_policy(self, root, min_max_stats):
        """
        Returns the final policy distribution (probability vector)
        over all actions based on the search results.
        Returns: torch.Tensor of shape (num_actions,)
        """
        pass


class VisitFrequencyPolicy(RootPolicyStrategy):
    """
    Standard AlphaZero Policy:
    Returns probabilities proportional to the visit counts of the children.
    Includes temperature support (visits ^ (1/T)).
    """

    def __init__(self, config, device, num_actions, temperature=1.0):
        super().__init__(config, device, num_actions)
        self.temperature = temperature

    def get_policy(self, root, min_max_stats):
        # Gather visits
        visits = torch.zeros(self.num_actions, device=self.device)
        for action, child in root.children.items():
            visits[action] = child.visits

        if self.temperature == 0:
            # Greedy Argmax (break ties randomly or first)
            # Create a one-hot distribution on the max visited action
            max_visit = torch.max(visits)
            # Mask for all actions that share the max visit count
            mask = (visits == max_visit).float()
            # Normalize to handle ties
            policy = mask / mask.sum()
            return policy

        elif self.temperature == 1.0:
            # Standard Proportional
            sum_visits = torch.sum(visits)
            if sum_visits > 0:
                return visits / sum_visits
            else:
                # Fallback if no visits (should not happen in expanded root)
                return torch.ones_like(visits) / self.num_actions

        else:
            # Temperature Adjusted
            # prevent numerical instability with 0 visits by adding epsilon or masking
            visits_powered = torch.pow(visits, 1.0 / self.temperature)
            sum_powered = torch.sum(visits_powered)
            if sum_powered > 0:
                return visits_powered / sum_powered
            return torch.ones_like(visits) / self.num_actions


class GreedyValuePolicy(RootPolicyStrategy):
    """
    Selects the action with the highest Q-value (Mean Value) at the root.
    Useful for evaluation if you want to trust value over counts.
    """

    def get_policy(self, root, min_max_stats):
        q_values = torch.full((self.num_actions,), -float("inf"), device=self.device)

        for action, child in root.children.items():
            if child.expanded():
                # Get normalized Q-value
                q_values[action] = min_max_stats.normalize(
                    root.get_child_q_from_parent(child)
                )
            else:
                # Handle unexpanded children (should be rare for root choices)
                q_values[action] = min_max_stats.normalize(root.value())

        # Argmax
        max_q = torch.max(q_values)
        mask = (q_values == max_q).float()
        return mask / mask.sum()


class GumbelRootPolicy(RootPolicyStrategy):
    """
    Gumbel MuZero Policy:
    Calculates the 'Improved Policy' (pi0) using the Completed Q-values and Sigma transformation.
    This is distinct from visit counts and is the mathematically correct policy target for Gumbel MuZero.
    """

    def get_policy(self, root, min_max_stats):
        # 1. Compute Completed Q-values
        completed_q = self._get_completed_q(root, min_max_stats)

        # 2. Compute Sigma
        sigma = self._calculate_gumbel_sigma(root, completed_q)

        # 3. Compute Improved Policy (pi0)
        # Note: We must use the raw logits from the root's stored network policy
        # If root.network_policy is probabilities, we convert to logits.
        eps = 1e-12
        # root.network_policy is usually on CPU or Device, ensure consistency
        policy_tensor = root.network_policy.to(self.device)
        logits = torch.log(policy_tensor + eps)

        pi0_logits = logits + sigma
        pi0 = torch.softmax(pi0_logits, dim=0)

        return pi0

    # --- Self-Contained Gumbel Helpers ---

    def _get_completed_q(self, node, min_max_stats):
        v_mix = node.get_v_mix()

        # Initialize with v_mix (value of unvisited actions)
        completedQ = torch.full(
            (self.num_actions,), min_max_stats.normalize(v_mix), device=self.device
        )

        for action, child in node.children.items():
            if child.expanded():
                completedQ[action] = min_max_stats.normalize(
                    node.get_child_q_from_parent(child)
                )
        return completedQ

    def _calculate_gumbel_sigma(self, node, completedQ):
        # Find max visits among children
        if len(node.children) > 0:
            max_visits = max([ch.visits for ch in node.children.values()])
        else:
            max_visits = 0

        # Sigma formula: (c_visit + max_N) * c_scale * Q(a)
        return (
            (self.config.gumbel_cvisit + max_visits)
            * self.config.gumbel_cscale
            * completedQ
        )
