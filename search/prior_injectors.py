from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any

import numpy as np
import torch

from modules.utils import support_to_scalar
from search.initial_action_sets import ActionSet, SelectAll, SelectTopK
from utils.utils import get_legal_moves


class PriorInjector(ABC):
    @abstractmethod
    def inject(self, policy, config, trajectory_action=None):
        """Modifies the context (policy, scores, etc.) in place."""
        pass


class DirichletInjector(PriorInjector):
    def inject(self, policy, config, trajectory_action=None):
        # Only apply noise to legal moves
        noise = np.random.dirichlet([config.root_dirichlet_alpha] * len(policy))
        frac = config.root_exploration_fraction

        # Map noise back to the full policy tensor (or just relevant indices)
        # Note: We operate on the policy probabilities
        new_policy = policy.clone()
        for i, action in enumerate(policy):
            new_policy[action] = (1 - frac) * policy[action] + frac * noise[i]

        return policy


class ActionTargetInjector(PriorInjector):
    """
    Corresponds to the logic in the original ActionInjectionStrategy.
    Boosts the prior of the trajectory_action.
    """

    def inject(self, policy, config, trajectory_action=None):
        # TODO: a clean way of properly ensuring policy is masked here using legal moves
        if trajectory_action is None:
            return policy

        # Sanity check: user must ensure trajectory_action is legal/possible if filtering
        # Ideally, this injector runs after a selector that ensures the action is present,
        # but here we modify priors before selection to ensure it gets picked if using TopK.

        inject_frac = config.injection_frac

        # Calculate total mass to normalize existing priors
        # We sum over legal moves or all moves depending on how policy is masked.
        # Assuming policy is valid over legal moves.
        total_prior = torch.sum(policy).item()

        # Renormalize priors: put (1-inject_frac) of current mass on existing priors
        policy = (1.0 - inject_frac) * (policy / total_prior)

        # Boost injected action
        policy[trajectory_action] += inject_frac
        return policy


class GumbelInjector(PriorInjector):
    """
    Injects Gumbel noise into the SCORES (logits), used for Gumbel MuZero selection.
    Does not modify the 'policy' probabilities (which remain the raw network output),
    but calculates the 'root_score' values.
    """

    def inject(self, policy, config, trajectory_action=None):
        # Gumbel noise: g = -log(-log(uniform))
        g = -torch.log(-torch.log(torch.rand(len(policy))))

        # Update scores: Score = g + logits
        # We map these back to the full actions space in context.scores
        # TODO: MUST STORE NETWORK PRIOR AND PRIOR IS ESSENTIALLY PRIOR SCORE NOT NETWORK PRIOR
        logits = torch.log(policy + 1e-12).cpu()
        noisy_scores = g + logits

        # TODO: RETURN NOISY SCORES POLICY, TURN LOGITS INTO POLICY, IS BELOW RIGHT?
        return torch.softmax(noisy_scores, dim=-1)
