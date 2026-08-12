"""Functional helper utilities for PettingZoo multi-agent environments."""

from typing import Any
import torch


def get_agent_index(env: Any) -> int:
    """Returns 0-indexed integer for PettingZoo active agent selection."""
    return env.possible_agents.index(env.agent_selection)


def get_terminal_rewards_vec(env: Any) -> torch.Tensor:
    """Returns absolute rewards vector for each agent from PettingZoo env.rewards dict."""
    return torch.tensor(
        [env.rewards[agent] for agent in env.possible_agents],
        dtype=torch.float32,
    )
