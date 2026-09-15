from .state import RolloutBufferState
import torch
from tensordict import TensorDict


def write_rollout_step_(
    buffer: RolloutBufferState, step: int, transition: TensorDict
) -> None:
    """Writes a transition batch into the rollout buffer along the time dimension."""
    buffer.data[:, step] = transition


def record_truncations_(
    buffer: RolloutBufferState,
    step: int,
    truncated_envs: torch.Tensor,
    final_observations: torch.Tensor,
) -> None:
    """
    Records final observations for truncated environments.
    Decoupled from Gymnasium's info dictionary structure.

    Args:
        buffer (RolloutBufferState): The rollout buffer state.
        step (int): The current rollout step.
        truncated_envs (torch.Tensor): 1D tensor of environment indices that were truncated.
        final_observations (torch.Tensor): 2D tensor of final observations for those environments.
    """
    for i, env_idx in enumerate(truncated_envs):
        buffer.truncation_records.append((step, int(env_idx), final_observations[i]))
