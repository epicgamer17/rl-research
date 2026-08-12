import torch


# TODO: I think truncation right now works for single env trajectories without auto resetting, but would fail for vectorized envs with auto resetting. as the next_q_values would be on the resetted state instead of the one from the info. verify this, and if it is an issue, unify the vectorization logic in our buffer to work with offline buffers as well somehow.
def compute_v_td_target(
    next_values: torch.Tensor,
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    gamma: torch.Tensor | float,
) -> torch.Tensor:
    """
    Calculates the 1-step Temporal Difference target for state values V(s).
    Formula: y = R_{t} + gamma * (1 - terminated) * V(s_{t+1})

    Supports arbitrary tensor shapes [*Batch, *Channels] (e.g. 1D [B], 2D [B, 1], 2D [B, N_players]).

    Args:
        next_values: Value estimates of the next states [*Batch, *Channels].
        rewards: Rewards for the transitions [*Batch, *Channels].
        terminated: Booleans/floats indicating whether the episodes terminated [*Batch, *Channels].
        gamma: Discount factor(s) (scalar float or tensor matching next_values shape).

    Returns:
        torch.Tensor: The detached TD target of shape [*Batch, *Channels]. Gradients will NOT flow through this tensor.
    """
    assert (
        next_values.shape == rewards.shape == terminated.shape
    ), (
        f"Exact shape mismatch in compute_v_td_target! All input tensors must match shape.\n"
        f"  next_values: {next_values.shape}\n"
        f"  rewards:     {rewards.shape}\n"
        f"  terminated:  {terminated.shape}"
    )

    target = rewards + gamma * next_values * (1.0 - terminated.float())
    return target.detach()

