"""
Temporal Return and Advantage Calculation Primitives (returns.py)

This module provides pure, stateless temporal return and advantage kernels  (Monte Carlo returns, N-step bootstrapped returns, Generalized Advantage Estimation (GAE), and TD(lambda) returns).

1. The Trailing Channel Paradigm [B, T, *Channels]:
   All functions in this module operate on tensors with shape [B, T, *Channels]:
     - dim 0 (B): Batch size (e.g. parallel environment count or trajectory batch).
     - dim 1 (T): Time steps (sequence length of the rollout/episode).
     - dim 2+ (*Channels): Arbitrary trailing value/reward channels.

   Why *Channels?
   In Reinforcement Learning, state values and rewards are not always scalar floats [B, T]:
     - Single-Agent Scalar RL (PPO, A2C, DQN): [B, T, 1]
     - Multi-Player / Vector Value RL (PettingZoo, AlphaZero): [B, T, N_players]
     - Multi-Objective RL / Vector Rewards (Speed, Energy, Safety): [B, T, F_objectives]
     - Multi-Agent Multi-Objective RL: [B, T, N_players, F_objectives]
     - Multi-Agent Categorical Value RL: [B, T, N_agents, N_atoms]

   Because temporal discounting math (r_t + gamma * (1-done) * V_{t+1}) is linear along the time axis (dim 1), it operates purely element-wise across all trailing channels without needing any domain-specific branching or magic rank-dispatching logic.

2. Fail-Fast Exact Geometry Assertions:
   PyTorch automatically broadcasts trailing dimensions from right to left, which can cause silent, hard-to-debug logic errors (e.g. accidentally broadcasting a [B, T, 1] value tensor against a [B, T, 4] reward tensor). To prevent this:
     - All input tensors (rewards, values, next_values, terminated, truncated) MUST have the EXACT same shape: [B, T, *Channels].
     - Input tensors will fail fast with an explicit AssertionError if shapes mismatch at the call site.
     - Functions NEVER automatically squeeze or unsqueeze trailing dimensions internally.

USAGE EXAMPLE:
--------------
  # PPO Single-Agent (CartPole)
  advantages = compute_gae(rewards, terminated, truncated, values, next_values, gamma, gae_lambda)
  # Input shapes:  all [B, T, 1]
  # Output shape:  [B, T, 1]

  # AlphaZero / PettingZoo 4-Player Board Game advantages = compute_gae(rewards, terminated, truncated, values, next_values, gamma, gae_lambda)
  # Input shapes:  all [B, T, 4]
  # Output shape:  [B, T, 4]
"""

import torch
import warnings


# TODO: eventually will have to figure out how to get these working with MuZero (alternating players and all). it works fine for single agent or if we treat the opponent as part of the environment (ie not an agent), but for things like muzero where the agent learns as both players and value depends on the agents perspective this needs to be updated. (ie for chess, players alternate so if p1 won at the end of the game they get a reward 1, but the mc returns for all the states is not 1, it is 1, -1, -1, 1, -1, ...). There may be several solutions, for one i know mctx does there tree search without worrying about player turn, is there a way to do that? maybe some kind of higher order function to make it player turn aware? or another function to flip the sign of the returns depending on the player the user uses before passing to these or something?


def compute_mc_returns(
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """
    Computes batched Monte Carlo discounted returns. The same as a compute discounted rewards function, with the soft expectation that all trajectories have been terminated at some point.

    Formula: G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...

    MC returns are computed by backward-iterating through the rewards, stopping at
    either termination or truncation.

    Args:
        rewards: Reward tensor [B, T, *Channels].
        terminated: Termination mask matching rewards shape [B, T, *Channels].
        truncated: Truncation mask matching rewards shape [B, T, *Channels].
        gamma: Discount factor.

    Returns:
        Discounted returns [B, T, *Channels].
        Note: The returned tensor is not explicitly detached.
    """
    assert (
        rewards.ndim >= 2
    ), f"Expected at least 2D rewards [B, T, ...], got {rewards.shape}"
    assert (
        rewards.shape == terminated.shape == truncated.shape
    ), f"Shape mismatch in MC returns inputs: rewards {rewards.shape}, terminated {terminated.shape}, truncated {truncated.shape}"

    # Combine masks: MC returns stop at either terminated or truncated
    # Both mean the trajectory ended for the purpose of the current return calculation.
    done = (terminated.bool() | truncated.bool()).float()

    # Validation: Warn if any trajectory in the batch is never done.
    # Infinite-horizon MC returns are biased if not properly handled.
    is_never_terminal = ~(terminated.bool().view(rewards.shape[0], -1).any(dim=1))
    if is_never_terminal.any():
        warnings.warn(
            "Found trajectory in batch where terminated is always False (never terminal). "
            "MC returns for these trajectories may be biased."
        )
    is_truncated_but_not_terminated = (
        (truncated.bool() & ~terminated.bool()).view(rewards.shape[0], -1).any(dim=1)
    )
    if is_truncated_but_not_terminated.any():
        warnings.warn(
            "Found trajectory in batch where truncated is True but not terminated."
            "MC returns for these trajectories may be biased."
        )

    returns = torch.zeros_like(rewards)
    R = torch.zeros_like(rewards[:, 0])  # [B, *Channels]

    # Backward iteration for efficient O(T) calculation
    for t in reversed(range(rewards.size(1))):
        # If done is True (1.0), the next R gets multiplied by 0 (bootstrapping stops)
        mask = 1.0 - done[:, t]
        R = rewards[:, t] + gamma * R * mask
        returns[:, t] = R

    return returns


def compute_n_step_returns(
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    gamma: float,
    n: int,
) -> torch.Tensor:
    """
    Computes batched n-step bootstrapped discounted returns.

    Formula: G_t^{(n)} = r_t + gamma r_{t+1} + ... + gamma^{n-1} r_{t+n-1} + gamma^n V(s_{t+n})

    Correctly handles both terminated (no bootstrap) and truncated (bootstrap) states.

    Args:
        rewards: Reward tensor [B, T, *Channels].
        terminated: Termination mask [B, T, *Channels].
        truncated: Truncation mask [B, T, *Channels].
        values: State values V(s_t) [B, T, *Channels].
        next_values: Next state values V(s_{t+1}) [B, T, *Channels].
        gamma: Discount factor.
        n: Number of steps to look ahead.

    Returns:
        N-step returns [B, T, *Channels].
        Note: The returned tensor is not explicitly detached.
    """
    assert (
        rewards.ndim >= 2
    ), f"Expected at least 2D rewards [B, T, ...], got {rewards.shape}"
    assert (
        rewards.shape
        == terminated.shape
        == truncated.shape
        == values.shape
        == next_values.shape
    ), (
        f"Exact shape mismatch in n-step returns! All input tensors must have the exact same shape.\n"
        f"  rewards:     {rewards.shape}\n"
        f"  terminated:  {terminated.shape}\n"
        f"  truncated:   {truncated.shape}\n"
        f"  values:      {values.shape}\n"
        f"  next_values: {next_values.shape}"
    )
    assert n >= 1, f"n-step must be at least 1, got {n}"

    T = rewards.size(1)
    done = (terminated.bool() | truncated.bool()).float()

    # Base case: 1-step returns (Pure Bellman math)
    # G_t^{(1)} = r_t + gamma * (1 - terminated_t) * V(s_{t+1})
    returns = rewards + gamma * (1.0 - terminated.float()) * next_values.detach()

    # Recursive calculation: G_t^{(i)} = r_t + gamma * (1 - done_t) * G_{t+1}^{(i-1)}
    for i in range(1, n):
        if T > i:
            # Only update steps where the episode hasn't ended.
            # If done[:, t] is True, G_t already contains the terminal/truncated return.
            mask = (1.0 - done[:, :-i]).bool()
            returns[:, :-i] = torch.where(
                mask,
                rewards[:, :-i] + gamma * returns[:, 1 : T - i + 1],
                returns[:, :-i],
            )

    return returns


def compute_gae(
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> torch.Tensor:
    """
    Computes the Generalized Advantage Estimate (GAE) for a batch of trajectories.

    Args:
        rewards (torch.Tensor): The reward tensor of shape [batch, time, *Channels].
        terminated (torch.Tensor): Boolean/float mask indicating episode termination.
        truncated (torch.Tensor): Boolean/float mask indicating episode truncation.
        values (torch.Tensor): The values tensor of shape [batch, time, *Channels].
        next_values (torch.Tensor): The values for the next state in the batch.
        gamma (float): The discount factor.
        gae_lambda (float): The GAE lambda parameter (typically 0.95 or 0.99).

    Returns:
        The Generalized Advantage Estimate [B, T, *Channels].
        Note: The returned tensor is not explicitly detached.
    """
    assert (
        rewards.ndim >= 2
    ), f"Expected at least 2D rewards [B, T, ...], got {rewards.shape}"
    assert (
        rewards.shape
        == terminated.shape
        == truncated.shape
        == values.shape
        == next_values.shape
    ), (
        f"Exact shape mismatch in compute_gae! All input tensors must have the exact same shape.\n"
        f"  rewards:     {rewards.shape}\n"
        f"  terminated:  {terminated.shape}\n"
        f"  truncated:   {truncated.shape}\n"
        f"  values:      {values.shape}\n"
        f"  next_values: {next_values.shape}"
    )

    # Combined done mask
    done = (terminated.bool() | truncated.bool()).float()

    # Pure Bellman Math
    deltas = rewards + gamma * (1.0 - terminated.float()) * next_values - values

    gae = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(rewards[:, 0])  # [B, *Channels]

    for t in reversed(range(rewards.shape[1])):
        mask = 1.0 - done[:, t].float()
        last_gae = deltas[:, t] + gamma * gae_lambda * mask * last_gae
        gae[:, t] = last_gae

    return gae


def compute_td_lambda_returns(
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> torch.Tensor:
    """
    Computes TD(lambda) returns.
    Mathematically, TD(λ) returns are equivalent to GAE + State Values.
    We preserve compute_gae as a separate function for semantic readability,
    but compose it here to unify the returns API.

    Returns:
        The TD(lambda) returns [B, T, *Channels].
        Note: The returned tensor is not explicitly detached.
    """
    advantages = compute_gae(
        rewards, terminated, truncated, values, next_values, gamma, gae_lambda
    )
    return advantages + values.detach()
