from ...utils import _allocate_tensordict
from .state import RolloutBufferState
import torch
from tensordict import TensorDict
from typing import Tuple, Callable, List, Optional, Union
from dataclasses import dataclass, field
import numpy as np


def get_rollout_next_values(
    buffer: RolloutBufferState,
    last_values: torch.Tensor,  # shape [B, 1]
    get_value_fn: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """Assembles patched next_values [B, T]."""
    B, T = buffer.data.batch_size
    values = buffer.data["values"]

    if "truncated" in buffer.data.keys():
        truncated_mask = buffer.data["truncated"].bool()
        if truncated_mask.ndim > 2:
            truncated_mask = truncated_mask.any(
                dim=tuple(range(2, truncated_mask.ndim))
            )
        truncated_steps = truncated_mask.nonzero(as_tuple=False)
        if truncated_steps.numel() > 0:
            expected_records = {
                (int(step), int(env_idx)) for env_idx, step in truncated_steps
            }
            actual_records = {
                (int(step), int(env_idx))
                for step, env_idx, _ in buffer.truncation_records
            }
            missing_records = expected_records - actual_records
            if missing_records:
                missing_preview = sorted(missing_records)[:5]
                raise RuntimeError(
                    "Missing final observations for truncated rollout steps. "
                    "Cannot safely bootstrap truncated states because vector env "
                    "autoreset may expose reset observations instead. "
                    f"Missing (step, env_idx) records: {missing_preview}"
                )

    # Explicitly add the time dimension (dim=1) since last_values represents a single timestep
    last_values = last_values.unsqueeze(1)

    assert (
        last_values.ndim == values.ndim
    ), f"Shape mismatch: last_values.ndim ({last_values.ndim}) != values.ndim ({values.ndim}). Ensure last_values matches the feature dimensions of values."
    assert (
        last_values.shape[2:] == values.shape[2:]
    ), f"Feature dimensions mismatch: last_values {last_values.shape[2:]} vs values {values.shape[2:]}"

    # next_values[b, t] should be values[b, t+1] for t < T-1, and last_values[b] for t = T-1
    next_values = torch.cat([values[:, 1:], last_values], dim=1)

    if buffer.truncation_records:
        obs_batch = torch.stack([r[2] for r in buffer.truncation_records]).to(device)
        with torch.inference_mode():
            # get_value_fn must return a tensor shaped [N, *FeatureDims]
            v_patch = get_value_fn(obs_batch)
            assert (
                v_patch.shape[1:] == values.shape[2:]
            ), f"v_patch feature shape {v_patch.shape[1:]} does not match values feature shape {values.shape[2:]}"

        for i, (step, env_idx, _) in enumerate(buffer.truncation_records):
            next_values[env_idx, step] = v_patch[i]

    return next_values


# NOTE (Architectural Rationale for keeping flatten_rollout_buffer):
# In on-policy algorithms (PPO, A2C), rollout collection inherently preserves environment
# and temporal topology with shape [NumEnvs, StepsPerEnv, ...] for GAE calculation, truncation patching,
# and recurrent trajectory tracking. During optimization, training transitions are treated as an i.i.d.
# pool of transitions [NumEnvs * StepsPerEnv, ...].
# flatten_rollout_buffer explicitly marks this semantic phase boundary in algorithm scripts, returning
# a zero-copy view while strictly establishing the 1D batch shape contract required by yield_shuffled_minibatches.
def flatten_rollout_buffer(buffer: RolloutBufferState) -> TensorDict:
    """Flattens [B, T] into [B*T]. (Note: Returns a view of the buffer data)"""
    return buffer.data.flatten(0, 1)


# TODO: this is both an offline (replay buffer) and online (rollout buffer) sampling method, can we make it more general? can we unify the files? what is happening in one that really isnt happening in the other file?
def yield_shuffled_minibatches(
    batch: TensorDict, minibatch_size: int, rng_key: Optional[torch.Generator] = None
):
    """
    Yields shuffled minibatches from a flattened TensorDict.
    Used in PPO, SAC, Behaviour Cloning, and Offline RL (Efficient Zero, MuZero Unplugged, etc).
    """
    assert (
        len(batch.batch_size) == 1
    ), f"Expected 1D batched data, got {batch.batch_size}. Call flatten() first."

    total_size = batch.batch_size[0]
    indices = torch.randperm(total_size, generator=rng_key, device=batch.device)

    for start_idx in range(0, total_size, minibatch_size):
        end_idx = min(start_idx + minibatch_size, total_size)
        mb_indices = indices[start_idx:end_idx]
        yield batch[mb_indices]


# NOTE: Potential fix under consideration: unify yield_shuffled_minibatches and yield_sequential_minibatches
# into a unified generator (e.g. yield_minibatches with a sequential/shuffle flag). Deferred for now pending final decision.
# TODO: should this be done via our new capacity shape system? ie stuff is stored as sequences? or how exactly does sequential minibatches work? I feel like that would kind of make sense to store it as sequences (like we do for muzero) and then only one shuffle_minibatches (that shuffles on a given or hardcoded dim) and returns the shuffled minibatches? or would this idea not work?
def yield_sequential_minibatches(
    buffer_data: TensorDict,
    num_envs: int,
    num_minibatches: int,
    initial_lstm_states: Tuple[torch.Tensor, torch.Tensor],
    rng_key: Optional[torch.Generator] = None,
):
    """
    Yields mini-batches preserving the sequential order of time steps.
    Instead of shuffling individual steps, it shuffles entire environments
    and yields their full unrolled sequences.

    Args:
        buffer_data: TensorDict of shape [envs, steps, ...]
        num_envs: Total number of environments
        num_minibatches: How many mini-batches to split the environments into
        initial_lstm_states: Tuple of (hidden, cell) states from the start of the rollout.
            Expected shape: [layers, envs, hidden_dim]
        rng_key: Optional torch generator for reproducibility.

    Yields:
        mb_flat: Flattened TensorDict of shape [steps * envs_per_batch, ...]
        (mb_initial_h, mb_initial_c): Initial LSTM states for the selected environments.
    """
    assert (
        num_envs % num_minibatches == 0
    ), f"num_envs ({num_envs}) must be divisible by num_minibatches ({num_minibatches}) for LSTM PPO"
    assert (
        len(buffer_data.batch_size) == 2
    ), f"Expected [envs, steps] data, got {buffer_data.batch_size}."

    envs_per_batch = num_envs // num_minibatches

    # Shuffle environment indices, NOT time indices
    env_indices = torch.randperm(num_envs, generator=rng_key, device=buffer_data.device)

    for start in range(0, num_envs, envs_per_batch):
        end = start + envs_per_batch
        mb_env_inds = env_indices[start:end]

        # 1. Select the initial LSTM states for these specific environments
        # All states are typically [layers, envs, hidden_dim]
        mb_initial_states = tuple(
            s[:, mb_env_inds].detach() for s in initial_lstm_states
        )

        # 2. Slice the buffer data for these environments across ALL steps
        # buffer_data is [envs, steps, ...] -> slice to [envs_per_batch, steps, ...]
        mb_data = buffer_data[mb_env_inds, :]
        steps = mb_data.batch_size[1]

        # 3. Flatten the sequence and environment dimensions to [steps * envs_per_batch, ...]
        # We use time-major order (t b) so that when reshaped to (T, B), it stays sequential
        # Output shape: [T * B_per_batch, ...]
        mb_flat = mb_data.apply(
            lambda x: x.transpose(0, 1).reshape(steps * envs_per_batch, *x.shape[2:]),
            batch_size=[steps * envs_per_batch],
        )

        yield mb_flat, mb_initial_states
