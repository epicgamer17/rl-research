"""Unit tests for process-safe SharedReplayBuffer and SharedPERBuffer."""

import pytest
import torch
from tensordict import TensorDict
from atomic_rl.buffers import SharedReplayBuffer, SharedPERBuffer


def test_shared_replay_buffer_add_and_sample():
    capacity = 100
    shapes = {"obs": (4,), "action": (1,)}
    buffer = SharedReplayBuffer(capacity, shapes)

    sample = TensorDict(
        {"obs": torch.randn(10, 4), "action": torch.ones(10, 1)}, batch_size=[10]
    )
    buffer.add_transitions(sample)

    assert buffer.size == 10

    generator = torch.Generator().manual_seed(42)
    batch, coords = buffer.sample_batch(generator, batch_size=4, min_size=5)
    assert batch is not None
    assert coords.shape == (4,)
    assert batch["obs"].shape == (4, 4)
    assert batch["action"].shape == (4, 1)


def test_shared_per_buffer_add_and_sample():
    capacity = 100
    shapes = {"obs": (4,), "action": (1,)}
    buffer = SharedPERBuffer(capacity, shapes)

    transitions = TensorDict(
        {"obs": torch.randn(5, 4), "action": torch.zeros(5, 1)}, batch_size=[5]
    )
    buffer.add_transitions(transitions)

    assert buffer.size == 5

    batch, coords, weights = buffer.sample_batch(batch_size=2, beta=0.4, min_size=3)
    assert batch is not None
    assert coords.shape == (2,)
    assert weights is not None
    assert batch["obs"].shape == (2, 4)

    # Test update_priorities using coords directly
    buffer.update_priorities(coords, torch.tensor([1.0, 2.0]), alpha=0.6)


def test_shared_replay_buffer_multidimensional():
    # 20 sequences, each length 15
    capacity = (20, 15)
    shapes = {"obs": (4,), "action": (1,)}
    buffer = SharedReplayBuffer(capacity, shapes)

    # Add 2 sequences
    sample = TensorDict(
        {"obs": torch.randn(2, 15, 4), "action": torch.ones(2, 15, 1)},
        batch_size=[2, 15],
    )
    buffer.add_transitions(sample)

    assert buffer.size == 30

    # Sample individual timesteps
    generator = torch.Generator().manual_seed(42)
    step_batch, (seq_idx, step_idx) = buffer.sample_batch(generator, batch_size=8, min_size=1)
    assert step_batch is not None
    assert step_batch.batch_size == torch.Size([8])
    assert step_batch["obs"].shape == (8, 4)
    assert seq_idx.shape == (8,)
    assert step_idx.shape == (8,)


def test_shared_per_buffer_timestep_level():
    # 10 sequences, each length 8
    capacity = (10, 8)
    shapes = {"obs": (4,), "action": (1,)}
    buffer = SharedPERBuffer(capacity, shapes)

    sample = TensorDict(
        {"obs": torch.randn(3, 8, 4), "action": torch.zeros(3, 8, 1)},
        batch_size=[3, 8],
    )
    buffer.add_transitions(sample)

    assert buffer.size == 24

    batch, (seq_idx, step_idx), weights = buffer.sample_batch(batch_size=4, beta=0.4, min_size=1)
    assert batch is not None
    assert batch.batch_size == torch.Size([4])
    assert batch["obs"].shape == (4, 4)
    assert seq_idx.shape == (4,)
    assert step_idx.shape == (4,)
    assert weights.shape == (4,)

    # Test multi-dimensional update_priorities using coords tuple directly
    buffer.update_priorities((seq_idx, step_idx), torch.ones(4), alpha=0.6)


def test_shared_replay_buffer_sample_unrolls():
    capacity = (10, 20)
    shapes = {"obs": (4,), "action": (1,)}
    buffer = SharedReplayBuffer(capacity, shapes)

    sample = TensorDict(
        {"obs": torch.randn(2, 20, 4), "action": torch.ones(2, 20, 1)},
        batch_size=[2, 20],
    )
    buffer.add_transitions(sample)

    generator = torch.Generator().manual_seed(42)
    unrolls, mask, coords = buffer.sample_unrolls(
        generator, batch_size=4, unroll_steps=5, min_size=1
    )
    assert unrolls is not None
    assert unrolls.batch_size == torch.Size([4, 6])
    assert mask.shape == (4, 6)
    assert isinstance(coords, tuple)
    assert coords[0].shape == (4,)
    assert coords[1].shape == (4,)


def test_shared_per_buffer_sample_unrolls():
    capacity = (10, 20)
    shapes = {"obs": (4,), "action": (1,)}
    buffer = SharedPERBuffer(capacity, shapes)

    sample = TensorDict(
        {"obs": torch.randn(2, 20, 4), "action": torch.ones(2, 20, 1)},
        batch_size=[2, 20],
    )
    buffer.add_transitions(sample)

    unrolls, mask, coords, weights = buffer.sample_unrolls(
        batch_size=4, unroll_steps=5, beta=0.4, min_size=1
    )
    assert unrolls is not None
    assert unrolls.batch_size == torch.Size([4, 6])
    assert mask.shape == (4, 6)
    assert weights.shape == (4,)
    assert isinstance(coords, tuple)


