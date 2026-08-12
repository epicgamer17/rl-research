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
    buffer.add_samples(sample)

    assert buffer.size == 10

    generator = torch.Generator().manual_seed(42)
    batch = buffer.sample_batch(generator, batch_size=4, min_size=5)
    assert batch is not None
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

    batch, tree_indices, weights = buffer.sample_batch(batch_size=2, beta=0.4, min_size=3)
    assert batch is not None
    assert tree_indices is not None
    assert weights is not None
    assert batch["obs"].shape == (2, 4)
