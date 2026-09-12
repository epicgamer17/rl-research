import pytest
import torch
from tensordict import TensorDict

from atomic_rl.buffers.replay import (
    init_buffer,
    init_per_buffer,
    sample_per,
    uniform_sample,
    update_priorities,
    with_unroll_slicing,
)

pytestmark = pytest.mark.unit


def test_uniform_sample():
    capacity = 10
    state = init_buffer(capacity, {"data": (1,)})
    state.valid_lengths.fill_(5)
    state.data["data"][:5] = torch.arange(5, dtype=torch.float32).reshape(-1, 1)

    gen = torch.Generator()
    gen.manual_seed(42)

    sample, coords = uniform_sample(state, gen, batch_size=3)
    assert sample.batch_size == torch.Size([3])
    assert torch.all(sample["data"] < 5)
    assert coords.shape == (3,)
    assert torch.equal(sample["data"], state.data["data"][coords])

    with pytest.raises(ValueError):
        uniform_sample(state, gen, batch_size=10)


def test_sample_per():
    capacity = 4
    state = init_per_buffer(capacity, {"data": (1,)})
    state.valid_lengths.fill_(4)
    state.data["data"][:] = torch.arange(4, dtype=torch.float32).reshape(-1, 1)

    # Set priorities using coords [0, 1, 2, 3] such that the last one is very likely
    coords = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    priorities = torch.tensor([0.1, 0.1, 0.1, 10.0])
    state = update_priorities(state, coords, priorities, alpha=1.0, epsilon=1e-4)

    beta = torch.tensor(0.4)
    batch, sampled_coords, is_weights = sample_per(state, batch_size=100, beta=beta)

    # Most sampled should be coord 3
    counts = torch.bincount(sampled_coords)
    assert torch.argmax(counts) == 3

    # Check IS weights shape
    assert is_weights.shape == (100,)
    # max weight should be for the lowest priority item
    assert is_weights[sampled_coords == 0].max() > is_weights[sampled_coords == 3].max()


def test_uniform_sample_multidimensional():
    # Capacity: 4 sequences, each length 5
    capacity = (4, 5)
    state = init_buffer(capacity, {"data": (1,)})
    state.steps_seen = 2
    state.valid_lengths[:2] = 5
    assert state.size == 10  # 2 sequences of length 5 = 10 total steps

    # Fill data: seq 0 has values 0..4, seq 1 has values 10..14
    state.data["data"][0] = torch.arange(5, dtype=torch.float32).reshape(-1, 1)
    state.data["data"][1] = (torch.arange(5, dtype=torch.float32) + 10).reshape(-1, 1)

    gen = torch.Generator().manual_seed(42)

    # Sample individual timesteps via prefix sum
    step_batch, (seq_indices, step_indices) = uniform_sample(
        state, gen, batch_size=6
    )
    assert step_batch.batch_size == torch.Size([6])
    assert step_batch["data"].shape == (6, 1)
    assert seq_indices.shape == (6,)
    assert step_indices.shape == (6,)
    assert (seq_indices < 2).all()
    assert (step_indices < 5).all()


def test_sample_per_timestep_level():
    # Capacity: 2 sequences, each of length 3 -> total 6 timesteps
    capacity = (2, 3)
    state = init_per_buffer(capacity, {"data": (1,)})
    state.steps_seen = 2
    state.valid_lengths[:2] = 3
    assert state.size == 6

    # Put distinct data in each step
    val = 0
    for s in range(2):
        for t in range(3):
            state.data["data"][s, t] = float(val)
            val += 1

    # Update priorities directly using multi-dimensional coordinates:
    # (seq 1, step 2) has high priority
    seq_c = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    step_c = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long)
    coords = (seq_c, step_c)
    priorities = torch.tensor([0.01, 0.01, 0.01, 0.01, 0.01, 100.0])
    state = update_priorities(state, coords, priorities, alpha=1.0, epsilon=1e-4)

    beta = torch.tensor(0.5)
    batch, (seq_indices, step_indices), is_weights = sample_per(
        state, batch_size=50, beta=beta
    )

    # Batch should be individual timesteps
    assert batch.batch_size == torch.Size([50])
    assert batch["data"].shape == (50, 1)
    assert is_weights.shape == (50,)
    assert seq_indices.shape == (50,)
    assert step_indices.shape == (50,)
    # Most sampled should be value 5.0 (seq 1, step 2)
    assert (batch["data"] == 5.0).sum() > 40


def test_uniform_sample_3d_generic():
    # Capacity: (3, 4, 5) -> e.g. 3 runs, 4 agents, 5 steps
    capacity = (3, 4, 5)
    state = init_buffer(capacity, {"val": (1,)})
    state.valid_lengths[:2, :] = 5
    assert state.size == 40  # 2 runs * 4 agents * 5 steps = 40 total elements
    gen = torch.Generator().manual_seed(42)

    # Sample over all 3 dims: (0, 1, 2)
    batch, (d0, d1, d2) = uniform_sample(
        state, gen, batch_size=10
    )
    assert batch.batch_size == torch.Size([10])
    assert d0.shape == (10,)
    assert d1.shape == (10,)
    assert d2.shape == (10,)
    assert (d0 < 2).all()
    assert (d1 < 4).all()
    assert (d2 < 5).all()


def test_prefix_sum_sampling_variable_lengths():
    """Validates Option A Prefix Sum Sampling over sequences with variable valid lengths."""
    # Capacity: 4 sequences, max length 10
    capacity = (4, 10)
    state = init_buffer(capacity, {"step_val": (1,)})
    # Seq 0: length 3
    # Seq 1: length 7
    # Seq 2: length 2
    # Seq 3: empty (0)
    lengths = torch.tensor([3, 7, 2, 0], dtype=torch.long)
    state.valid_lengths = lengths.clone()
    assert state.size == 12  # 12 valid steps

    for s in range(3):
        for t in range(lengths[s].item()):
            state.data["step_val"][s, t] = float(s * 100 + t)

    gen = torch.Generator().manual_seed(123)
    all_seq_idx = []
    all_step_idx = []
    all_vals = []
    for _ in range(200):
        batch, (s_idx, t_idx) = uniform_sample(
            state, gen, batch_size=12
        )
        all_seq_idx.append(s_idx)
        all_step_idx.append(t_idx)
        all_vals.append(batch["step_val"].squeeze(-1).long())

    seq_idx = torch.cat(all_seq_idx)
    step_idx = torch.cat(all_step_idx)
    sampled_vals = torch.cat(all_vals)

    # 1. 100% of samples must strictly fall within valid sequence bounds
    for i in range(len(seq_idx)):
        s = seq_idx[i].item()
        t = step_idx[i].item()
        assert 0 <= s < 3, f"Sampled invalid sequence index {s}"
        assert 0 <= t < lengths[s].item(), f"Sampled invalid step index {t} for seq {s} with length {lengths[s]}"

    # 2. Check empirical distribution: each of the 12 items should be sampled ~ 2400 / 12 = 200 times
    unique_vals, counts = torch.unique(sampled_vals, return_counts=True)
    assert len(unique_vals) == 12, "All 12 valid steps should be sampled"
    # Binomial test tolerance: min count > 100, max count < 320
    assert counts.min() > 100, f"Under-sampled element: min count {counts.min()}"
    assert counts.max() < 320, f"Over-sampled element: max count {counts.max()}"


def test_with_unroll_slicing_uniform():
    capacity = (4, 10)
    state = init_buffer(capacity, {"val": (1,)})
    state.valid_lengths[:2] = 8
    state.data["val"][0] = torch.arange(10, dtype=torch.float32).reshape(-1, 1)
    state.data["val"][1] = (torch.arange(10, dtype=torch.float32) + 20).reshape(-1, 1)

    gen = torch.Generator().manual_seed(42)

    # Test via higher-order wrapper directly
    unroll_sampler = with_unroll_slicing(uniform_sample)
    unrolls, mask, coords = unroll_sampler(state, gen, batch_size=4, unroll_steps=3)
    assert unrolls.batch_size == torch.Size([4, 4])
    assert mask.shape == (4, 4)
    assert coords[0].shape == (4,)
    assert coords[1].shape == (4,)


def test_with_unroll_slicing_per():
    capacity = (4, 10)
    state = init_per_buffer(capacity, {"val": (1,)})
    state.steps_seen = 2
    state.valid_lengths[:2] = 8
    state.data["val"][:] = 1.0

    # Set non-zero priorities on written sequences
    coords = (torch.tensor([0, 1]), torch.tensor([0, 0]))
    state = update_priorities(state, coords, torch.tensor([1.0, 2.0]), alpha=0.6, epsilon=1e-4)

    beta = torch.tensor(0.4)
    # Test via higher-order wrapper directly
    unroll_sampler = with_unroll_slicing(sample_per)
    unrolls, mask, coords, weights = unroll_sampler(
        state, batch_size=4, beta=beta, unroll_steps=3
    )
    assert unrolls.batch_size == torch.Size([4, 4])
    assert mask.shape == (4, 4)
    assert weights.shape == (4,)


