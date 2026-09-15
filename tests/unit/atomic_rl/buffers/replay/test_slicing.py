"""Unit tests for slice_unrolls."""

import pytest
import torch
from tensordict import TensorDict

from atomic_rl.buffers import slice_unrolls

pytestmark = pytest.mark.unit


def test_slice_unrolls_basic():
    # 2 sequences, each of length 10
    capacity = 2
    seq_len = 10
    data = TensorDict(
        {
            "obs": torch.arange(capacity * seq_len, dtype=torch.float32).reshape(capacity, seq_len, 1),
            "reward": torch.ones(capacity, seq_len, 1),
        },
        batch_size=[capacity, seq_len],
    )

    # Sample batch of 3 unrolls:
    # Item 0: seq 0, step 2, unroll 3 -> steps [2, 3, 4, 5]
    # Item 1: seq 1, step 0, unroll 3 -> steps [0, 1, 2, 3]
    # Item 2: seq 0, step 8, unroll 3 -> steps [8, 9, 10, 11] (clamped past 9)
    seq_indices = torch.tensor([0, 1, 0], dtype=torch.long)
    step_indices = torch.tensor([2, 0, 8], dtype=torch.long)
    coords = (seq_indices, step_indices)
    unroll_steps = 3
    valid_lengths = torch.tensor([10, 10], dtype=torch.long)

    unrolls, valid_mask = slice_unrolls(
        data, coords, unroll_steps=unroll_steps, valid_lengths=valid_lengths
    )

    assert unrolls.batch_size == torch.Size([3, 4])
    assert valid_mask.shape == (3, 4)

    # Check item 0
    expected_obs_0 = torch.tensor([[2.0], [3.0], [4.0], [5.0]])
    assert torch.equal(unrolls["obs"][0], expected_obs_0)
    assert torch.all(valid_mask[0])

    # Check item 1
    expected_obs_1 = torch.tensor([[10.0], [11.0], [12.0], [13.0]])
    assert torch.equal(unrolls["obs"][1], expected_obs_1)
    assert torch.all(valid_mask[1])

    # Check item 2: steps 8, 9 are valid (< 10); steps 10, 11 are past boundary
    assert torch.equal(valid_mask[2], torch.tensor([True, True, False, False]))


def test_slice_unrolls_with_variable_lengths():
    # Sequence 0 has valid length 4, Sequence 1 has valid length 8
    capacity = 2
    seq_len = 10
    data = TensorDict(
        {"obs": torch.arange(capacity * seq_len, dtype=torch.float32).reshape(capacity, seq_len, 1)},
        batch_size=[capacity, seq_len],
    )
    valid_lengths = torch.tensor([4, 8], dtype=torch.long)

    # Start at step 2 with unroll_steps=3 -> target steps [2, 3, 4, 5]
    seq_indices = torch.tensor([0, 1], dtype=torch.long)
    step_indices = torch.tensor([2, 2], dtype=torch.long)
    coords = (seq_indices, step_indices)

    unrolls, valid_mask = slice_unrolls(
        data, coords, unroll_steps=3, valid_lengths=valid_lengths
    )

    # For seq 0 (len 4): steps 2, 3 are valid; 4, 5 are invalid
    assert torch.equal(valid_mask[0], torch.tensor([True, True, False, False]))

    # For seq 1 (len 8): steps 2, 3, 4, 5 are all valid (< 8)
    assert torch.equal(valid_mask[1], torch.tensor([True, True, True, True]))


def test_slice_unrolls_3d_generic():
    # 3 runs, 4 agents, 10 steps -> batch_size [3, 4, 10]
    data = TensorDict(
        {"val": torch.arange(120, dtype=torch.float32).reshape(3, 4, 10, 1)},
        batch_size=[3, 4, 10],
    )
    run_idx = torch.tensor([0, 1, 2], dtype=torch.long)
    agent_idx = torch.tensor([1, 2, 3], dtype=torch.long)
    step_idx = torch.tensor([7, 1, 0], dtype=torch.long)
    coords = (run_idx, agent_idx, step_idx)
    valid_lengths = torch.full((3, 4), 10, dtype=torch.long)

    unrolls, mask = slice_unrolls(data, coords, unroll_steps=3, valid_lengths=valid_lengths)
    assert unrolls.batch_size == torch.Size([3, 4])
    assert mask.shape == (3, 4)
    # Item 0: start at step 7, len 10 -> steps [7, 8, 9, 10 (invalid)]
    assert torch.equal(mask[0], torch.tensor([True, True, True, False]))
