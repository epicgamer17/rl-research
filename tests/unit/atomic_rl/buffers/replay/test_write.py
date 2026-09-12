import pytest
import torch
import random
import math
from tensordict import TensorDict

from atomic_rl.buffers.replay import (
    circular_write_strategy_,
    compute_is_weights,
    init_buffer,
    init_per_buffer,
    reservoir_write_strategy_,
    update_priorities,
    with_per_tracking,
)
from atomic_rl.buffers.replay.write import _coords_to_tree_indices

pytestmark = pytest.mark.unit


def test_circular_write_strategy():
    capacity = 5
    shapes = {"data": (1,)}
    state = init_buffer(capacity, shapes)

    # 1. Write 3 items
    batch = TensorDict({"data": torch.tensor([[1.0], [2.0], [3.0]])}, batch_size=[3])
    state, indices = circular_write_strategy_(state, batch)

    assert state.size == 3
    assert state.pointer == 3
    assert state.steps_seen == 3
    torch.testing.assert_close(indices, torch.tensor([0, 1, 2]))
    torch.testing.assert_close(state.data["data"][:3], batch["data"])

    # 2. Write 4 items (should wrap around)
    batch2 = TensorDict(
        {"data": torch.tensor([[4.0], [5.0], [6.0], [7.0]])}, batch_size=[4]
    )
    state, indices2 = circular_write_strategy_(state, batch2)

    assert state.size == 5  # capped at capacity
    assert state.pointer == 2  # (3 + 4) % 5 = 2
    assert state.steps_seen == 7
    torch.testing.assert_close(indices2, torch.tensor([3, 4, 0, 1]))
    # Verify values at indices
    assert state.data["data"][3] == 4.0
    assert state.data["data"][4] == 5.0
    assert state.data["data"][0] == 6.0
    assert state.data["data"][1] == 7.0


def test_reservoir_write_strategy():
    random.seed(42)
    capacity = 5
    shapes = {"data": (1,)}

    state = init_buffer(capacity, shapes)

    # Write 10 items
    batch = TensorDict(
        {"data": torch.arange(10, dtype=torch.float32).reshape(-1, 1)}, batch_size=[10]
    )
    state, indices = reservoir_write_strategy_(state, batch)

    assert state.steps_seen == 10
    assert state.size == 5
    # Since it's random, we just check that indices are valid
    assert indices.numel() <= 10
    assert torch.all(indices < 5)


def test_reservoir_write_strategy_2d():
    random.seed(42)
    capacity = 3
    seq_len = 4
    shapes = {"data": (1,), "valid": ()}

    state = init_buffer((capacity, seq_len), shapes)

    # Initially valid_lengths should be all 0s
    assert torch.equal(state.valid_lengths, torch.zeros(capacity, dtype=torch.long))

    # Write 2 sequences with lengths 2 and 3
    valid_mask = torch.tensor(
        [
            [True, True, False, False],
            [True, True, True, False],
        ]
    )
    batch = TensorDict(
        {
            "data": torch.zeros(2, seq_len, 1),
            "valid": valid_mask,
        },
        batch_size=[2, seq_len],
    )
    state, indices = reservoir_write_strategy_(state, batch, valid_mask=valid_mask)

    assert state.steps_seen == 2
    assert indices.tolist() == [0, 1]
    assert state.valid_lengths[0].item() == 2
    assert state.valid_lengths[1].item() == 3
    assert state.valid_lengths[2].item() == 0
    assert state.size == 5


def test_explicit_valid_mask():
    capacity = 3
    seq_len = 4
    shapes = {"data": (1,)}

    state = init_buffer((capacity, seq_len), shapes)

    # Batch with NO "valid" key
    batch = TensorDict({"data": torch.zeros(2, seq_len, 1)}, batch_size=[2, seq_len])
    valid_mask = torch.tensor(
        [
            [True, True, False, False],
            [True, True, True, True],
        ]
    )
    # Pass valid_mask explicitly
    state, indices = circular_write_strategy_(state, batch, valid_mask=valid_mask)

    assert state.valid_lengths[0].item() == 2
    assert state.valid_lengths[1].item() == 4
    assert state.valid_lengths[2].item() == 0
    assert state.size == 6


def test_per_tree_logic():
    capacity = 4
    state = init_per_buffer(capacity, {"data": (1,)})
    # tree_capacity = 4, tree_nodes = 7
    # leaf indices: 3, 4, 5, 6

    # Test _coords_to_tree_indices for 1D
    coords = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    expected_tree_indices = torch.tensor([3, 4, 5, 6], dtype=torch.long)
    assert torch.equal(_coords_to_tree_indices(state, coords), expected_tree_indices)

    # Test _coords_to_tree_indices for 2D
    state_2d = init_per_buffer((2, 2), {"data": (1,)})
    coords_2d = (torch.tensor([0, 1]), torch.tensor([1, 0]))
    # linear: 0*2+1=1, 1*2+0=2 -> tree: 1 + 4 - 1 = 4, 2 + 4 - 1 = 5
    assert torch.equal(_coords_to_tree_indices(state_2d, coords_2d), torch.tensor([4, 5], dtype=torch.long))

    # Update priorities via coords
    priorities = torch.tensor([1.0, 2.0, 3.0, 4.0])
    state = update_priorities(state, coords, priorities, alpha=1.0, epsilon=1e-4)

    # Sum tree should be:
    # root: 10
    # children: 3, 7
    # leaves: 1, 2, 3, 4
    assert math.isclose(state.sum_tree[0].item(), 10.0, rel_tol=1e-5)
    assert math.isclose(state.sum_tree[1].item(), 3.0, rel_tol=1e-5)
    assert math.isclose(state.sum_tree[2].item(), 7.0, rel_tol=1e-5)
    torch.testing.assert_close(state.sum_tree[3:], priorities)

    # Min tree:
    assert math.isclose(state.min_tree[0].item(), 1.0, rel_tol=1e-5)
    assert math.isclose(state.min_tree[1].item(), 1.0, rel_tol=1e-5)
    assert math.isclose(state.min_tree[2].item(), 3.0, rel_tol=1e-5)


def test_with_per_tracking():
    capacity = 4
    state = init_per_buffer(capacity, {"data": (1,)})
    wrapped_write = with_per_tracking(circular_write_strategy_, epsilon=1e-4)

    # 1. Automatic max priority
    batch = TensorDict({"data": torch.tensor([[10.0]])}, batch_size=[1])
    state = wrapped_write(state, batch)

    assert state.size == 1
    assert state.pointer == 1
    # Check if sum tree was updated at leaf index 3
    assert state.sum_tree[3] == state.max_priority
    assert state.sum_tree[0] == state.max_priority

    # 2. Explicit priority override
    batch_prio = TensorDict(
        {"data": torch.tensor([[20.0]]), "priority": torch.tensor([[5.0]])},
        batch_size=[1],
    )
    state = wrapped_write(state, batch_prio)
    # pointer was 1, so written to index 1, tree index 1 + 4 - 1 = 4
    assert state.sum_tree[4] == 5.0

    # 3. 2D Sequence write with valid_mask
    state_2d = init_per_buffer((2, 3), {"data": (1,)})
    wrapped_write_2d = with_per_tracking(circular_write_strategy_, epsilon=1e-4)
    # 1 sequence with 2 valid steps and 1 padding step
    batch_2d = TensorDict({"data": torch.zeros(1, 3, 1)}, batch_size=[1, 3])
    valid_mask_2d = torch.tensor([[True, True, False]])
    state_2d = wrapped_write_2d(state_2d, batch_2d, valid_mask=valid_mask_2d)

    # tree capacity is next power of 2 for 2*3=6, which is 8
    # leaf indices start at 8 - 1 = 7:
    # seq 0, step 0 -> leaf 7 (valid: max_priority = 1.0)
    # seq 0, step 1 -> leaf 8 (valid: max_priority = 1.0)
    # seq 0, step 2 -> leaf 9 (padding: priority = 0.0)
    assert state_2d.sum_tree[7] == 1.0
    assert state_2d.sum_tree[8] == 1.0
    assert state_2d.sum_tree[9] == 0.0
    assert state_2d.sum_tree[0] == 2.0
    # min_tree should track the minimum NON-ZERO priority
    assert state_2d.min_tree[0] == 1.0


def test_compute_is_weights():
    """Test Importance Sampling weight computation with global and batch min_prob."""
    leaf_priorities = torch.tensor([1.0, 2.0, 4.0])
    total_priority = 10.0
    beta = torch.tensor(0.5)

    # 1. Global normalization: passing min_prob from the global buffer
    min_prob_global = 0.05
    is_weights_global = compute_is_weights(
        leaf_priorities, min_prob_global, total_priority, beta
    )

    # probs = [0.1, 0.2, 0.4]
    # is = [ (0.1/0.05)^-0.5, (0.2/0.05)^-0.5, (0.4/0.05)^-0.5 ]
    # is = [ 2^-0.5, 4^-0.5, 8^-0.5 ]
    # is = [ 0.7071, 0.5, 0.3535 ]
    expected_is_global = torch.tensor([0.70710678118, 0.5, 0.35355339059])
    torch.testing.assert_close(is_weights_global, expected_is_global)

    # 2. Batch normalization: passing batch min_prob
    min_prob_batch = leaf_priorities.min() / total_priority  # 0.1
    is_weights_batch = compute_is_weights(
        leaf_priorities, min_prob_batch, total_priority, beta
    )
    # min in batch is 1.0 -> (1/1)^-0.5 = 1.0, (2/1)^-0.5 = 0.7071, (4/1)^-0.5 = 0.5
    expected_is_batch = torch.tensor([1.0, 0.70710678118, 0.5])
    torch.testing.assert_close(is_weights_batch, expected_is_batch)
    assert math.isclose(is_weights_batch.max().item(), 1.0, rel_tol=1e-5)


def test_per_epsilon_minimum_priority():
    """Verify that with_per_tracking and update_priorities enforce the epsilon minimum priority floor."""
    capacity = 4
    shapes = {"data": (1,)}
    eps = 1e-3

    # 1. Test write strategy with explicit zero/tiny priority clamped to epsilon
    state = init_per_buffer(capacity, shapes)
    wrapped_write = with_per_tracking(circular_write_strategy_, epsilon=eps)
    batch = TensorDict(
        {"data": torch.tensor([[1.0], [2.0]]), "priority": torch.tensor([0.0, 1e-7])},
        batch_size=[2],
    )
    state = wrapped_write(state, batch)
    # Both items should be clamped to eps
    assert state.min_tree[0].item() == pytest.approx(eps, rel=1e-5)
    # Sum of two leaves with eps priority
    assert state.sum_tree[0].item() == pytest.approx(2 * eps, rel=1e-5)

    # 2. Test update_priorities with zero TD errors clamped to epsilon
    coords = torch.tensor([0, 1])
    td_errors = torch.tensor([0.0, 0.0])
    state = update_priorities(state, coords, td_errors, alpha=1.0, epsilon=eps)
    assert state.min_tree[0].item() == pytest.approx(eps, rel=1e-5)
    assert state.sum_tree[0].item() == pytest.approx(2 * eps, rel=1e-5)
