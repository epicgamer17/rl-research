import pytest
import torch
from atomic_rl.td.targets import (
    compute_v_td_target,
    compute_q_td_target,
    compute_categorical_q_td_target,
)

pytestmark = pytest.mark.unit


def test_compute_q_td_target():
    # Batch size 2, Actions 3
    next_q = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    next_actions = torch.tensor([2, 0])  # Q-values: 3.0, 4.0
    rewards = torch.tensor([0.5, 1.0])
    terminated = torch.tensor([0.0, 1.0])

    gamma = torch.tensor([0.9, 0.9])

    # Target 0: Terminated=0 -> Should bootstrap
    # 0.5 + 0.9 * 3.0 = 0.5 + 2.7 = 3.2
    # Target 1: Terminated=1 -> Should NOT bootstrap
    # 1.0 + 0.9 * 4.0 * (1 - 1) = 1.0
    expected = torch.tensor([3.2, 1.0])

    target = compute_q_td_target(next_q, next_actions, rewards, terminated, gamma)
    torch.testing.assert_close(target, expected)


def test_compute_categorical_q_td_target():
    # Small scale example for C51 projection
    atom_size = 3
    v_min, v_max = 0.0, 2.0
    support = torch.linspace(v_min, v_max, atom_size)  # [0.0, 1.0, 2.0]

    # Batch size 1, Action 1
    next_logits = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.0, 100.0, 0.0]]]
    )  # Next action 1 is certain at atom 1 (val 1.0)
    next_actions = torch.tensor([1])
    rewards = torch.tensor([0.5])
    terminated = torch.tensor([0.0])

    gamma = torch.tensor([1.0])  # For simplicity

    target_dist = compute_categorical_q_td_target(
        next_logits,
        next_actions,
        rewards,
        terminated,
        gamma,
        support,
        v_min,
        v_max,
        atom_size,
    )

    expected_dist = torch.tensor([[0.0, 0.5, 0.5]])
    torch.testing.assert_close(target_dist, expected_dist)


def test_compute_categorical_q_td_target_terminal():
    atom_size = 3
    v_min, v_max = 0.0, 2.0
    support = torch.linspace(v_min, v_max, atom_size)  # [0, 1, 2]

    next_logits = torch.randn(1, 1, atom_size)  # doesn't matter
    next_actions = torch.tensor([0])
    rewards = torch.tensor([1.2])
    terminated = torch.tensor([1.0])

    gamma = torch.tensor([0.9])

    target_dist = compute_categorical_q_td_target(
        next_logits,
        next_actions,
        rewards,
        terminated,
        gamma,
        support,
        v_min,
        v_max,
        atom_size,
    )

    expected_dist = torch.tensor([[0.0, 0.8, 0.2]])
    torch.testing.assert_close(target_dist, expected_dist)


def test_compute_v_td_target():
    next_values = torch.tensor([3.0, 4.0])
    rewards = torch.tensor([0.5, 1.0])
    terminated = torch.tensor([0.0, 1.0])
    gamma = torch.tensor([0.9, 0.9])

    # Target 0: 0.5 + 0.9 * 3.0 = 3.2
    # Target 1: 1.0 + 0.9 * 4.0 * 0 = 1.0
    expected = torch.tensor([3.2, 1.0])
    target = compute_v_td_target(next_values, rewards, terminated, gamma)
    torch.testing.assert_close(target, expected)


def test_td_multidimensional_channels():
    """Verify that compute_v_td_target and compute_q_td_target support [*Batch, *Channels]."""
    # 1D [B] scalar Q-target
    next_q_1d = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # [2, 3]
    next_actions_1d = torch.tensor([2, 0])  # [2]
    rewards_1d = torch.tensor([0.5, 1.0])  # [2]
    term_1d = torch.tensor([0.0, 1.0])  # [2]

    target_1d = compute_q_td_target(next_q_1d, next_actions_1d, rewards_1d, term_1d, gamma=0.9)
    assert target_1d.shape == (2,)
    torch.testing.assert_close(target_1d, torch.tensor([3.2, 1.0]))

    # 2D [B, 1] single-agent value channel Q-target
    next_q_2d = torch.tensor([[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]])  # [2, 3, 1]
    next_actions_2d = torch.tensor([[2], [0]])  # [2, 1]
    rewards_2d = torch.tensor([[0.5], [1.0]])  # [2, 1]
    term_2d = torch.tensor([[0.0], [1.0]])  # [2, 1]

    target_2d = compute_q_td_target(next_q_2d, next_actions_2d, rewards_2d, term_2d, gamma=0.9)
    assert target_2d.shape == (2, 1)
    torch.testing.assert_close(target_2d, torch.tensor([[3.2], [1.0]]))

    # 3D [B, N_players] multi-player Q-target
    # Batch=1, Actions=2, Players=2
    next_q_3d = torch.tensor([[[1.0, -1.0], [3.0, -3.0]]])  # [1, 2, 2]
    next_actions_3d = torch.tensor([[1]])  # [1, 1]
    rewards_3d = torch.tensor([[0.5, -0.5]])  # [1, 2]
    term_3d = torch.tensor([[0.0, 0.0]])  # [1, 2]

    target_3d = compute_q_td_target(next_q_3d, next_actions_3d, rewards_3d, term_3d, gamma=0.9)
    assert target_3d.shape == (1, 2)
    torch.testing.assert_close(target_3d, torch.tensor([[3.2, -3.2]]))


def test_td_assertions():
    with pytest.raises(AssertionError, match="Exact shape mismatch in compute_v_td_target"):
        compute_v_td_target(
            torch.randn(2, 2), torch.randn(2, 1), torch.randn(2, 2), 0.9
        )

    with pytest.raises(AssertionError, match="Exact shape mismatch in compute_v_td_target"):
        compute_v_td_target(
            torch.randn(2), torch.randn(3), torch.randn(2), 0.9
        )

    with pytest.raises(AssertionError, match="Expected 3D next_logits"):
        compute_categorical_q_td_target(
            torch.randn(2, 2),
            torch.randn(2),
            torch.randn(2),
            torch.randn(2),
            torch.randn(2),
            torch.randn(2),
            0.0,
            1.0,
            2,
        )

    with pytest.raises(AssertionError, match=r"Expected \[B\] next_actions"):

        compute_categorical_q_td_target(
            torch.randn(2, 2, 2),
            torch.randn(2, 2),
            torch.randn(2),
            torch.randn(2),
            torch.randn(2),
            torch.randn(2),
            0.0,
            1.0,
            2,
        )
