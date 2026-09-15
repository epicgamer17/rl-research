"""Unit tests for action encoding transformations."""

import torch
import pytest
from atomic_rl.transforms import build_action_encodings


def test_build_action_encodings_grid():
    """Test 1-hot grid encoding for TicTacToe (3x3)."""
    def tictactoe_map(action: int, target: torch.Tensor):
        target[0, action // 3, action % 3] = 1.0

    encodings = build_action_encodings(9, (1, 3, 3), tictactoe_map)
    assert encodings.shape == (9, 1, 3, 3)

    # Check corners and center
    assert encodings[0, 0, 0, 0] == 1.0
    assert encodings[4, 0, 1, 1] == 1.0
    assert encodings[8, 0, 2, 2] == 1.0
    assert encodings[2, 0, 0, 2] == 1.0
    assert encodings.sum() == 9.0

    # Test batched indexing
    actions = torch.tensor([0, 4, 8, 2], dtype=torch.long)
    batch_planes = encodings[actions]
    assert batch_planes.shape == (4, 1, 3, 3)
    assert batch_planes[0, 0, 0, 0] == 1.0
    assert batch_planes[1, 0, 1, 1] == 1.0
    assert batch_planes[2, 0, 2, 2] == 1.0
    assert batch_planes[3, 0, 0, 2] == 1.0


def test_build_action_encodings_columns():
    """Test column-strip encoding for Connect-4 (6 rows, 7 cols)."""
    def connect4_map(action: int, target: torch.Tensor):
        target[0, :, action] = 1.0

    encodings = build_action_encodings(7, (1, 6, 7), connect4_map)
    assert encodings.shape == (7, 1, 6, 7)

    # Verify column 3 illuminates all 6 rows in column 3
    assert (encodings[3, 0, :, 3] == 1.0).all()
    assert encodings[3, 0, :, :3].sum() == 0.0
    assert encodings[3, 0, :, 4:].sum() == 0.0
    assert encodings.sum() == 7 * 6


def test_build_action_encodings_broadcast():
    """Test broadcast / tiled constant plane encoding for Atari."""
    num_actions = 4

    def atari_map(action: int, target: torch.Tensor):
        target[0, :, :] = float(action) / float(num_actions)

    encodings = build_action_encodings(num_actions, (1, 8, 8), atari_map)
    assert encodings.shape == (4, 1, 8, 8)
    assert (encodings[0] == 0.0).all()
    assert (encodings[2] == 0.5).all()
