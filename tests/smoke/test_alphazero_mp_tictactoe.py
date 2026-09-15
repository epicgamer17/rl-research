"""Smoke tests for AlphaZero TicTacToe network components and search functions."""

import torch
import pytest
from examples.alphazero.alphazero_mp_tictactoe import (
    TicTacToeNet,
    create_recurrent_fn,
    run_batched_mcts,
)

pytestmark = pytest.mark.smoke


def test_alphazero_net_shapes():
    net = TicTacToeNet(in_channels=7, num_filters=16, num_res_blocks=2)
    obs = torch.zeros((2, 7, 3, 3), dtype=torch.float32)

    policy_logits, value = net(obs)
    assert policy_logits.shape == (2, 9)
    assert value.shape == (2,)


def test_alphazero_batched_mcts():
    net = TicTacToeNet(in_channels=7, num_filters=16, num_res_blocks=2)
    net.eval()
    recurrent_fn = create_recurrent_fn(net)

    batch_size = 2
    root_embed = torch.zeros((batch_size, 7, 3, 3), dtype=torch.float32)
    curr_players = torch.tensor([0, 1], dtype=torch.long)
    legal_masks = torch.ones((batch_size, 9), dtype=torch.bool)

    probs = run_batched_mcts(
        model=net,
        recurrent_fn=recurrent_fn,
        root_embed=root_embed,
        curr_players=curr_players,
        legal_masks=legal_masks,
        num_simulations=5,
        temperature=1.0,
        explore_dirichlet=False,
    )

    assert probs.shape == (batch_size, 9)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(batch_size))
