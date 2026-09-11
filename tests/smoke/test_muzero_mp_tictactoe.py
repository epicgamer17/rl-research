"""Smoke tests for MuZero TicTacToe network components and search functions."""

import torch
import pytest
from examples.muzero.muzero_mp_tictactoe import (
    TicTacToeMuZeroNet,
    NUM_CHANNELS,
    TICTACTOE_ACTION_ENCODINGS,
    min_max_normalize,
    scale_gradient,
    create_recurrent_fn,
    run_batched_mcts,
)

pytestmark = pytest.mark.smoke


def test_tictactoe_action_encodings():
    actions = torch.tensor([0, 4, 8, 2], dtype=torch.long)
    planes = TICTACTOE_ACTION_ENCODINGS[actions]

    assert planes.shape == (4, 1, 3, 3)
    assert planes[0, 0, 0, 0] == 1.0
    assert planes[1, 0, 1, 1] == 1.0
    assert planes[2, 0, 2, 2] == 1.0
    assert planes[3, 0, 0, 2] == 1.0
    assert planes.sum() == 4.0


def test_min_max_normalize():
    tensor = torch.tensor([[[[1.0, 2.0], [3.0, 5.0]]]], dtype=torch.float32)
    normed = min_max_normalize(tensor, dim=(-3, -2, -1))

    assert normed.min() == 0.0
    assert normed.max() == 1.0
    assert torch.allclose(normed[0, 0, 0, 0], torch.tensor(0.0))
    assert torch.allclose(normed[0, 0, 1, 1], torch.tensor(1.0))


def test_scale_gradient():
    x = torch.tensor([2.0], requires_grad=True)
    scaled = scale_gradient(x, 0.5)
    loss = (scaled ** 2).sum()
    loss.backward()

    # d(x^2)/dx = 2 * x = 4. Scaled by 0.5 -> 2.0
    assert torch.allclose(x.grad, torch.tensor([2.0]))


def test_muzero_net_shapes():
    net = TicTacToeMuZeroNet(in_channels=NUM_CHANNELS, num_filters=16, num_res_blocks=2)
    obs = torch.zeros((2, NUM_CHANNELS, 3, 3), dtype=torch.float32)

    # 1. Initial inference
    s0, logits, val = net.initial_inference(obs)
    assert s0.shape == (2, 16, 3, 3)
    assert logits.shape == (2, 9)
    assert val.shape == (2,)

    # 2. Recurrent inference
    actions = torch.tensor([1, 5], dtype=torch.long)
    next_s, reward, r_logits, r_val = net.recurrent_inference(s0, actions)
    assert next_s.shape == (2, 16, 3, 3)
    assert reward.shape == (2, 1)
    assert r_logits.shape == (2, 9)
    assert r_val.shape == (2,)


def test_muzero_batched_mcts():
    net = TicTacToeMuZeroNet(in_channels=NUM_CHANNELS, num_filters=16, num_res_blocks=2)
    net.eval()
    recurrent_fn = create_recurrent_fn(net)

    batch_size = 2
    root_embed = torch.zeros((batch_size, NUM_CHANNELS, 3, 3), dtype=torch.float32)
    curr_players = torch.tensor([0, 1], dtype=torch.long)
    legal_masks = torch.ones((batch_size, 9), dtype=torch.bool)

    probs, root_values = run_batched_mcts(
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
    assert root_values.shape == (batch_size,)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(batch_size))

