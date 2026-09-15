import torch
import pytest
import numpy as np
from pettingzoo.classic import tictactoe_v3
from atomic_rl.envs.wrappers import (
    PettingZooChannelFirstWrapper,
    FrameStackWrapper,
    TwoPlayerPlayerPlaneWrapper,
)
from atomic_rl.envs.functions.tictactoe import tictactoe_dynamics_fn


def test_tictactoe_dynamics_multi_channel():
    """Verify tictactoe_dynamics_fn works for 2, 3, 4, and 5 channel representations."""
    batch_size = 2

    for channels in [2, 3, 4, 5]:
        embed = torch.zeros((batch_size, channels, 3, 3), dtype=torch.float32)
        if channels % 2 == 1:
            # Set player plane for P0
            embed[:, -1] = 1.0

        actions = torch.tensor([0, 4], dtype=torch.long)
        to_play = torch.tensor([0, 0], dtype=torch.long)

        next_embed, reward_vec, next_to_play, is_terminal, next_legal_mask = (
            tictactoe_dynamics_fn(embed, actions, to_play)
        )

        assert next_embed.shape == (batch_size, channels, 3, 3)
        assert (next_to_play == 1).all()
        assert not is_terminal.any()
        assert next_legal_mask.shape == (batch_size, 9)

        if channels % 2 == 1:
            # P1 is next -> Player plane must be 0.0
            assert (next_embed[:, -1] == 0.0).all()


def test_wrappers_composition():
    """Verify FrameStackWrapper and TwoPlayerPlayerPlaneWrapper compose properly."""
    base_env = tictactoe_v3.env()
    env = PettingZooChannelFirstWrapper(base_env)
    env = FrameStackWrapper(env, k=2, channel_first=True)
    env = TwoPlayerPlayerPlaneWrapper(env, channel_first=True)

    env.reset()
    obs, reward, term, trunc, info = env.last()

    assert "observation" in obs
    assert obs["observation"].shape == (5, 3, 3)
    # Active player is P0 -> player plane is 1.0
    assert (obs["observation"][-1] == 1.0).all()

    # P0 plays action 0
    env.step(0)
    obs, reward, term, trunc, info = env.last()

    assert obs["observation"].shape == (5, 3, 3)
    # Active player is P1 -> player plane is 0.0
    assert (obs["observation"][-1] == 0.0).all()


def _embed_board(board, player):
    """Builds a [1, 3, 3, 2] PettingZoo 2-channel embedding from a 3x3 board and player."""
    b = torch.tensor(board, dtype=torch.float32)
    emb = torch.zeros(1, 3, 3, 2)
    p_val = player.item() if isinstance(player, torch.Tensor) else player
    if p_val == 0:
        emb[0, ..., 0] = (b == 1.0).float()
        emb[0, ..., 1] = (b == -1.0).float()
    else:
        emb[0, ..., 0] = (b == -1.0).float()
        emb[0, ..., 1] = (b == 1.0).float()
    return emb


def test_dynamics_p2_places_minus_one_and_flips_turn():
    """P2 (to_play=1) must drop a piece and hand the turn back to P1."""
    next_embed, reward_vec, next_to_play, is_terminal, legal = tictactoe_dynamics_fn(
        _embed_board([[0] * 3, [0] * 3, [0] * 3], 1), torch.tensor([4]), torch.tensor([1])
    )
    # Next active player is P1 (0). Opponent channel (channel 1) contains P2's center piece.
    assert next_embed[0, 1, 1, 1].item() == 1.0  # P2's piece is opponent channel for P1
    assert next_to_play.item() == 0
    assert is_terminal.item() is False
    assert reward_vec[0, 0].item() == 0.0
    assert reward_vec[0, 1].item() == 0.0
    assert legal[0, 4].item() is False  # center now occupied
    assert legal[0, 0].item() is True


def test_dynamics_p2_win_and_loss_reward_sign():
    """Terminal rewards are from the mover's POV; a P2 win is a P1 loss and
    vice-versa, and the reward sign is color-symmetric."""
    # P2 (to_play=1) completes the first row with -1, -1, -1 -> P2 won (P0 reward is -1.0, P1 reward is +1.0)
    _, reward_vec, _, is_term, legal = tictactoe_dynamics_fn(
        _embed_board([[-1, -1, 0], [0, 0, 0], [0, 0, 0]], 1), torch.tensor([2]), torch.tensor([1])
    )
    assert reward_vec[0, 0].item() == -1.0
    assert reward_vec[0, 1].item() == 1.0
    assert is_term.item() is True
    assert legal.sum().item() == 6.0  # terminal row full, six cells still empty

    # Mirror: P1 (to_play=0) completes the same row with +1, +1, +1 -> P1 won (P0 reward is +1.0, P1 reward is -1.0)
    _, reward_vec, _, is_term, _ = tictactoe_dynamics_fn(
        _embed_board([[1, 1, 0], [0, 0, 0], [0, 0, 0]], 0), torch.tensor([2]), torch.tensor([0])
    )
    assert reward_vec[0, 0].item() == 1.0
    assert reward_vec[0, 1].item() == -1.0
    assert is_term.item() is True


def test_dynamics_color_swap_mirror_symmetry():
    """The dynamics must be exactly symmetric under swapping players.

    State B = (board=-A, to_play=1) is the color-swap mirror of state A
    (board=A, to_play=0). Playing the SAME action in both must yield mirrored
    next boards and identical reward / terminal / legal-mask signals.
    """
    board_a = [[1, 0, -1], [0, 1, 0], [0, -1, 0]]  # X at (0,0),(1,1); O at (0,2),(2,1)
    emb_a = _embed_board(board_a, 0)
    emb_b = _embed_board([[-v for v in row] for row in board_a], 1)

    action = torch.tensor([1])  # empty cell (0,1), legal in both mirrors
    next_a, reward_a, to_a, term_a, legal_a = tictactoe_dynamics_fn(emb_a, action, torch.tensor([0]))
    next_b, reward_b, to_b, term_b, legal_b = tictactoe_dynamics_fn(emb_b, action, torch.tensor([1]))

    torch.testing.assert_close(next_a[..., 0], next_b[..., 0])
    assert to_a.item() == 1 - to_b.item()
    torch.testing.assert_close(reward_a, reward_b)
    assert term_a.item() == term_b.item()
    torch.testing.assert_close(legal_a, legal_b)


def test_winner_check_is_color_symmetric():
    """check_tictactoe_winner treats mirrored rows identically."""
    from atomic_rl.envs.functions.tictactoe import check_tictactoe_winner

    reward_a, term_a = check_tictactoe_winner(
        torch.tensor([[1, 1, 1], [0, 0, 0], [0, 0, 0]])[None]
    )
    reward_b, term_b = check_tictactoe_winner(
        torch.tensor([[-1, -1, -1], [0, 0, 0], [0, 0, 0]])[None]
    )
    assert reward_a[0, 0].item() == 1.0
    assert reward_a[0, 1].item() == -1.0
    assert reward_b[0, 0].item() == -1.0
    assert reward_b[0, 1].item() == 1.0
    assert term_a.item() is True
    assert term_b.item() is True

