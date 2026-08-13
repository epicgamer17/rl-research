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
