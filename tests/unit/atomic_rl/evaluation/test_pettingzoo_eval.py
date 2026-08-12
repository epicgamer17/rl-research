"""Unit test for evaluate_pettingzoo_vs_random utility."""

import pytest
import torch
from pettingzoo.classic import tictactoe_v3
from atomic_rl.envs.wrappers import PettingZooChannelFirstWrapper
from atomic_rl.evaluation import evaluate_pettingzoo_vs_random


def test_evaluate_pettingzoo_vs_random():
    def dummy_policy(env, curr_player_idx, obs_t, action_mask_t):
        legal_actions = action_mask_t.nonzero(as_tuple=False).squeeze(-1)
        return legal_actions[0].item()

    metrics = evaluate_pettingzoo_vs_random(
        env_fn=lambda: PettingZooChannelFirstWrapper(tictactoe_v3.env()),
        select_action_fn=dummy_policy,
        num_games=4,
        device=torch.device("cpu"),
    )

    assert "eval/win_rate" in metrics
    assert "eval/draw_rate" in metrics
    assert "eval/loss_rate" in metrics
    assert "eval/p1_win_rate" in metrics
    assert "eval/p2_win_rate" in metrics
    assert metrics["eval/win_rate"] >= 0.0
