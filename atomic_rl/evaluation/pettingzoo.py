"""Evaluation utilities for PettingZoo multi-agent environments."""

import random
from typing import Any, Callable, Dict
import torch

from atomic_rl.envs.functions.pettingzoo import get_agent_index


def evaluate_pettingzoo_vs_random(
    env_fn: Callable[[], Any],
    select_action_fn: Callable[[Any, int, torch.Tensor, torch.Tensor], int],
    num_games: int = 100,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """
    Evaluates an agent policy against a Random baseline in a PettingZoo AEC environment.

    Alternates player roles (eval agent as Player 0, then Player 1, etc.) to ensure balanced evaluation.

    Args:
        env_fn: Factory function returning an instantiated AEC PettingZoo environment.
        select_action_fn: Function (env, player_idx, obs_tensor, legal_mask_tensor) -> action_idx.
        num_games: Total number of evaluation games.
        device: Torch device.

    Returns:
        Dict of evaluation metrics (overall win/draw/loss rates and per-player rates).
    """
    temp_env = env_fn()
    possible_agents = list(temp_env.possible_agents)
    num_players = len(possible_agents)
    temp_env.close()

    p_wins = [0] * num_players
    p_draws = [0] * num_players
    p_losses = [0] * num_players

    for game_i in range(num_games):
        eval_player_idx = game_i % num_players
        eval_agent_name = possible_agents[eval_player_idx]

        env = env_fn()
        env.reset()

        eval_agent_reward = 0.0

        for agent in env.agent_iter():
            obs, reward, term, trunc, _ = env.last()

            if agent == eval_agent_name:
                eval_agent_reward += reward

            if term or trunc:
                env.step(None)
                continue

            curr_player_idx = get_agent_index(env)
            action_mask_t = torch.tensor(
                obs["action_mask"], device=device, dtype=torch.bool
            )

            if curr_player_idx == eval_player_idx:
                obs_t = torch.tensor(
                    obs["observation"], device=device, dtype=torch.float32
                )
                action_idx = select_action_fn(
                    env, curr_player_idx, obs_t, action_mask_t
                )
            else:
                legal_actions = (
                    action_mask_t.nonzero(as_tuple=False).squeeze(-1).tolist()
                )
                action_idx = random.choice(legal_actions)

            env.step(action_idx)

        env.close()

        if eval_agent_reward > 0:
            p_wins[eval_player_idx] += 1
        elif eval_agent_reward < 0:
            p_losses[eval_player_idx] += 1
        else:
            p_draws[eval_player_idx] += 1

    total_wins = sum(p_wins)
    total_draws = sum(p_draws)
    total_losses = sum(p_losses)
    total_games = max(1, num_games)

    metrics = {
        "eval/win_rate": total_wins / total_games,
        "eval/draw_rate": total_draws / total_games,
        "eval/loss_rate": total_losses / total_games,
    }

    for p_idx in range(num_players):
        p_total = max(1, p_wins[p_idx] + p_draws[p_idx] + p_losses[p_idx])
        metrics[f"eval/p{p_idx + 1}_win_rate"] = p_wins[p_idx] / p_total
        metrics[f"eval/p{p_idx + 1}_draw_rate"] = p_draws[p_idx] / p_total
        metrics[f"eval/p{p_idx + 1}_loss_rate"] = p_losses[p_idx] / p_total

    return metrics
