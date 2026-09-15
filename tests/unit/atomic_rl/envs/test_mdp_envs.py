import pytest
import numpy as np
import torch
import gymnasium as gym
from atomic_rl.envs.mdp import (
    BairdsCounterexampleEnv,
    RandomWalkEnv,
    SlipperyGridWorldEnv,
)


def test_bairds_counterexample_env_gym_api():
    env = BairdsCounterexampleEnv()
    assert isinstance(env, gym.Env)
    assert env.observation_space.shape == (8,)
    assert env.action_space.n == 2

    obs, info = env.reset(seed=42)
    assert isinstance(obs, torch.Tensor)
    assert obs.shape == (8,)
    assert isinstance(info, dict)
    assert "state" in info

    next_obs, reward, terminated, truncated, step_info = env.step()
    assert isinstance(next_obs, torch.Tensor)
    assert next_obs.shape == (8,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "rho" in step_info


def test_random_walk_env_gym_api():
    env = RandomWalkEnv(num_states=5, start_state=2)
    assert isinstance(env, gym.Env)
    assert env.observation_space.shape == (5,)
    assert env.action_space.n == 2

    obs, info = env.reset(seed=123)
    assert isinstance(obs, torch.Tensor)
    assert obs.shape == (5,)
    assert obs[2] == 1.0  # start state is 2

    next_obs, reward, terminated, truncated, step_info = env.step(1)
    assert isinstance(next_obs, torch.Tensor)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_slippery_grid_world_env_gym_api():
    env = SlipperyGridWorldEnv(slip_prob=0.30, max_steps=30)
    assert isinstance(env, gym.Env)
    assert env.observation_space.shape == (16,)
    assert env.action_space.n == 4

    obs, info = env.reset(seed=42)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (16,)
    assert obs[0] == 1.0  # Start position at (0, 0)
    assert np.sum(obs) == 1.0
    assert info["row"] == 0
    assert info["col"] == 0
    assert info["legal_moves"] == [0, 1, 2, 3]
    assert len(info["action_mask"]) == 4

    next_obs, reward, terminated, truncated, step_info = env.step(1)
    assert isinstance(next_obs, np.ndarray)
    assert next_obs.shape == (16,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "chance" in step_info
    assert step_info["chance"] in (0, 1, 2)


def test_slippery_grid_world_deterministic_transitions():
    env = SlipperyGridWorldEnv(slip_prob=0.0, max_steps=30)
    obs, info = env.reset(seed=0)
    assert info["row"] == 0 and info["col"] == 0

    # UP into wall -> bounce, stay at (0, 0)
    obs, reward, term, trunc, info = env.step(0)
    assert info["row"] == 0 and info["col"] == 0
    assert reward == -0.01
    assert not term and not trunc
    assert info["chance"] == 0

    # LEFT into wall -> bounce, stay at (0, 0)
    obs, reward, term, trunc, info = env.step(3)
    assert info["row"] == 0 and info["col"] == 0
    assert reward == -0.01

    # RIGHT -> move to (0, 1)
    obs, reward, term, trunc, info = env.step(1)
    assert info["row"] == 0 and info["col"] == 1
    assert reward == -0.01

    # DOWN -> move to (1, 1) which is 'H' (hole)
    obs, reward, term, trunc, info = env.step(2)
    assert info["row"] == 1 and info["col"] == 1
    assert reward == -1.0
    assert term is True


def test_slippery_grid_world_goal_reach():
    # S...
    # .H.H
    # ...H
    # H..G
    # Path to G: R(0,1), R(0,2), D(1,2), D(2,2), D(3,2), R(3,3)
    env = SlipperyGridWorldEnv(slip_prob=0.0, max_steps=30)
    env.reset(seed=0)

    moves = [1, 1, 2, 2, 2, 1]
    for m in moves[:-1]:
        obs, reward, term, trunc, info = env.step(m)
        assert not term
        assert reward == -0.01

    obs, reward, term, trunc, info = env.step(moves[-1])
    assert info["row"] == 3 and info["col"] == 3
    assert info["cell"] == "G"
    assert reward == 1.0
    assert term is True
    assert trunc is False


def test_slippery_grid_world_truncation():
    env = SlipperyGridWorldEnv(slip_prob=0.0, max_steps=3)
    env.reset(seed=0)

    # Step into wall repeatedly
    _, reward, term, trunc, _ = env.step(0)
    assert not term and not trunc and reward == -0.01
    _, reward, term, trunc, _ = env.step(0)
    assert not term and not trunc and reward == -0.01
    _, reward, term, trunc, _ = env.step(0)
    assert trunc is True
    assert reward == -1.0


def test_slippery_grid_world_slips():
    # With slip_prob = 1.0, intended moves never occur (chance is always 1 or 2)
    env = SlipperyGridWorldEnv(slip_prob=1.0, max_steps=50)
    env.reset(seed=42)

    for _ in range(10):
        _, _, term, trunc, info = env.step(0)
        assert info["chance"] in (1, 2)
        if term or trunc:
            env.reset()
