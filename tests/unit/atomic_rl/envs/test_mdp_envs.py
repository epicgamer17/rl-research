import pytest
import numpy as np
import torch
import gymnasium as gym
from atomic_rl.envs.mdp import (
    BairdsCounterexampleEnv,
    RandomWalkEnv,
    SlipperyGridEnv,
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


def test_slippery_grid_env_gym_api():
    env = SlipperyGridEnv(slip_probability=0.2)
    assert isinstance(env, gym.Env)
    assert env.observation_space.shape == (3, 5, 6)

    obs, info = env.reset(seed=42)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (3, 5, 6)

    next_obs, reward, terminated, truncated, step_info = env.step(1)
    assert isinstance(next_obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
