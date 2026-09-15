import pytest
import numpy as np
import gymnasium as gym
from atomic_rl.envs.custom.envs.matching_pennies import (
    MatchingPenniesEnv,
    MatchingPenniesGymEnv,
)


def test_matching_pennies_aec_env_api():
    env = MatchingPenniesEnv(max_cycles=10)
    env.reset(seed=42)
    assert len(env.agents) == 2

    agent = env.agent_selection
    obs = env.observe(agent)
    assert obs.shape == (6,)

    env.step(0)
    agent2 = env.agent_selection
    obs2 = env.observe(agent2)
    assert obs2.shape == (6,)


def test_matching_pennies_gym_env_api():
    env = MatchingPenniesGymEnv(max_rounds=5)
    assert isinstance(env, gym.Env)
    assert env.observation_space.shape == (6,)
    assert env.action_space.n == 2

    obs, info = env.reset(seed=42)
    assert obs.shape == (6,)
    assert isinstance(info, dict)

    next_obs, reward, terminated, truncated, step_info = env.step(0)
    assert next_obs.shape == (6,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
