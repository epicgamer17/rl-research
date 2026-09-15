from unittest.mock import MagicMock
import numpy as np
import pytest

from atomic_rl.envs.mdp.armed_bandits import ArmedBanditsEnv
from atomic_rl.envs.streams.nonstationary_armed_bandits import NonstationaryArmedBanditsEnv


def test_armed_bandits_direct_construction():
    """Verify ArmedBanditsEnv can be instantiated directly without AttributeError on self.spec."""
    means = [0.1, 0.5, 0.9]
    std_devs = [0.1, 0.1, 0.1]
    steps = 100

    env = ArmedBanditsEnv(means=means, std_devs=std_devs, steps=steps)
    assert env.spec is None
    assert env.reward_threshold == pytest.approx(0.9 * 100)

    obs, info = env.reset(seed=42)
    assert obs == 0
    assert info == {}

    next_obs, reward, terminated, truncated, info = env.step(1)
    assert next_obs == 0
    assert isinstance(reward, (float, np.floating))
    assert not terminated
    assert not truncated


def test_armed_bandits_with_spec():
    """Verify ArmedBanditsEnv sets spec.reward_threshold if self.spec is not None."""
    mock_spec = MagicMock()
    means = [0.2, 0.8]
    std_devs = [0.1, 0.1]

    # Subclass or attach spec to test if spec.reward_threshold is updated
    class CustomBanditsEnv(ArmedBanditsEnv):
        def __init__(self, spec=None, **kwargs):
            self.spec = spec
            super().__init__(**kwargs)

    env = CustomBanditsEnv(spec=mock_spec, means=means, std_devs=std_devs, steps=50)
    assert env.spec.reward_threshold == pytest.approx(0.8 * 50)


def test_nonstationary_armed_bandits_direct_construction():
    """Verify NonstationaryArmedBanditsEnv can be instantiated directly without AttributeError."""
    means = [0.1, 0.5]
    std_devs = [0.1, 0.1]
    target_means = [0.5, 0.9]
    steps = 10

    env = NonstationaryArmedBanditsEnv(
        means=np.array(means, dtype=np.float64),
        std_devs=std_devs,
        steps=steps,
        mode="linear",
        target_means=np.array(target_means, dtype=np.float64),
    )
    assert env.spec is None
    assert env.reward_threshold == pytest.approx(0.5 * 10)

    obs, info = env.reset(seed=42)
    assert obs == 0

    for step_idx in range(steps):
        _, _, terminated, truncated, _ = env.step(0)
        assert not terminated
        if step_idx == steps - 1:
            assert truncated
        else:
            assert not truncated
