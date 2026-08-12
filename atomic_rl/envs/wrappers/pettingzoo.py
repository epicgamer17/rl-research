"""PettingZoo environment wrappers."""

from typing import Any, Dict, Tuple
import numpy as np
import gymnasium.spaces
from pettingzoo.utils.wrappers.base import BaseWrapper


class PettingZooChannelFirstWrapper(BaseWrapper):
    """
    PettingZoo AEC Wrapper that transposes 3D observation images/grids
    from Channel-Last (H, W, C) to Channel-First (C, H, W).

    Works on Dict observation spaces containing an 'observation' key,
    or direct Box observation spaces.
    """

    def observation_space(self, agent: Any) -> gymnasium.spaces.Space:
        orig_space = self.env.observation_space(agent)
        if isinstance(orig_space, gymnasium.spaces.Box) and len(orig_space.shape) == 3:
            h, w, c = orig_space.shape
            return gymnasium.spaces.Box(
                low=orig_space.low.min(),
                high=orig_space.high.max(),
                shape=(c, h, w),
                dtype=orig_space.dtype,
            )
        elif (
            isinstance(orig_space, gymnasium.spaces.Dict)
            and "observation" in orig_space.spaces
        ):
            obs_space = orig_space["observation"]
            if isinstance(obs_space, gymnasium.spaces.Box) and len(obs_space.shape) == 3:
                h, w, c = obs_space.shape
                new_dict = dict(orig_space.spaces)
                new_dict["observation"] = gymnasium.spaces.Box(
                    low=obs_space.low.min(),
                    high=obs_space.high.max(),
                    shape=(c, h, w),
                    dtype=obs_space.dtype,
                )
                return gymnasium.spaces.Dict(new_dict)
        return orig_space

    def _transform_obs(self, obs: Any) -> Any:
        if obs is None:
            return None
        if isinstance(obs, dict) and "observation" in obs:
            raw_obs = obs["observation"]
            if isinstance(raw_obs, np.ndarray) and raw_obs.ndim == 3:
                new_obs = dict(obs)
                new_obs["observation"] = np.transpose(raw_obs, (2, 0, 1))
                return new_obs
        elif isinstance(obs, np.ndarray) and obs.ndim == 3:
            return np.transpose(obs, (2, 0, 1))
        return obs

    def observe(self, agent: Any) -> Any:
        obs = self.env.observe(agent)
        return self._transform_obs(obs)

    def last(self, observe: bool = True) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        obs, reward, term, trunc, info = self.env.last(observe=observe)
        if observe:
            obs = self._transform_obs(obs)
        return obs, reward, term, trunc, info
