"""PettingZoo environment wrappers."""

import collections
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


class FrameStackWrapper(BaseWrapper):
    """
    PettingZoo AEC Wrapper that stacks K historical observation frames along the channel axis.
    """

    def __init__(self, env: Any, k: int = 4, channel_first: bool = True):
        super().__init__(env)
        self.k = k
        self.channel_first = channel_first
        self.stacks: Dict[Any, collections.deque] = {}

    def reset(self, seed: Any = None, options: Any = None):
        super().reset(seed=seed, options=options)
        self.stacks = {
            agent: collections.deque(maxlen=self.k)
            for agent in getattr(self.env, "possible_agents", [])
        }
        for agent in getattr(self.env, "possible_agents", []):
            obs = self.env.observe(agent)
            grid = obs["observation"] if isinstance(obs, dict) and "observation" in obs else obs
            if isinstance(grid, np.ndarray):
                for _ in range(self.k):
                    self.stacks[agent].append(grid.copy())

    def observation_space(self, agent: Any) -> gymnasium.spaces.Space:
        orig_space = self.env.observation_space(agent)
        is_dict = (
            isinstance(orig_space, gymnasium.spaces.Dict)
            and "observation" in orig_space.spaces
        )
        obs_space = orig_space["observation"] if is_dict else orig_space
        if isinstance(obs_space, gymnasium.spaces.Box) and len(obs_space.shape) == 3:
            if self.channel_first:
                c, h, w = obs_space.shape
                new_shape = (c * self.k, h, w)
            else:
                h, w, c = obs_space.shape
                new_shape = (h, w, c * self.k)
            new_box = gymnasium.spaces.Box(
                low=obs_space.low.min(),
                high=obs_space.high.max(),
                shape=new_shape,
                dtype=obs_space.dtype,
            )
            if is_dict:
                new_dict = dict(orig_space.spaces)
                new_dict["observation"] = new_box
                return gymnasium.spaces.Dict(new_dict)
            return new_box
        return orig_space

    def _transform_obs(self, agent: Any, obs: Any) -> Any:
        if obs is None:
            return None
        is_dict = isinstance(obs, dict) and "observation" in obs
        grid = obs["observation"] if is_dict else obs

        if isinstance(grid, np.ndarray):
            if agent not in self.stacks:
                self.stacks[agent] = collections.deque(maxlen=self.k)
                for _ in range(self.k):
                    self.stacks[agent].append(grid.copy())
            else:
                self.stacks[agent].append(grid.copy())

            frames = list(reversed(self.stacks[agent]))
            stacked_grid = np.concatenate(frames, axis=0 if self.channel_first else -1)

            if is_dict:
                new_obs = dict(obs)
                new_obs["observation"] = stacked_grid
                return new_obs
            return stacked_grid
        return obs

    def observe(self, agent: Any) -> Any:
        obs = self.env.observe(agent)
        return self._transform_obs(agent, obs)

    def last(self, observe: bool = True) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        obs, reward, term, trunc, info = self.env.last(observe=observe)
        if observe and obs is not None:
            agent = getattr(self.env, "agent_selection", None)
            obs = self._transform_obs(agent, obs)
        return obs, reward, term, trunc, info


class TwoPlayerPlayerPlaneWrapper(BaseWrapper):
    """
    PettingZoo AEC Wrapper that appends a 1-channel player plane (color plane).
    Fills plane with 1.0 for Player 0 (first agent in possible_agents) and 0.0 for Player 1.
    """

    def __init__(self, env: Any, channel_first: bool = True):
        super().__init__(env)
        self.channel_first = channel_first

    def observation_space(self, agent: Any) -> gymnasium.spaces.Space:
        orig_space = self.env.observation_space(agent)
        is_dict = (
            isinstance(orig_space, gymnasium.spaces.Dict)
            and "observation" in orig_space.spaces
        )
        obs_space = orig_space["observation"] if is_dict else orig_space
        if isinstance(obs_space, gymnasium.spaces.Box) and len(obs_space.shape) == 3:
            if self.channel_first:
                c, h, w = obs_space.shape
                new_shape = (c + 1, h, w)
            else:
                h, w, c = obs_space.shape
                new_shape = (h, w, c + 1)
            new_box = gymnasium.spaces.Box(
                low=obs_space.low.min(),
                high=obs_space.high.max(),
                shape=new_shape,
                dtype=obs_space.dtype,
            )
            if is_dict:
                new_dict = dict(orig_space.spaces)
                new_dict["observation"] = new_box
                return gymnasium.spaces.Dict(new_dict)
            return new_box
        return orig_space

    def _transform_obs(self, agent: Any, obs: Any) -> Any:
        if obs is None:
            return None
        is_dict = isinstance(obs, dict) and "observation" in obs
        grid = obs["observation"] if is_dict else obs

        if isinstance(grid, np.ndarray) and grid.ndim == 3:
            possible = getattr(self.env, "possible_agents", [])
            agent_idx = possible.index(agent) if agent in possible else 0
            plane_val = 1.0 if agent_idx == 0 else 0.0

            if self.channel_first:
                h, w = grid.shape[1], grid.shape[2]
                plane = np.full((1, h, w), plane_val, dtype=grid.dtype)
                new_grid = np.concatenate([grid, plane], axis=0)
            else:
                h, w = grid.shape[0], grid.shape[1]
                plane = np.full((h, w, 1), plane_val, dtype=grid.dtype)
                new_grid = np.concatenate([grid, plane], axis=-1)

            if is_dict:
                new_obs = dict(obs)
                new_obs["observation"] = new_grid
                return new_obs
            return new_grid
        return obs

    def observe(self, agent: Any) -> Any:
        obs = self.env.observe(agent)
        return self._transform_obs(agent, obs)

    def last(self, observe: bool = True) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        obs, reward, term, trunc, info = self.env.last(observe=observe)
        if observe and obs is not None:
            agent = getattr(self.env, "agent_selection", None)
            obs = self._transform_obs(agent, obs)
        return obs, reward, term, trunc, info

