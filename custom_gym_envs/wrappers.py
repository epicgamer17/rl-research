from __future__ import annotations

import sys

sys.path.append("../../")


from typing import Any

import gymnasium.spaces
import numpy as np

from pettingzoo.utils.env import ActionType, AECEnv, AgentID, ObsType
from gymnasium.spaces import Box, Discrete

from supersuit.utils.frame_stack import stack_init, stack_obs, stack_obs_space
from collections import deque

from utils import action_mask_to_legal_moves
from pettingzoo.utils.wrappers.base import BaseWrapper


def action_mask_to_info(state, info, current_player):
    info["legal_moves"] = action_mask_to_legal_moves(state["action_mask"])

    info["player"] = current_player

    if "observation" in state:
        state = state["observation"]

    return state


class ActionMaskInInfoWrapper(BaseWrapper):
    """Wrapper to convert dict observations to processed Box observations."""

    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        orig_space = self.env.observation_space(agent)

        if isinstance(orig_space, gymnasium.spaces.Dict):
            obs_space = orig_space["observation"]
            shape = obs_space.shape
            return gymnasium.spaces.Box(
                low=np.min(obs_space.low),
                high=np.max(obs_space.high),
                shape=shape,
                dtype=obs_space.dtype,
            )
        else:
            return orig_space

    def reset(self, seed: int | None = None, options: dict | None = None):
        self.env.reset(seed=seed, options=options)
        for agent in self.env.agents:
            obs = self.env.observe(agent)
            self.env.infos[agent] = getattr(self.env, "infos", {}).get(agent, {})
            _ = action_mask_to_info(obs, self.env.infos[agent], agent)

    def observe(self, agent: AgentID) -> np.ndarray:
        obs = self.env.observe(agent)
        info = self.env.infos[agent]
        return action_mask_to_info(obs, info, agent)

    def step(self, action: ActionType):
        self.env.step(action)
        # Process observation and info for current agent
        agent = self.env.agent_selection
        obs = self.env.observe(agent)
        info = self.env.infos[agent]
        _ = action_mask_to_info(obs, info, agent)


class ChannelLastToFirstWrapper(BaseWrapper):
    """Wrapper to convert image observations from HWC to CHW."""

    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        orig_space = self.env.observation_space(agent)

        if isinstance(orig_space, gymnasium.spaces.Box) and len(orig_space.shape) == 3:
            # HWC -> CHW
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
            if len(obs_space.shape) == 3:
                h, w, c = obs_space.shape
                return gymnasium.spaces.Box(
                    low=obs_space.low.min(),
                    high=obs_space.high.max(),
                    shape=(c, h, w),
                    dtype=obs_space.dtype,
                )
        return orig_space

    def observe(self, agent: AgentID) -> np.ndarray:
        obs = self.env.observe(agent)

        if isinstance(obs, dict) and "observation" in obs:
            obs = obs["observation"]

        if isinstance(obs, np.ndarray) and obs.ndim == 3:  # image
            obs = np.transpose(obs, (2, 0, 1))  # HWC -> CHW

        return obs

    def step(self, action: ActionType) -> None:
        self.env.step(action)

    def reset(self, seed: int | None = None, options: dict | None = None):
        self.env.reset(seed=seed, options=options)


class TwoPlayerPlayerPlaneWrapper(BaseWrapper):
    """
    Wrapper to add a player plane to observations.

    Adds an extra channel where all entries are 0 if player 1, 1 if player 2 (or agent index > 0),
    compatible with both HWC (channel-last) and CHW (channel-first) observations.
    """

    def __init__(
        self, env: AECEnv[AgentID, ObsType, ActionType], channel_first: bool = True
    ):
        super().__init__(env)
        self.channel_first = channel_first

    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        orig_space = self.env.observation_space(agent)

        if isinstance(orig_space, gymnasium.spaces.Box):
            shape = orig_space.shape
            if len(shape) == 3:
                if self.channel_first:
                    c, h, w = shape if shape[0] != 3 else shape  # CHW
                    new_shape = (c + 1, h, w)
                else:
                    h, w, c = shape
                    new_shape = (h, w, c + 1)
                return gymnasium.spaces.Box(
                    low=0,
                    high=1,
                    shape=new_shape,
                    dtype=orig_space.dtype,
                )
        elif (
            isinstance(orig_space, gymnasium.spaces.Dict)
            and "observation" in orig_space.spaces
        ):
            obs_space = orig_space["observation"]
            shape = obs_space.shape
            if len(shape) == 3:
                if self.channel_first:
                    c, h, w = shape if shape[0] != 3 else shape
                    new_shape = (c + 1, h, w)
                else:
                    h, w, c = shape
                    new_shape = (h, w, c + 1)
                return gymnasium.spaces.Box(
                    low=0,
                    high=1,
                    shape=new_shape,
                    dtype=obs_space.dtype,
                )

        return orig_space

    def observe(self, agent: AgentID) -> np.ndarray:
        obs = self.env.observe(agent)

        if isinstance(obs, dict) and "observation" in obs:
            obs = obs["observation"]

        # Determine player plane value (0 for first agent, 1 for others)
        plane_val = 0 if agent == self.env.agents[0] else 1

        if isinstance(obs, np.ndarray) and obs.ndim == 3:
            if self.channel_first:
                # CHW
                h, w = obs.shape[1], obs.shape[2]
                plane = np.full((1, h, w), plane_val, dtype=obs.dtype)
                obs = np.concatenate([obs, plane], axis=0)
            else:
                # HWC
                h, w = obs.shape[0], obs.shape[1]
                plane = np.full((h, w, 1), plane_val, dtype=obs.dtype)
                obs = np.concatenate([obs, plane], axis=2)

        return obs

    def step(self, action: ActionType) -> None:
        self.env.step(action)

    def reset(self, seed: int | None = None, options: dict | None = None):
        self.env.reset(seed=seed, options=options)


class FrameStackWrapper(BaseWrapper):
    """
    Wrapper to stack the last k frames along the channel dimension.

    Args:
        env: The PettingZoo AECEnv to wrap.
        k: Number of frames to stack.
        channel_first: Whether the input is channel-first (CHW) or channel-last (HWC).
    """

    def __init__(
        self,
        env: AECEnv[AgentID, ObsType, ActionType],
        k: int = 4,
        channel_first: bool = True,
    ):
        super().__init__(env)
        self.k = k
        self.channel_first = channel_first
        self.stacks: dict[AgentID, deque] = {
            agent: deque(maxlen=k) for agent in self.env.possible_agents
        }

    def reset(self, seed: int | None = None, options: dict | None = None):
        self.env.reset(seed=seed, options=options)
        # Clear stacks
        for agent in self.env.agents:
            self.stacks[agent].clear()
            obs = self.env.observe(agent)
            if isinstance(obs, dict) and "observation" in obs:
                obs = obs["observation"]
            self._init_stack(agent, obs)

    def _init_stack(self, agent: AgentID, obs: np.ndarray):
        """Fill the deque with the initial frame k times."""
        for _ in range(self.k):
            self.stacks[agent].append(obs)

    def observe(self, agent: AgentID) -> np.ndarray:
        obs = self.env.observe(agent)
        if isinstance(obs, dict) and "observation" in obs:
            obs = obs["observation"]

        self.stacks[agent].append(obs)
        frames = list(self.stacks[agent])[::-1]  # newest first, oldest last

        if self.channel_first:
            # CHW: concatenate along channel axis
            return np.concatenate(frames, axis=0)
        else:
            # HWC: concatenate along last axis
            return np.concatenate(frames, axis=-1)

    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        orig_space = self.env.observation_space(agent)
        if (
            isinstance(orig_space, gymnasium.spaces.Dict)
            and "observation" in orig_space.spaces
        ):
            obs_space = orig_space["observation"]
        else:
            obs_space = orig_space

        shape = obs_space.shape
        if len(shape) != 3:
            # Only stack 3D frames
            return obs_space

        if self.channel_first:
            c, h, w = shape
            new_shape = (c * self.k, h, w)
        else:
            h, w, c = shape
            new_shape = (h, w, c * self.k)

        return gymnasium.spaces.Box(
            low=np.min(obs_space.low),
            high=np.max(obs_space.high),
            shape=new_shape,
            dtype=obs_space.dtype,
        )

    def step(self, action: ActionType) -> None:
        self.env.step(action)
