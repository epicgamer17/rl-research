import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import copy
import rlcard


class LeducHoldemEnv(gym.Env):
    metadata = {"render_modes": [], "render_fps": 1}

    def __init__(self, render_mode=None):
        self.game = rlcard.make("leduc-holdem")
        self.observation_space = spaces.Box(
            low=-10, high=10, shape=(36,), dtype=np.int8
        )

        # We have 9 actions, corresponding to each cell
        self.action_space = spaces.Discrete(4)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    # def _get_obs(self):
    #     return self.game.get_state()

    def _get_info(self):
        return {"legal_moves": self._legal_moves, "player": self._player}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        dict, self._player = self.game.reset()
        self._legal_moves = dict["legal_actions"]
        observation = dict["obs"]
        move_history = dict["action_record"]

        info = self._get_info()

        # if self.render_mode == "human":
        #     self._render_frame()

        return observation, info

    def step(self, action):
        if action < 0 or action > self.action_space.n - 1:
            raise ValueError(
                "Action must be between 0 and {}".format(self.action_space.n - 1)
            )
        # if action not in self._legal_moves:
        # raise ValueError(
        #     "Illegal move {} Legal Moves {}".format(action, self._legal_moves)
        # )

        dict, self._player = self.game.step(action)
        self._legal_moves = dict["legal_actions"]
        observation = dict["obs"]
        move_history = dict["action_record"]

        terminated = self.game.is_over()
        reward = self.game.get_payoffs() if terminated else 0
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info
