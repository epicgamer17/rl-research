import environments.environments_.envs as gym
from environments.environments_.envs import spaces
import numpy as np


class TicTacToe(gym.Env):
    metadata = {}

    def __init__(self):
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(3, 3), dtype=np.int)
        self.grid = np.zeros((3, 3))
        self.done = False
        self.players = [-1, 1]
        self.turn = 0

    def reset(self):
        self.grid = np.zeros((3, 3))
        self.done = False
        self.turn = 0
        return self.grid, self.turn

    def step(self, action):
        if action < 0 or action > 8:
            raise ValueError("Action must be between 0 and 8")
        self.grid[action // 3][action % 3] = self.players[self.turn % 2]
        self.winner()
        reward = 1 if self.done else 0
        self.turn += 1
        info = self._get_info()
        return (
            self.grid,
            reward,
            self.done,
            False,
            info,
        )

    def winner(self):
        for i in range(3):
            if (self.grid[i][0] == self.grid[i][1] == self.grid[i][2]) and (
                self.grid[i][0] != 0
            ):
                self.done = True
            if (self.grid[0][i] == self.grid[1][i] == self.grid[2][i]) and (
                self.grid[0][i] != 0
            ):
                self.done = True
            if (self.grid[0][0] == self.grid[1][1] == self.grid[2][2]) and (
                self.grid[0][0] != 0
            ):
                self.done = True
            if (self.grid[0][2] == self.grid[1][1] == self.grid[2][0]) and (
                self.grid[0][2] != 0
            ):
                self.done = True
        return self.done

    def _get_info(self):
        dico = {}
        dico["turn"] = self.turn
        l = []
        for i in range(9):
            if self.grid[i // 3][i % 3] == 0:
                l.append(i)
        dico["possible_actions"] = l
        return dico
