import gym
from gymnasium import spaces
import pygame
import numpy as np
from gym.envs.registration import register

class TicTacToe(gym.Env):
    metadata = {}

    def __init__(self):
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(3, 3), dtype=np.int)
        self.grid = np.zeros((3, 3))
        self.done = False
        self.colors = [-1, 1]
        self.turn = 0

    def reset(self):
        self.grid = np.zeros((3, 3))
        self.done = False
        self.turn = 0
        return self.grid, self.turn
    
    def step(self, action):
        if action <0 or action > 8:
            raise ValueError("Action must be between 0 and 8")
        self.grid[action//3][action%3] = self.player[self.turn%2]
        self.winner()
        reward = 1 if self.done else 0
        self.turn += 1
        return self.grid, reward, self.done, False, self.turn 
    

    def winner(self):
        for i in range(3):
            if (self.grid[i][0] == self.grid[i][1] == self.grid[i][2]) and (self.grid[i][0]!= 0):
                self.done = True
            if (self.grid[0][i] == self.grid[1][i] == self.grid[2][i]) and (self.grid[0][i]!= 0):
                self.done = True
        return self.done


register(
    id='TicTacToe-v0',
    entry_point='environments.tic_tac_toe:TicTacToe',
    max_episode_steps=9,
)   
