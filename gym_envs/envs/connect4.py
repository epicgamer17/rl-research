import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import copy


class Connect4Env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}

    def __init__(self, render_mode=None, size=(6, 7), win_length=4):
        self.size = size  # The size of the square grid
        self.win_length = win_length  # The number of consecutive tokens needed to win
        self.window_size = (
            size[1] * (512 / size[0]),
            512,
        )  # The size of the PyGame window

        # Observations are planes.
        # The first plane represents player 1s tokens, the second player 2s and the third encodes the current players turn.
        self.observation_space = spaces.Box(
            low=0, high=1, shape=self.size + (3,), dtype=np.float64
        )
        print(self.observation_space)

        # We have 9 actions, corresponding to each cell
        self.action_space = spaces.Discrete(self.size[1])

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

    def _get_obs(self):
        return copy.deepcopy(self._grid)

    def _get_info(self):
        return {"legal_moves": self._legal_moves}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Set a blank board
        self._grid = np.zeros(self.size + (3,))
        self._grid[:, :, 2] = 0  # It's player 1's turn

        # Reset legal moves
        self._legal_moves = np.array(list(range(self.action_space.n)))

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        if action < 0 or action > 8:
            raise ValueError("Action must be between 0 and 8")
        if action not in self._legal_moves:
            # could return a negative reward
            raise ValueError(
                "Illegal move {} Legal Moves {}".format(action, self._legal_moves)
            )
        # output next player's token first (since that's the one we're inputting to)
        current_player_board = copy.deepcopy(self._grid[:, :, 0])
        self._grid[:, :, 0] = self._grid[:, :, 1]
        self._grid[:, :, 1] = current_player_board
        # place token
        for i in range(self.size[0] - 1, -1, -1):
            if self._grid[i, action, 0] == 0 and self._grid[i, action, 1] == 0:
                self._grid[i, action, 1] = 1
                break
        if i == 0:
            self._legal_moves = np.delete(
                self._legal_moves, np.where(self._legal_moves == action)
            )

        # encode turn as 1 or 0
        self._grid[:, :, 2] = 1 - self._grid[:, :, 2]
        # An episode is done iff there is a winner or the board is full
        terminated = self.winner()
        truncated = len(self._legal_moves) == 0
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface(self.window_size)
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size[0] / self.size[1]
        )  # The size of a single grid square in pixels

        # First we draw the X's and 0's
        turn = int(self._grid[0, 0, 2])
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                if self._grid[i, j, turn] == 1:
                    pygame.draw.circle(
                        canvas,
                        (255, 0, 0),
                        (
                            j * pix_square_size + pix_square_size // 2,
                            i * pix_square_size + pix_square_size // 2,
                        ),
                        pix_square_size // 3,
                        width=3,
                    )
                if self._grid[i, j, 1 - turn] == 1:
                    pygame.draw.circle(
                        canvas,
                        (255, 255, 0),
                        (
                            j * pix_square_size + pix_square_size // 2,
                            i * pix_square_size + pix_square_size // 2,
                        ),
                        pix_square_size // 3,
                        width=3,
                    )

        # Finally, add some gridlines
        for i in range(self.size[0]):
            pygame.draw.line(
                canvas,
                (0, 0, 0),
                (0, i * pix_square_size),
                (self.window_size[0], i * pix_square_size),
            )
        for j in range(self.size[1]):
            pygame.draw.line(
                canvas,
                (0, 0, 0),
                (j * pix_square_size, 0),
                (j * pix_square_size, self.window_size[1]),
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def winner(self):
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                if self._grid[i, j, 1] == 1:
                    if self._check_win(i, j):
                        return True
        return False

    def _check_win(self, i, j):
        # Check row
        if j + self.win_length <= self.size[1]:
            if np.all(self._grid[i, j : j + self.win_length, 1]):
                return True
        # Check column
        if i + self.win_length <= self.size[0]:
            if np.all(self._grid[i : i + self.win_length, j, 1]):
                return True
        # Check diagonal
        if i + self.win_length <= self.size[0] and j + self.win_length <= self.size[1]:
            if np.all([self._grid[i + k, j + k, 1] for k in range(self.win_length)]):
                return True
        # Check other diagonal
        if i + self.win_length <= self.size[0] and j - self.win_length + 1 >= 0:
            if np.all([self._grid[i + k, j - k, 1] for k in range(self.win_length)]):
                return True

        return False
