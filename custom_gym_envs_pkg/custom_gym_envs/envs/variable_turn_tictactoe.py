import os

import gymnasium
import numpy as np
import pygame
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import agent_selector


def get_image(path):
    from os import path as os_path

    cwd = os_path.dirname(__file__)
    image_path = os_path.join(cwd, "variable_turn_tictactoe_img", path)
    if not os.path.exists(image_path):
         # Fallback to local img for ease of testing or if just simple filenames provided
         image_path = path
    image = pygame.image.load(image_path)
    return image


def env(**kwargs):
    env = raw_env(**kwargs)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class raw_env(AECEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "variable_turn_tictactoe_v0",
        "is_parallelizable": False,
        "render_fps": 1,
    }

    def __init__(self, render_mode=None, size=3, win_length=3, min_moves=2, max_moves=3, screen_height=1000):
        super().__init__()
        self.size = size
        self.win_length = win_length
        self.min_moves = min_moves
        self.max_moves = max_moves
        self.render_mode = render_mode
        self.screen_height = screen_height
        self.screen = None

        if self.render_mode == "human":
            self.clock = pygame.time.Clock()

        self.agents = ["player_0", "player_1"]
        self.possible_agents = self.agents[:]

        self.action_spaces = {i: spaces.Discrete(size * size) for i in self.agents}
        self.observation_spaces = {
            i: spaces.Dict(
                {
                    "observation": spaces.Box(
                        low=0, high=1, shape=(self.size, self.size, 2), dtype=np.int8
                    ),
                    "action_mask": spaces.Box(
                        low=0, high=1, shape=(self.size * self.size,), dtype=np.int8
                    ),
                }
            )
            for i in self.agents
        }

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _get_obs(self, agent):
        # Observation:
        # Channel 0: where 'agent' has tokens
        # Channel 1: where 'opponent' has tokens
        
        agent_idx = self.agents.index(agent)
        opponent_idx = 1 - agent_idx
        
        # board values: 0 = empty, 1 = player_1, 2 = player_2
        # (internally I'll use 0=empty, 1=p1, 2=p2 for convenience, or 0/1/2 logic)
        
        # Let's align with the board representation
        # self.board: 0=empty, 1=player_1, 2=player_2
        
        current_player_token = agent_idx + 1
        opponent_token = opponent_idx + 1
        
        obs = np.zeros((self.size, self.size, 2), dtype=np.int8)
        obs[:, :, 0] = (self.board == current_player_token).astype(np.int8)
        obs[:, :, 1] = (self.board == opponent_token).astype(np.int8)
        
        return obs

    def observe(self, agent):
        legal_moves = self._legal_moves() if agent == self.agent_selection else []
        action_mask = np.zeros(self.size * self.size, dtype=np.int8)
        
        # If the game is over, legal moves might be empty or we might want to mask everything
        # In PZ, if agent is dead/done, observe returns empty mask usually?
        # But here observe uses the current state.
        
        if agent == self.agent_selection:
             for move in legal_moves:
                action_mask[move] = 1

        return {"observation": self._get_obs(agent), "action_mask": action_mask}

    def _legal_moves(self):
        return [i for i, x in enumerate(self.board.flatten()) if x == 0]

    def reset(self, seed=None, options=None):
        self.np_random, seed = gymnasium.utils.seeding.np_random(seed)
        
        self.agents = self.possible_agents[:]
        self.rewards = {i: 0 for i in self.agents}
        self._cumulative_rewards = {i: 0 for i in self.agents}
        self.terminations = {i: False for i in self.agents}
        self.truncations = {i: False for i in self.agents}
        self.infos = {i: {} for i in self.agents}

        self.board = np.zeros((self.size, self.size), dtype=np.int8)
        
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        
        # Determine initial moves for the first player
        self.moves_left = self.np_random.integers(self.min_moves, self.max_moves + 1)

        if self.render_mode == "human":
            self.screen = pygame.display.set_mode(
                (self.screen_height, self.screen_height)
            )
            pygame.display.set_caption("Variable Turn Tic-Tac-Toe")
        else:
            self.screen = pygame.Surface((self.screen_height, self.screen_height))

        
    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return

        # Check legality
        flat_board = self.board.flatten()
        if flat_board[action] != 0:
             # Illegal move logic: usually undefined behavior or mapping to random legal move
             # But here we assume wrapper or valid agent.
             # If we want to be strict, we can raise error or give negative reward.
             # For now, assume valid action or simple ignore/error.
             # In PZ, it is "undefined" but typically we expect legal actions.
             pass

        # Apply move
        agent_idx = self.agents.index(self.agent_selection)
        token = agent_idx + 1
        
        row, col = divmod(action, self.size)
        self.board[row, col] = token
        
        self.moves_left -= 1
        
        # Check win
        if self._check_win(token):
            self.rewards[self.agent_selection] = 1
            self.rewards[self.agents[1 - agent_idx]] = -1
            self.terminations = {i: True for i in self.agents}
            self._cumulative_rewards[self.agent_selection] = 0
            self._cumulative_rewards[self.agents[1 - agent_idx]] = 0
        elif np.all(self.board != 0):
            # Draw
            self.rewards = {i: 0 for i in self.agents}
            self.terminations = {i: True for i in self.agents}
        else:
            # Game continues
            # Check turn switch
            if self.moves_left <= 0:
                self.agent_selection = self._agent_selector.next()
                self.moves_left = self.np_random.integers(self.min_moves, self.max_moves + 1)
            else:
                # Same agent continues
                pass
                
        self._accumulate_rewards()
        
        if self.render_mode == "human":
            self.render()

    def _check_win(self, token):
        # Rows
        for r in range(self.size):
            if np.all(self.board[r, :] == token):
                return True
        # Cols
        for c in range(self.size):
            if np.all(self.board[:, c] == token):
                return True
        # Diagonals
        if np.all(np.diag(self.board) == token):
            return True
        if np.all(np.diag(np.fliplr(self.board)) == token):
            return True
        return False

    def render(self):
        if self.render_mode is None:
            return

        screen_height = self.screen_height
        screen_width = self.screen_height

        # Setup dimensions for 'x' and 'o' marks
        # Assuming 3x3 layout like standard TTT for simplicity, or scale by self.size?
        # Standard TTT is 3x3. self.size is variable.
        # But PZ assets are likely designed for 3x3.
        # However, let's try to adapt tile_size.
        tile_size = int(screen_height / (self.size + 1)) # heuristic

        # Load and blit the board image for the game
        board_img = get_image("board.png")
        board_img = pygame.transform.scale(
            board_img, (int(screen_width), int(screen_height))
        )

        self.screen.blit(board_img, (0, 0))

        # Helper to get symbol name
        # self.board: 0=Empty, 1=P0(cross?), 2=P1(circle?)
        # PZ TTT: Player 1 = X (cross), Player 2 = O (circle)
        # In my invalid mapping: agents = [player_0, player_1]
        # player_0 idx=0 -> token=1 -> Cross
        # player_1 idx=1 -> token=2 -> Circle
        
        def getSymbol(token):
            if token == 0:
                return None
            elif token == 1:
                return "cross"
            else:
                return "circle"

        # Draw tokens
        # board is shape (size, size)
        # To match PZ TTT visual layout (which is 0-8 in 3x3 grid):
        # 0 | 1 | 2
        # 3 | 4 | 5
        # 6 | 7 | 8
        # Wait, PZ TTT layout is:
        # 0 | 3 | 6
        # 1 | 4 | 7
        # 2 | 5 | 8
        # Column major?
        
        # My board is usually row-major in printed array: board[row, col].
        # If I iterate rows and cols:
        
        for r in range(self.size):
            for c in range(self.size):
                 token = self.board[r, c]
                 mark = getSymbol(token)
                 
                 if mark is None:
                     continue
                 
                 mark_img = get_image(mark + ".png")
                 mark_img = pygame.transform.scale(mark_img, (tile_size, tile_size))
                 
                 # Calculate position
                 # PZ Logic:
                 # (screen_width / 3.1) * x + (screen_width / 17)
                 # x is col index? y is row index?
                 # PZ loop: for x in 3: for y in 3: mark = board[pos].
                 # PZ board is 0-8 list.
                 # Their loop x, y implies x is row or col?
                 # Based on get_image calls:
                 # x * width stuff.
                 # If x is 0, 1, 2.
                 # If x is column index, it shifts horizontally.
                 
                 # Let's assume standard grid mapping:
                 # Grid width is screen_width.
                 # Cell width is screen_width / size.
                 # Center the mark in the cell.
                 
                 # Using PZ's specific constants for 3x3 to match "Exact Look" if size is 3.
                 if self.size == 3:
                      # Mapping r, c to x, y used in PZ
                      # PZ: x=0, y=0 -> Pos 0 (Top Left)
                      # PZ: x=1, y=0 -> Pos 3 (Top Middle) -> wait, 3 is Top Middle?
                      # PZ Docs:
                      # 0 | 3 | 6
                      # 1 | 4 | 7
                      # 2 | 5 | 8
                      
                      # My board:
                      # (0,0) (0,1) (0,2)
                      # (1,0) (1,1) (1,2)
                      
                      # If I want visual match:
                      # (0,0) -> Visual Top Left.
                      # PZ Loop: x goes 0..2, y goes 0..2.
                      # mark_pos increments.
                      # pos=0 -> x=0, y=0.
                      # pos=1 -> x=0, y=1.
                      # pos=2 -> x=0, y=2.
                      # pos=3 -> x=1, y=0.
                      
                      # So PZ iterates x (cols) then y (rows)? No.
                      # x=0, y=0 -> pos 0.
                      # x=0, y=1 -> pos 1.
                      # ...
                      # x=1, y=0 -> pos 3.
                      
                      # So PZ `board_state` is layout:
                      # Col 0 (0,1,2), Col 1 (3,4,5), Col 2 (6,7,8).
                      
                      # If my board[r, c] is standard matrix (row r, col c).
                      # (0,0) is Top Left.
                      # In PZ terms: x=0 implies Left Col?
                      # Let's see the coord math:
                      # (screen_width / 3.1) * x + ...
                      # If x=0 -> Left.
                      # If x=1 -> Middle.
                      # So x is COLUMN index.
                      #
                      # (screen_width / 3.145) * y + ...
                      # If y=0 -> Top.
                      # So y is ROW index.
                      
                      # So PZ iterates x (Col) from 0..2.
                      #   Nested y (Row) from 0..2.
                      #     mark_pos 0 -> x=0, y=0 -> Col 0, Row 0. (Top Left). Correct.
                      #     mark_pos 1 -> x=0, y=1 -> Col 0, Row 1. (Mid Left). Correct.
                      #     mark_pos 2 -> x=0, y=2 -> Col 0, Row 2. (Bot Left). Correct.
                      
                      # So PZ layout is Column-Major linear list.
                      
                      # My board is [row, col].
                      # So to match, I use c as x, r as y.
                      
                      x = c
                      y = r
                      
                      pos_x = (screen_width / 3.1) * x + (screen_width / 17)
                      pos_y = (screen_height / 3.145) * y + (screen_height / 19)
                      
                      self.screen.blit(mark_img, (pos_x, pos_y))
                 else:
                      # General fallback for size != 3 (if ever used)
                      cell_w = screen_width / self.size
                      cell_h = screen_height / self.size
                      
                      # Center
                      pos_x = c * cell_w + (cell_w - tile_size) / 2
                      pos_y = r * cell_h + (cell_h - tile_size) / 2
                      self.screen.blit(mark_img, (pos_x, pos_y))

        if self.render_mode == "human":
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])

        observation = np.array(pygame.surfarray.pixels3d(self.screen))

        return (
            np.transpose(observation, axes=(1, 0, 2))
            if self.render_mode == "rgb_array"
            else None
        )

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
