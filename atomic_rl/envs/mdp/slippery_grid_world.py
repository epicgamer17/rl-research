from typing import List, Optional, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SlipperyGridWorldEnv(gym.Env):
    """
    Gymnasium environment for a 4x4 Slippery GridWorld (Single Player).

    Layout:
        S . . .       S: Start (0, 0)
        . H . H       .: Frozen safe ground
        . . . H       H: Hole / Trap (Reward = -1.0, Terminal)
        H . . G       G: Goal (3, 3) (Reward = +1.0, Terminal)

    Actions:
        0: UP
        1: RIGHT
        2: DOWN
        3: LEFT

    Stochasticity (Slipping):
        With probability 1 - slip_prob (e.g. 0.70): moves in intended direction.
        With probability slip_prob / 2 (e.g. 0.15): slips 90 deg clockwise (CW).
        With probability slip_prob / 2 (e.g. 0.15): slips 90 deg counter-clockwise (CCW).
        Moving into a boundary wall bounces and stays in place.

    Rewards:
        Reaching Goal (G): +1.0 (Terminal)
        Falling into Hole (H): -1.0 (Terminal)
        Living step penalty: -0.01 (to encourage shortest path)
        Timeout (max_steps): -1.0 (Terminal)

    Observation:
        One-hot float32 vector of length 16 representing current cell (row * 4 + col).
    """

    metadata = {"render_modes": ["human"]}

    DEFAULT_MAP = [
        "S...",
        ".H.H",
        "...H",
        "H..G",
    ]

    ACTIONS = {
        0: (-1, 0),  # UP
        1: (0, 1),  # RIGHT
        2: (1, 0),  # DOWN
        3: (0, -1),  # LEFT
    }

    def __init__(
        self,
        grid_map: Optional[List[str]] = None,
        slip_prob: float = 0.30,
        max_steps: int = 30,
    ):
        super().__init__()
        self.map = grid_map if grid_map is not None else self.DEFAULT_MAP
        self.height = len(self.map)
        self.width = len(self.map[0])
        self.slip_prob = slip_prob
        self.max_steps = max_steps

        self.num_states = self.height * self.width
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(self.num_states,), dtype=np.float32
        )

        self.row = 0
        self.col = 0
        self.step_count = 0

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(self.num_states, dtype=np.float32)
        obs[self.row * self.width + self.col] = 1.0
        return obs

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.row = 0
        self.col = 0
        self.step_count = 0

        mask = np.ones(4, dtype=bool)
        info = {
            "legal_moves": [0, 1, 2, 3],
            "action_mask": mask,
            "chance": 0,
            "row": self.row,
            "col": self.col,
            "step_count": self.step_count,
        }
        return self._get_obs(), info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        self.step_count += 1
        terminated = False
        truncated = False

        # Determine actual movement direction stochastically
        roll = self.np_random.random()
        intended_prob = 1.0 - self.slip_prob
        slip_half = self.slip_prob / 2.0

        if roll < intended_prob:
            actual_action = action
            chance_code = 0  # Intended
        elif roll < intended_prob + slip_half:
            actual_action = (action + 1) % 4  # 90 deg CW
            chance_code = 1  # Slip CW
        else:
            actual_action = (action - 1) % 4  # 90 deg CCW
            chance_code = 2  # Slip CCW

        dr, dc = self.ACTIONS[actual_action]
        new_row = min(max(0, self.row + dr), self.height - 1)
        new_col = min(max(0, self.col + dc), self.width - 1)

        self.row = new_row
        self.col = new_col

        cell = self.map[self.row][self.col]

        if cell == "G":
            reward = 1.0
            terminated = True
        elif cell == "H":
            reward = -1.0
            terminated = True
        else:
            reward = -0.01  # Small living step penalty
            if self.step_count >= self.max_steps:
                truncated = True
                reward = -1.0

        mask = np.ones(4, dtype=bool)
        info = {
            "legal_moves": [0, 1, 2, 3],
            "action_mask": mask,
            "chance": chance_code,
            "row": self.row,
            "col": self.col,
            "cell": cell,
            "step_count": self.step_count,
        }
        return self._get_obs(), reward, terminated, truncated, info

