import functools
from typing import TypedDict, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import agent_selector

# --- Catanatron Game Logic and Constants (from original code) ---
from catanatron.game import Game, TURNS_LIMIT
from catanatron.models.player import Color, Player
from catanatron.models.map import BASE_MAP_TEMPLATE, NUM_NODES, LandTile, build_map
from catanatron.models.enums import RESOURCES, Action, ActionType
from catanatron.models.board import get_edges
from catanatron.features import (
    create_sample,
    get_feature_ordering,
)
from catanatron.gym.board_tensor_features import (
    create_board_tensor,
    get_channels,
    is_graph_feature,
)

BASE_TOPOLOGY = BASE_MAP_TEMPLATE.topology
TILE_COORDINATES = [x for x, y in BASE_TOPOLOGY.items() if y == LandTile]
ACTIONS_ARRAY = [
    (ActionType.ROLL, None),
    *[(ActionType.MOVE_ROBBER, tile) for tile in TILE_COORDINATES],
    (ActionType.DISCARD, None),
    *[(ActionType.BUILD_ROAD, tuple(sorted(edge))) for edge in get_edges()],
    *[(ActionType.BUILD_SETTLEMENT, node_id) for node_id in range(NUM_NODES)],
    *[(ActionType.BUILD_CITY, node_id) for node_id in range(NUM_NODES)],
    (ActionType.BUY_DEVELOPMENT_CARD, None),
    (ActionType.PLAY_KNIGHT_CARD, None),
    *[
        (ActionType.PLAY_YEAR_OF_PLENTY, (first_card, RESOURCES[j]))
        for i, first_card in enumerate(RESOURCES)
        for j in range(i, len(RESOURCES))
    ],
    *[(ActionType.PLAY_YEAR_OF_PLENTY, (first_card,)) for first_card in RESOURCES],
    (ActionType.PLAY_ROAD_BUILDING, None),
    *[(ActionType.PLAY_MONOPOLY, r) for r in RESOURCES],
    *[
        (ActionType.MARITIME_TRADE, tuple(4 * [i] + [j]))
        for i in RESOURCES
        for j in RESOURCES
        if i != j
    ],
    *[
        (ActionType.MARITIME_TRADE, tuple(3 * [i] + [None, j]))
        for i in RESOURCES
        for j in RESOURCES
        if i != j
    ],
    *[
        (ActionType.MARITIME_TRADE, tuple(2 * [i] + [None, None, j]))
        for i in RESOURCES
        for j in RESOURCES
        if i != j
    ],
    (ActionType.END_TURN, None),
]
ACTION_SPACE_SIZE = len(ACTIONS_ARRAY)


def normalize_action(action):
    # (Function implementation from the original code)
    normalized = action
    if normalized.action_type == ActionType.ROLL:
        return Action(action.color, action.action_type, None)
    elif normalized.action_type == ActionType.MOVE_ROBBER:
        return Action(action.color, action.action_type, action.value[0])
    elif normalized.action_type == ActionType.BUILD_ROAD:
        return Action(action.color, action.action_type, tuple(sorted(action.value)))
    elif normalized.action_type == ActionType.BUY_DEVELOPMENT_CARD:
        return Action(action.color, action.action_type, None)
    elif normalized.action_type == ActionType.DISCARD:
        return Action(action.color, action.action_type, None)
    return normalized


def to_action_space(action):
    normalized = normalize_action(action)
    return ACTIONS_ARRAY.index((normalized.action_type, normalized.value))


def from_action_space(action_int, playable_actions):
    (action_type, value) = ACTIONS_ARRAY[action_int]
    for action in playable_actions:
        normalized = normalize_action(action)
        if normalized.action_type == action_type and normalized.value == value:
            return action
    raise ValueError(f"Action {action_int} not found in playable_actions")


HIGH = 19 * 5

# --- PettingZoo AEC Environment Implementation ---


class CatanAECEnv(AECEnv):
    """
    A PettingZoo AEC environment for the game of Catan.
    This environment supports multiple agents, turn-based gameplay,
    and provides action masks for legal moves.
    """

    metadata = {
        "render_modes": ["human"],
        "name": "catanatron_v1",
        "is_parallelizable": False,
    }

    def __init__(
        self,
        num_players=2,
        map_type="BASE",
        vps_to_win=10,
        representation="vector",
        invalid_action_reward=-1,
    ):
        super().__init__()

        # Game configuration
        self.map_type = map_type
        self.vps_to_win = vps_to_win
        self.representation = representation
        self.invalid_action_reward = invalid_action_reward
        assert self.representation in ["mixed", "vector"]
        assert 2 <= num_players <= 4, "Catan must be played with 2 to 4 players"

        # Agent setup
        self.possible_agents = [f"player_{i}" for i in range(num_players)]

        self.color_map = {
            agent: list(Color)[i] for i, agent in enumerate(self.possible_agents)
        }
        self.agent_map = {color: agent for agent, color in self.color_map.items()}
        self.catan_players = [Player(color) for agent, color in self.color_map.items()]

        # Define action and observation spaces
        self._action_spaces = {
            agent: spaces.Discrete(ACTION_SPACE_SIZE) for agent in self.possible_agents
        }

        self.features = get_feature_ordering(len(self.possible_agents), self.map_type)
        if self.representation == "mixed":
            channels = get_channels(len(self.possible_agents))
            board_tensor_space = spaces.Box(
                low=0, high=1, shape=(channels, 21, 11), dtype=np.float32
            )
            self.numeric_features = [
                f for f in self.features if not is_graph_feature(f)
            ]
            numeric_space = spaces.Box(
                low=0, high=HIGH, shape=(len(self.numeric_features),), dtype=np.float32
            )
            core_obs_space = spaces.Dict(
                {"board": board_tensor_space, "numeric": numeric_space}
            )
        else:
            core_obs_space = spaces.Box(
                low=0, high=HIGH, shape=(len(self.features),), dtype=np.float32
            )

        self._observation_spaces = {
            agent: spaces.Dict(
                {
                    "observation": core_obs_space,
                    "action_mask": spaces.Box(
                        low=0, high=1, shape=(ACTION_SPACE_SIZE,), dtype=np.int8
                    ),
                }
            )
            for agent in self.possible_agents
        }

        # Game state holders
        self.game: Game = None
        self.invalid_actions_count = {}
        self.max_invalid_actions = 10

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self._observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self._action_spaces[agent]

    def observe(self, agent):
        """Returns the observation for the specified agent."""
        agent_color = self.color_map[agent]
        core_observation = self._get_core_observation(agent_color)

        # The action mask is only valid for the agent whose turn it is
        legal_moves = (
            self.game.state.playable_actions
            if self.agent_map.get(self.game.state.current_color()) == agent
            else []
        )
        action_mask = self._get_action_mask(legal_moves)

        return {"observation": core_observation, "action_mask": action_mask}

    def reset(self, seed=None, options=None):
        """Resets the environment to a starting state."""
        # Initialize Catan Game
        catan_map = build_map(self.map_type)
        for player in self.catan_players:
            player.reset_state()

        self.game = Game(
            players=self.catan_players,
            seed=seed,
            catan_map=catan_map,
            vps_to_win=self.vps_to_win,
        )
        self.invalid_actions_count = {agent: 0 for agent in self.possible_agents}

        # Initialize AEC state
        self.agents = self.possible_agents[:]
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self.agent_map[self.game.state.current_color()]

        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

    def step(self, action):
        """Executes one action for the current agent."""
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return

        current_agent = self.agent_selection
        is_action_valid = action in self._get_valid_action_indices()

        # Execute action or penalize
        if not is_action_valid:
            self.invalid_actions_count[current_agent] += 1
            self.rewards = {agent: 0 for agent in self.agents}
            self.rewards[current_agent] = self.invalid_action_reward
        else:
            catan_action = from_action_space(action, self.game.state.playable_actions)
            self.game.execute(catan_action)
            self.rewards = {agent: 0 for agent in self.agents}

        # Check for game end conditions
        winning_color = self.game.winning_color()
        is_terminated = winning_color is not None
        is_truncated = (
            self.game.state.num_turns >= TURNS_LIMIT
            or self.invalid_actions_count[current_agent] > self.max_invalid_actions
        )

        if is_terminated:
            winner_agent = self.agent_map[winning_color]
            for agent in self.agents:
                self.rewards[agent] = 1 if agent == winner_agent else -1
                self.terminations[agent] = True
        elif is_truncated:
            for agent in self.agents:
                self.truncations[agent] = True

        # Set next agent
        if not (is_terminated or is_truncated):
            self.agent_selection = self.agent_map[self.game.state.current_color()]
        else:
            # If the game is over, cycle through agents to allow final observations
            self.agent_selection = self._agent_selector.next()

        # Update rewards and info
        for agent in self.agents:
            self._cumulative_rewards[agent] += self.rewards[agent]
            self.infos[agent] = {"turn": self.game.state.num_turns}

    def render(self, mode="human"):
        """Renders the current state of the game."""
        if mode == "human":
            print("-" * 30)
            print(
                f"Turn: {self.game.state.num_turns}, Agent to play: {self.agent_selection}"
            )
            for player in self.game.state.players:
                agent_name = self.agent_map[player.color]
                print(
                    f"  - {agent_name} ({player.color.name}): {player.victory_points()} VPs"
                )
            if self.game.winning_color():
                print(f"Winner: {self.agent_map[self.game.winning_color()]}")
            print("-" * 30)

    def close(self):
        """Closes the environment."""
        pass

    def _get_action_mask(self, legal_moves):
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.int8)
        if not legal_moves:
            return mask
        legal_indices = [to_action_space(action) for action in legal_moves]
        mask[legal_indices] = 1
        return mask

    def _get_valid_action_indices(self):
        return {to_action_space(action) for action in self.game.state.playable_actions}

    def _get_core_observation(self, agent_color: Color) -> Union[np.ndarray, dict]:
        """Generates the core observation for a specific agent color."""
        sample = create_sample(self.game, agent_color)
        if self.representation == "mixed":
            board_tensor = create_board_tensor(
                self.game, agent_color, channels_first=True
            ).astype(np.float32)
            numeric = np.array(
                [float(sample[i]) for i in self.numeric_features], dtype=np.float32
            )
            return {"board": board_tensor, "numeric": numeric}
        else:
            return np.array([float(sample[i]) for i in self.features], dtype=np.float32)


# --- PettingZoo Environment Factory Functions ---


def env(**kwargs):
    """Factory function for creating the AEC environment."""
    env_instance = CatanAECEnv(**kwargs)
    env_instance = wrappers.AssertOutOfBoundsWrapper(env_instance)
    env_instance = wrappers.OrderEnforcingWrapper(env_instance)
    return env_instance


def parallel_env(**kwargs):
    """Factory function for creating the parallel API version of the environment."""
    from pettingzoo.utils import aec_to_parallel

    aec_env_instance = env(**kwargs)
    parallel_env_instance = aec_to_parallel(aec_env_instance)
    return parallel_env_instance
