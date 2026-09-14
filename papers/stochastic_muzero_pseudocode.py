# Copyright 2022 DeepMind Technologies Limited.
# Licensed under the Apache License, Version 2.0 and CC BY 4.0.
# You may not use this file except in compliance with these licenses.
# Copies of the licenses can be found at https://www.apache.org/licenses/LICENSE-2.0
# and https://creativecommons.org/licenses/by/4.0/legalcode.

"""Pseudocode description of the Stochastic MuZero algorithm.
This pseudocode was adapted from the original MuZero pseudocode.
"""
# pylint: disable=unused-argument
# pylint: disable=missing-docstring
# pylint: disable=g-explicit-length-test
import abc
import math
from typing import (
    Any,
    Dict,
    Callable,
    List,
    NamedTuple,
    Tuple,
    Union,
    Optional,
    Sequence,
)

import dataclasses
import numpy as np

MAXIMUM_FLOAT_VALUE = float("inf")

########################################
####### Environment interface ##########

# An action to apply to the environment.
# It can a single integer or a list of micro-actions for backgammon.
Action = Any

# The current player to play.
Player = int


class Environment:
    """Implements the rules of the environment."""

    def apply(self, action: Action):
        """Applies an action or a chance outcome to the environment."""

    def observation(self):
        """Returns the observation of the environment to feed to the network."""

    def is_terminal(self) -> bool:
        """Returns true if the environment is in a terminal state."""
        return False

    def legal_actions(self) -> Sequence[Action]:
        """Returns the legal actions for the current state."""
        return []

    def reward(self, player: Player) -> float:
        """Returns the last reward for the player."""
        return 0.0

    def to_play(self) -> Player:
        """Returns the current player to play."""
        return 0


##########################
####### Helpers ##########


class KnownBounds(NamedTuple):
    min: float
    max: float


class MinMaxStats(object):
    """A class that holds the min-max values of the tree."""

    def __init__(self, known_bounds: Optional[KnownBounds]):
        self.maximum = known_bounds.max if known_bounds else -MAXIMUM_FLOAT_VALUE
        self.minimum = known_bounds.min if known_bounds else MAXIMUM_FLOAT_VALUE

    def update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values.
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


# A chance outcome.
Outcome = Any

# An object that holds an action or a chance outcome.
ActionOrOutcome = Union[Action, Outcome]

LatentState = List[float]
AfterState = List[float]


class NetworkOutput(NamedTuple):
    value: float
    probabilities: Dict[ActionOrOutcome, float]
    reward: Optional[float] = 0.0


class Network:
    """An instance of the network used by stochastic MuZero."""

    def representation(self, observation) -> LatentState:
        """Representation function maps from observation to latent state."""
        return []

    def predictions(self, state: LatentState) -> NetworkOutput:
        """Returns the network predictions for a latent state."""
        return NetworkOutput(0, {}, 0)

    def afterstate_dynamics(self, state: LatentState, action: Action) -> AfterState:
        """Implements the dynamics from latent state and action to afterstate."""
        return []

    def afterstate_predictions(self, state: AfterState) -> NetworkOutput:
        """Returns the network predictions for an afterstate."""
        # No reward for afterstate transitions.
        return NetworkOutput(0, {})

    def dynamics(self, state: AfterState, action: Outcome) -> LatentState:
        """Implements the dynamics from afterstate and chance outcome to state."""
        return []

    def encoder(self, observation) -> Outcome:
        """An encoder maps an observation to an outcome."""


class NFSPAveragePolicyNetwork:
    """An instance of the average policy network for NFP."""

    def forward(self, observation) -> Dict[Action, float]:
        """Returns the average policy of NFSP."""


class NetworkCacher:
    """An object to share the network between the self-play and training jobs."""

    def __init__(self):
        self._networks = {}

    def save_network(self, step: int, network: Network):
        self._networks[step] = network

    def load_network(self) -> Tuple[int, Network]:
        training_step = max(self._networks.keys())
        return training_step, self._networks[training_step]


# Takes the training step and returns the temperature of the softmax policy.
VisitSoftmaxTemperatureFn = Callable[[int], float]

# Returns an instance of the environment.
EnvironmentFactory = Callable[[], Environment]

# The factory for the network.
NetworkFactory = Callable[[], Network]

# The factory for the NFSP Network.
NFSPNetworkFactory = Callable[[], NFSPAveragePolicyNetwork]


@dataclasses.dataclass
class StochasticMuZeroConfig:
    # A factory for the environment.
    environment_factory: EnvironmentFactory
    network_factory: NetworkFactory

    # Self-Play
    num_actors: int
    visit_softmax_temperature_fn: VisitSoftmaxTemperatureFn
    num_simulations: int
    discount: float

    # Root prior exploration noise.
    root_dirichlet_alpha: float
    root_dirichlet_fraction: float
    root_dirichlet_adaptive: bool

    # UCB formula
    pb_c_base: float = 19652
    pb_c_init: float = 1.25

    # If we already have some information about which values occur in the
    # environment, we can use them to initialize the rescaling.
    # This is not strictly necessary, but establishes identical behaviour to
    # AlphaZero in board games.
    known_bounds: Optional[KnownBounds] = None

    # Replay buffer.
    num_trajectories_in_buffer: int = int(1e6)
    batch_size: int = int(128)
    num_unroll_steps: int = 5
    td_steps: int = 6
    td_lambda: float = 1.0
    # Alpha and beta parameters for prioritization.
    # By default they are set to 0 which means uniform sampling.
    priority_alpha: float = 0.0
    priority_beta: float = 0.0

    # Reservoir sampling to be used only for NFSP.
    revervoir_replay_size: int = -1

    # A factor to decide the ratio between the average policy acting
    # and the best response one (Stochastic MuZero).
    # See https://arxiv.org/abs/1603.01121 for more details.
    anticipatory_factor: float = 0.1

    # A network factory for the average policy in NFSP.
    nfsp_network_factory: Optional[NFSPNetworkFactory] = None

    # The learning rate for the NFSP average policy.
    nfsp_learning_rate: float = 5e-3

    # Training
    training_steps: int = int(1e6)
    export_network_every: int = int(1e3)
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4

    # The number of chance codes (codebook size).
    # We use a codebook of size 32 for all our experiments.
    codebook_size: int = 32


##################################
## Environment specific configs ##


def twentyfortyeight_config() -> StochasticMuZeroConfig:
    """Returns the config for the game of 2048."""

    def environment_factory():
        # Returns an implementation of 2048.
        return Environment()

    def network_factory():
        # 10 layer fully connected Res V2 network with Layer normalization and size
        # 256.
        return Network()

    def visit_softmax_temperature(train_steps: int) -> float:
        if train_steps < 1e5:
            return 1.0
        elif train_steps < 2e5:
            return 0.5
        elif train_steps < 3e5:
            return 0.1
        else:
            # Greedy selection.
            return 0.0

    return StochasticMuZeroConfig(
        environment_factory=environment_factory,
        network_factory=network_factory,
        num_actors=1000,
        visit_softmax_temperature=visit_softmax_temperature,
        num_simulations=100,
        discount=0.999,
        root_dirichlet_alpha=0.3,
        root_dirichlet_fraction=0.1,
        root_dirichlet_adaptive=False,
        num_trajectories_in_buffer=int(125e3),
        td_steps=10,
        td_lambda=0.5,
        priority_alpha=1.0,
        priority_beta=1.0,
        training_steps=int(20e6),
        batch_size=1024,
        weight_decay=0.0,
    )


def backgammon_config() -> StochasticMuZeroConfig:
    """Returns the config for the game of 2048."""

    def environment_factory():
        # Returns an backgammon. We consider single games without a doubling cube.
        return Environment()

    def network_factory():
        # 10 layer fully connected Res V2 network with Layer normalization and size
        # 256.
        return Network()

    def visit_softmax_temperature(train_steps: int) -> float:
        return 1.0

    return StochasticMuZeroConfig(
        environment_factory=environment_factory,
        network_factory=network_factory,
        num_actors=1000,
        visit_softmax_temperature_fn=visit_softmax_temperature,
        num_simulations=1600,
        discount=1.0,
        # Unused, we use adaptive dirichlet for backgammon.
        root_dirichlet_alpha=-1.0,
        root_dirichlet_fraction=0.1,
        root_dirichlet_adaptive=True,
        # Max value is 3 for backgammon.
        known_bounds=KnownBounds(min=-3, max=3),
        # 1e5 full episodes stored.
        num_trajectories_in_buffer=int(1e5),
        # We use monte carlo returns.
        td_steps=int(1e3),
        training_steps=int(8e6),
        batch_size=1024,
        learning_rate=3e-4,
        weight_decay=1e-4,
    )


def leduc_poker_config() -> StochasticMuZeroConfig:
    """Returns the config for the game of 2048."""

    def environment_factory():
        # Returns an OpenSpiel implementation of Leduc poker.
        return Environment()

    def network_factory():
        # 2 layer fully connected network with size of 128.
        return Network()

    def nfsp_network_factory():
        # single layer network with a size of 256.
        return NFSPAveragePolicyNetwork()

    def visit_softmax_temperature(train_steps: int) -> float:
        # Greedy policy throughout.
        return 0.0

    return StochasticMuZeroConfig(
        # A factory for the environment.
        environment_factory=environment_factory,
        network_factory=network_factory,
        num_actors=1000,
        visit_softmax_temperature=visit_softmax_temperature,
        # We use 1600 simulations for leduc poker given the high
        # number of possible chance outcomes.
        num_simulations=1600,
        discount=1.0,
        root_dirichlet_alpha=0.25,
        root_dirichlet_fraction=0.1,
        root_dirichlet_adaptive=False,
        num_trajectories_in_buffer=int(5e4),
        revervoir_replay_size=int(2e6),
        anticipatory_factor=0.1,
        nfsp_network_factory=nfsp_network_factory,
        nfsp_learning_rate=5e-3,
        training_steps=int(3.5e6),
        batch_size=256,
        export_network_every=int(500),
        learning_rate=3e-4,
        weight_decay=0.0,
    )


##################################
############ Replay ##############
class SearchStats(NamedTuple):
    search_policy: Dict[Action, int]
    search_value: float


class State(NamedTuple):
    """Data for a single state."""

    observation: List[float]
    reward: float
    discount: float
    player: Player
    action: Action
    search_stats: SearchStats
    # For NFSP trajectories we store the average policy.
    # This can be used for V-Trace off-policy correction when training
    # on targets generated by the nfsp average policy.
    nfsp_average_policy: Optional[Dict[Action, float]] = None


Trajectory = Sequence[State]


class ReplayBuffer:
    """A replay buffer to hold the experience generated by the selfplay."""

    def __init__(self, config: StochasticMuZeroConfig):
        self.config = config
        self.data = []

    def save(self, seq: Trajectory):
        if len(self.data) > self.config.num_trajectories_in_buffer:
            # Remove the oldest sequence from the buffer.
            self.data.pop(0)
        self.data.append(seq)

    def sample_trajectory(self) -> Trajectory:
        """Samples a trajectory uniformly or using prioritization."""
        return self.data[0]

    def sample_index(self, seq: Trajectory) -> int:
        """Samples an index in the trajectory uniformly or using prioritization."""
        return 0

    def sample_element(self) -> Trajectory:
        """Samples a single element from the buffer."""
        # Sample a trajectory.
        trajectory = self.sample_trajectory()
        state_idx = self.sample_index(trajectory)
        limit = max([self.config.num_unroll_steps, self.config.td_steps])
        # Returns a trajectory of experiment.
        return trajectory[state_idx : state_idx + limit]

    def sample(self) -> Sequence[Trajectory]:
        """Samples a training batch."""
        return [self.sample_element() for _ in range(self.config.batch_size)]


class ReservoirBuffer(object):
    """A replay buffer to hold the experience generated by the selfplay."""

    def __init__(self, config: StochasticMuZeroConfig):
        self.config = config
        self.added = 0
        self.data = []

    def save(self, seq: Trajectory):
        for state in seq:
            # Add each state in the reservoir buffer.
            if len(self.data) < self.config.revervoir_replay_size:
                self.data.append(state)
            else:
                if np.random.randint(0, self.added) < self.config.revervoir_replay_size:
                    index = np.random.randint(0, self.config.revervoir_replay_size)
                    self.data[index] = state
            self.added += 1

    def sample(self) -> Sequence[State]:
        """Samples a training batch."""
        return [np.random.choice(self.data) for _ in range(self.config.batch_size)]


##################################
############ Search ##############
class ActionOutcomeHistory:
    """Simple history container used inside the search.
    Only used to keep track of the actions and chance outcomes executed.
    """

    def __init__(self, player: Player, history: Optional[List[ActionOrOutcome]] = None):
        self.initial_player = player
        self.history = list(history or [])

    def clone(self):
        return ActionOutcomeHistory(self.initial_player, self.history)

    def add_action_or_outcome(self, action_or_outcome: ActionOrOutcome):
        self.history.append(action_or_outcome)

    def last_action_or_outcome(self) -> ActionOrOutcome:
        return self.history[-1]

    def to_play(self) -> Player:
        # Returns the next player to play based on the initial player and the
        # history of actions and outcomes. For example for backgammon the two
        # players alternate, while for 2048 it is always the same player.
        return 0


class Node(object):
    """A Node in the MCTS search tree."""

    def __init__(self, prior: float, is_chance: bool = False):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.state = None
        self.is_chance = is_chance
        self.reward = 0

    def expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count


# Core Monte Carlo Tree Search algorithm.
# To decide on an action, we run N simulations, always starting at the root of
# the search tree and traversing the tree according to the UCB formula until we
# reach a leaf node.
def run_mcts(
    config: StochasticMuZeroConfig,
    root: Node,
    action_outcome_history: ActionOutcomeHistory,
    network: Network,
    min_max_stats: MinMaxStats,
):
    for _ in range(config.num_simulations):
        history = action_outcome_history.clone()
        node = root
        search_path = [node]

        while node.expanded():
            action_or_outcome, node = select_child(config, node, min_max_stats)
            history.add_action(action_or_outcome)
            search_path.append(node)

        # Inside the search tree we use the dynamics function to obtain the next
        # hidden state given an action and the previous hidden state.
        parent = search_path[-2]

        if parent.is_chance:
            # The parent is a chance node, afterstate to latent state transition.
            # The last action or outcome is a chance outcome.
            child_state = network_output.dynamics(
                parent.state, history.last_action_or_outcome()
            )
            network_output = network_output.predictions(child_state)
            # This child is a decision node.
            is_child_chance = False
        else:
            # The parent is a decision node, latent state to afterstate transition.
            # The last action or outcome is an action.
            child_state = network_output.afterstate_dynamics(
                parent.state, history.last_action_or_outcome()
            )
            network_output = network_output.afterstate_predictions(child_state)
            # The child is a chance node.
            is_child_chance = True

        # Expand the node.
        expand_node(
            node, child_state, network_output, history.to_play(), is_child_chance
        )
        # Backpropagate the value up the tree.
        backpropagate(
            search_path,
            network_output.value,
            history.to_play(),
            config.discount,
            min_max_stats,
        )


# Select the child with the highest UCB score.
def select_child(
    config: StochasticMuZeroConfig, node: Node, min_max_stats: MinMaxStats
):
    if node.is_chance:
        # If the node is chance we sample from the prior.
        outcomes, probs = zip(*[(o, n.prob) for o, n in node.children.items()])
        outcome = np.random.choice(outcomes, p=probs)
        return outcome, node.children[outcome]

    # For decision nodes we use the pUCT formula.
    _, action, child = max(
        (ucb_score(config, node, child, min_max_stats), action, child)
        for action, child in node.children.items()
    )
    return action, child


# The score for a node is based on its value, plus an exploration bonus based on
# the prior.
def ucb_score(
    config: StochasticMuZeroConfig,
    parent: Node,
    child: Node,
    min_max_stats: MinMaxStats,
) -> float:
    pb_c = (
        math.log((parent.visit_count + config.pb_c_base + 1) / config.pb_c_base)
        + config.pb_c_init
    )
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

    prior_score = pb_c * child.prior
    if child.visit_count > 0:
        value_score = min_max_stats.normalize(
            child.reward + config.discount * child.value()
        )
    else:
        value_score = 0
    return prior_score + value_score


# We expand a node using the value, reward and policy prediction obtained from
# the neural network.
def expand_node(
    node: Node,
    state: Union[LatentState, AfterState],
    network_output: NetworkOutput,
    player: Player,
    is_chance: bool,
):
    node.to_play = player
    node.state = state
    node.is_chance = is_chance
    node.reward = network_output.reward
    for action, prob in network_output.probabilities.items():
        node.children[action] = Node(prob)


# At the end of a simulation, we propagate the evaluation all the way up the
# tree to the root.
def backpropagate(
    search_path: List[Node],
    value: float,
    to_play: Player,
    discount: float,
    min_max_stats: MinMaxStats,
):
    for node in reversed(search_path):
        node.value_sum += value if node.to_play == to_play else -value
        node.visit_count += 1
        min_max_stats.update(node.value())
        value = node.reward + discount * value


# At the start of each search, we add dirichlet noise to the prior of the root
# to encourage the search to explore new actions.
def add_exploration_noise(config: StochasticMuZeroConfig, node: Node):
    actions = list(node.children.keys())
    dir_alpha = config.root_dirichlet_alpha
    if config.root_dirichlet_adaptive:
        dir_alpha = 1.0 / np.sqrt(len(actions))

    noise = np.random.dirichlet([dir_alpha] * len(actions))
    frac = config.root_exploration_fraction
    for a, n in zip(actions, noise):
        node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac


##################################
############ Self-play ###########


class Actor(metaclass=abc.ABCMeta):
    """An actor to interact with the environment."""

    @abc.abstractmethod
    def reset(self):
        """Resets the player for a new episode."""

    @abc.abstractmethod
    def select_action(self, env: Environment) -> Action:
        """Selects an action for the current state of the environment."""

    @abc.abstractmethod
    def stats(self) -> SearchStats:
        """Returns the stats for the player after it has selected an action."""


class NFSPAveragePolicyActor(Actor):
    """An actor which uses the nfsp network."""

    def __init__(self, cacher: NetworkCacher):
        self.cacher = cacher
        self.network = None

    def reset(self):
        _, self.network = self.cacher.load_network()
        self.last_average_policy = None

    def select_action(self, env: Environment) -> Action:
        self.last_average_policy = self.network.forward(env.observation)
        return softmax_sample(self.last_action_or_outcome, temperature=1.0)

    def stats(self):
        if self.last_average_policy is None:
            raise ValueError("No average policy was called.")
        return self.last_average_policy


class StochasticMuZeroActor(Actor):

    def __init__(self, config: StochasticMuZeroConfig, cacher: NetworkCacher):
        self.config = config
        self.cacher = cacher
        self.training_step = -1
        self.network = None

    def reset(self):
        # Read a network from the cacher for the new episode.
        self.training_step, self.network = self.cacher.load_network()
        self.root = None

    def _mask_illegal_actions(
        self, env: Environment, outputs: NetworkOutput
    ) -> NetworkOutput:
        """Masks any actions which are illegal at the root."""
        # We mask out and keep only the legal actions.
        masked_policy = {}
        network_policy = outputs.probabilities
        norm = 0
        for action in env.legal_actions():
            if action in network_policy:
                masked_policy[action] = network_policy[action]
            else:
                masked_policy[action] = 0.0
            norm += masked_policy[action]

        # Renormalize the masked policy.
        masked_policy = {a: v / norm for a, v in masked_policy.items()}
        return NetworkOutput(value=outputs.value, probabilities=masked_policy)

    def _select_action(self, root: Node):
        """Selects an action given the root node."""
        # Get the visit count distribution.
        actions, visit_counts = zip(
            *[(action, node.visit_counts) for action, node in node.children.items()]
        )

        # Temperature
        temperature = self.config.visit_softmax_temperature_fn(self.training_step)

        # Compute the search policy.
        search_policy = [v ** (1.0 / temperature) for v in visit_counts]
        norm = sum(search_policy)
        search_policy = [v / norm for v in search_policy]
        return np.random.choice(actions, p=search_policy)

    def select_action(self, env: Environment) -> Action:
        """Selects an action."""

        # New min max stats for the search tree.
        min_max_stats = MinMaxStats(self.config.known_bounds)

        # At the root of the search tree we use the representation function to
        # obtain a hidden state given the current observation.
        root = Node(0)

        # Provide the history of observations to the representation network to
        # get the initial latent state.
        latent_state = self.network.representation(env.observation())
        # Compute the predictions.
        outputs = self.network.predictions(latent_state)

        # Keep only the legal actions.
        outputs = self._mask_illegal_actions(env, outputs)

        # Expand the root node.
        expand_node(root, latent_state, outputs, env.to_play(), is_chance=False)

        # Backpropagate the value.
        backpropagate(
            [root], outputs.value, env.to_play(), self.config.discount, min_max_stats
        )
        # We add exploration noise to the root node.
        add_exploration_noise(self.config, root)

        # We then run a Monte Carlo Tree Search using only action sequences and the
        # model learned by the network.
        run_mcts(
            self.config,
            root,
            ActionOutcomeHistory(env.to_play()),
            self.network,
            min_max_stats,
        )

        # Keep track of the root to return the stats.
        self.root = root

        # Return an action.
        return self._select_action(root)

    def stats(self) -> SearchStats:
        """Returns the stats of the latest search."""
        if self.root is None:
            raise ValueError("No search was executed.")
        return SearchStats(
            search_policy={
                action: node.visit_counts for action, node in self.root.children.items()
            },
            search_value=self.root.value(),
        )


# Standard self-play.
# Each self-play job is independent of all others; it takes the latest network
# snapshot, produces an episode and makes it available to the training job by
# writing it to a shared replay buffer.
def run_selfplay(
    config: StochasticMuZeroConfig, cacher: NetworkCacher, replay_buffer: ReplayBuffer
):
    actor = StochasticMuZeroActor(config, cacher)
    while True:
        # Create a new instance of the environment.
        env = config.environment_factory()

        # Reset the actor.
        actor.reset()

        episode = []
        while not env.is_terminal():
            action = actor.select_action(env)
            state = State(
                observation=env.observation(),
                reward=env.reward(env.to_play()),
                discount=config.discount,
                player=env.to_play(),
                action=action,
                search_stats=actor.stats(),
            )
            episode.append(state)
            env.apply(action)

        # Send the episode to the replay.
        replay_buffer.save(episode)


# NFSP self-play.
def run_nfsp_selfplay(
    config: StochasticMuZeroConfig,
    cacher: NetworkCacher,
    nfsp_cacher: NetworkCacher,
    replay_buffer: ReplayBuffer,
    reservoir_buffer: ReservoirBuffer,
):
    best_response_actor = StochasticMuZeroActor(config, cacher)
    average_policy_actor = NFSPAveragePolicyActor(nfsp_cacher)
    while True:
        # Create a new instance of the environment.
        env = config.environment_factory()

        # Reset the actors.
        best_response_actor.reset()
        average_policy_actor.reset()

        # The agent perceives the environment from a single player
        # perspective, so we store the data for each player separately.
        episodes = [[], []]

        # Whether to execute actions using the average policy or the best response
        # one for each player.
        use_avg_policy = [
            np.random.uniform() > config.anticipatory_factor,
            np.random.uniform() > config.anticipatory_factor,
        ]

        while not env.is_terminal():
            current_player = env.to_play()
            best_response_action = best_response_actor.select_action(env)
            avg_policy_action = average_policy_actor.select_action(env)
            if use_avg_policy[current_player]:
                action = avg_policy_action
            else:
                action = best_response_action

            state = State(
                observation=env.observation(),
                reward=env.reward(env.to_play()),
                discount=config.discount,
                player=current_player,
                action=action,
                search_stats=best_response_actor.stats(),
                nfsp_average_policy=average_policy_actor.stats(),
            )
            episodes[current_player].append(state)
            env.apply(action)

        # Send the episodes for each player to the replay.
        for episode in episodes:
            replay_buffer.save(episode)

        # We store only the best response episodes to the reservoir buffer.
        for use_avg_p, episode in zip(use_avg_policy, episodes):
            if use_avg_p:
                reservoir_buffer.save(episode)


##################################
############ Training ############


class Learner(metaclass=abc.ABCMeta):
    """An learner to update the network weights based."""

    @abc.abstractmethod
    def learn(self):
        """Single training step of the learner."""

    @abc.abstractmethod
    def export(self) -> Union[Network, NFSPAveragePolicyNetwork]:
        """Exports the network."""


def policy_loss(predictions, labels):
    """Minimizes the KL-divergence of the predictions and labels."""
    return 0.0


def value_or_reward_loss(prediction, target):
    """Implements the value or reward loss for Stochastic MuZero.
    For backgammon this is implemented as an MSE loss of scalars.
    For 2048 and Leduc poker, we use the two hot representation proposed in
    MuZero, and this loss is implemented as a KL divergence between the value
    and value target representations.
    For 2048 we also apply a hyperbolic transformation to the target (see paper
    for more information).
    Args:
      prediction: The reward or value output of the network.
      target: The reward or value target.
    Returns:
      The loss to minimize.
    """
    return 0.0


class StochasticMuZeroLearner(Learner):
    """Implements the learning for Stochastic MuZero."""

    def __init__(self, config: StochasticMuZeroConfig, replay_buffer: ReplayBuffer):
        self.config = config
        self.replay_buffer = replay_buffer
        # Instantiate the network.
        self.network = config.network_factory()

    def transpose_to_time(self, batch):
        """Transposes the data so the leading dimension is time instead of batch."""
        return batch

    def learn(self):
        """Applies a single training step."""
        batch = self.replay_buffer.sample()

        # Transpose batch to make time the leading dimension.
        batch = self.transpose_to_time(batch)

        # Compute the initial step loss.
        latent_state = self.network.representation(batch[0].observation)
        predictions = self.network.predictions(latent_state)

        # Computes the td target for the 0th position.
        value_target = compute_td_target(
            self.config.td_steps, self.config.td_lambda, batch
        )

        # Train the network value towards the td target.
        total_loss = value_or_reward_loss(predictions.value, value_target)

        # Train the network policy towards the MCTS policy.
        total_loss += policy_loss(
            predictions.probabilities, batch[0].search_stats.search_policy
        )

        # Unroll the model for k steps.
        for t in range(1, self.config.num_unroll_steps + 1):
            # Condition the afterstate on the previous action.
            afterstate = self.network.afterstate_dynamics(
                latent_state, batch[t - 1].action
            )
            afterstate_predictions = self.network.afterstate_predictions(afterstate)

            # Call the encoder on the next observation.
            # The encoder returns the chance code which is a discrete one hot code.
            # The gradients flow to the encoder using a straight through estimator.
            chance_code = self.network.encoder(batch[t].observation)

            # The afterstate value is trained towards the previous value target
            # but conditioned on the selected action to obtain a Q-estimate.
            total_loss += value_or_reward_loss(
                afterstate_predictions.value, value_target
            )

            # The afterstate distribution is trained to predict the chance code
            # generated by the encoder.
            total_loss += policy_loss(afterstate_predictions.probabilities, chance_code)

            # Get the dynamic predictions.
            latent_state = self.network.dynamics(afterstate, chance_code)
            predictions = self.network.predictions(latent_state)

            # Compute the new value target.
            value_target = compute_td_target(
                self.config.td_steps, self.config.td_lambda, batch[t:]
            )
            # The reward loss for the dynamics network.
            total_loss += value_or_reward_loss(predictions.reward, batch[t].reward)
            total_loss += value_or_reward_loss(predictions.value, value_target)
            total_loss += policy_loss(
                predictions.probabilities, batch[t].search_stats.search_policy
            )

        minimize_with_adam_and_weight_decay(
            total_loss,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def export(self) -> Network:
        return self.network


class NFSPAveragePolicyLearner(Learner):
    """Implements the learning for average policy in NFSP."""

    def __init__(
        self, config: StochasticMuZeroConfig, reservoir_buffer: ReservoirBuffer
    ):
        self.config = config
        self.reservoir_buffer = reservoir_buffer
        # Instantiate the network.
        self.network = config.nfsp_network_factory()

    def learn(self):
        """Applies a single training step."""
        batch = self.reservoir_buffer.sample()
        nfsp_policy = self.network.forward(batch.observation)
        # Move the NFSP policy towards the search policy.
        total_loss = policy_loss(nfsp_policy, batch.search_stats.search_policy)
        minimize_with_sgd(total_loss, learning_rate=self.config.nfsp_learning_rate)

    def export(self) -> Network:
        return self.network


def train_stochastic_muzero(
    config: StochasticMuZeroConfig, cacher: NetworkCacher, replay_buffer: ReplayBuffer
):
    learner = StochasticMuZeroLearner(config, replay_buffer)

    # Export the network so the actors can start generating experience.
    cacher.save_network(0, learner.export())

    for step in range(config.training_steps):
        # Single learning step.
        learner.learn()
        if step > 0 and step % config.export_network_every == 0:
            cacher.save_network(step, learner.export())


def train_nfsp_average_policy(
    config: StochasticMuZeroConfig,
    cacher: NetworkCacher,
    reservoir_buffer: ReservoirBuffer,
):
    learner = NFSPAveragePolicyLearner(config, reservoir_buffer)

    # Export the network so the actors can start generating experience.
    cacher.save_network(0, learner.export())

    for step in range(config.training_steps):
        # Single learning step.
        learner.learn()
        if step > 0 and step % config.export_network_every == 0:
            cacher.save_network(step, learner.export())


##################################
############ RL loop #############
def launch_stochastic_muzero(config: StochasticMuZeroConfig):
    """Full RL loop for stochastic MuZero."""
    replay_buffer = ReplayBuffer(config)
    cacher = NetworkCacher()

    # Launch a learner job.
    launch_job(lambda: train_stochastic_muzero(config, cacher, replay_buffer))

    # Launch the actors.
    for _ in range(config.num_actors):
        launch_job(lambda: run_selfplay(config, cacher, replay_buffer))


def launch_nfsp_stochastic_muzero(config: StochasticMuZeroConfig):
    """Full RL loop for stochastic MuZero."""
    replay_buffer = ReplayBuffer(config)
    reservoir_buffer = ReservoirBuffer(config)
    # Cacher for the Stochastic MuZero network.
    cacher = NetworkCacher()
    # Cacher for the average policy network.
    nfsp_cacher = NetworkCacher()

    # Launch a learner for Stochastic MuZero.
    launch_job(lambda: train_stochastic_muzero(config, cacher, replay_buffer))

    # Launch a learner for the average policy.
    launch_job(lambda: train_nfsp_average_policy(config, nfsp_cacher, reservoir_buffer))

    # Launch the actors.
    def nfsp_selfplay():
        return run_nfsp_selfplay(
            config, cacher, nfsp_cacher, replay_buffer, reservoir_buffer
        )

    for _ in range(config.num_actors):
        launch_job(nfsp_selfplay)


# Stubs to make the typechecker happy.
def softmax_sample(distribution, temperature: float):
    return 0, 0


def compute_td_target(td_steps, td_lambda, trajectory):
    """Computes the TD lambda targets given a trajectory for the 0th element.
    In case of NFSP we use the V-Trace to correct for off-policyness
    of generating trajectories using the average policy and training the
    value of the best response.
    Args:
      td_steps: The number n of the n-step returns.
      td_lambda: The lambda in TD(lambda).
      trajectory: A sequence of states.
    Returns:
      The n-step return.
    """
    return 0.0


def minimize_with_sgd(loss, learning_rate):
    """Minimizes the loss using SGD."""


def minimize_with_adam_and_weight_decay(loss, learning_rate, weight_decay):
    """Minimizes the loss using Adam with weight decay."""


def launch_job(f):
    """Launches a job to run remotely."""
    return f()
