# Copyright 2022 DeepMind Technologies Limited.
# Licensed under the Apache License, Version 2.0 and CC BY 4.0.
# You may not use this file except in compliance with these licenses.
# Copies of the licenses can be found at https://www.apache.org/licenses/LICENSE-2.0
# and https://creativecommons.org/licenses/by/4.0/legalcode.

"""Pseudocode description of the MuZero Unplugged algorithm."""
# pylint: disable=unused-argument
# pylint: disable=missing-docstring
# pylint: disable=g-explicit-length-test

import collections
import math
import random
import typing
from typing import Dict, List, Optional

import dataclasses
import haiku as hk
import jax
import jax.numpy as jnp
import numpy
import optax

##########################
####### Helpers ##########

MAXIMUM_FLOAT_VALUE = float("inf")

KnownBounds = collections.namedtuple("KnownBounds", ["min", "max"])


class MinMaxStats(object):
    """A class that holds the min-max values of the tree."""

    def __init__(self):
        self.maximum = -MAXIMUM_FLOAT_VALUE
        self.minimum = MAXIMUM_FLOAT_VALUE

    def update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values.
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


def visit_softmax_temperature(num_moves, training_steps):
    if training_steps < 500e3:
        return 1.0
    elif training_steps < 750e3:
        return 0.5
    else:
        return 0.25


class MuZeroConfig(object):

    def __init__(
        self,
        action_space_size: int,
        max_moves: int,
        discount: float,
        dirichlet_alpha: float,
        num_simulations: int,
        td_steps: int,
        num_actors: int,
    ):
        ### Self-Play
        self.action_space_size = action_space_size
        self.num_actors = num_actors

        self.visit_softmax_temperature_fn = visit_softmax_temperature
        self.max_moves = max_moves
        self.num_simulations = num_simulations
        self.discount = discount

        # Root prior exploration noise.
        self.root_dirichlet_alpha = dirichlet_alpha
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        ### Training
        self.training_steps = int(1000e3)
        self.checkpoint_interval = 500
        self.window_size = int(1e6)
        self.batch_size = 1024
        self.num_unroll_steps = 5
        self.td_steps = td_steps

        # Cosine learning rate schedule with decoupled weight decay.
        self.lr_init = 1e-4
        self.weight_decay = 1e-4

        # Reanalyse fraction controls the trade-off between environment interactions
        # and running search on stored data.
        # 0.0 corresponds to only environment intercations, 1.0 to the full offline
        # case.
        self.reanalyse_fraction = 1.0

    def new_game(self):
        return Game(self.action_space_size, self.discount, Environment())


def make_atari_config() -> MuZeroConfig:

    return MuZeroConfig(
        action_space_size=18,
        max_moves=27000,  # Half an hour at action repeat 4.
        discount=0.997,
        dirichlet_alpha=0.25,
        num_simulations=50,
        td_steps=5,
        num_actors=350,
    )


class Action(object):

    def __init__(self, index: int):
        self.index = index

    def __hash__(self):
        return self.index

    def __eq__(self, other):
        return self.index == other.index

    def __gt__(self, other):
        return self.index > other.index


class Player(object):
    pass


class Node(object):

    def __init__(self, prior: float):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0

    def expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count


class ActionHistory(object):
    """Simple history container used inside the search.
    Only used to keep track of the actions executed.
    """

    def __init__(self, history: List[Action], action_space_size: int):
        self.history = list(history)
        self.action_space_size = action_space_size

    def clone(self):
        return ActionHistory(self.history, self.action_space_size)

    def add_action(self, action: Action):
        self.history.append(action)

    def last_action(self) -> Action:
        return self.history[-1]

    def action_space(self) -> List[Action]:
        return [Action(i) for i in range(self.action_space_size)]

    def to_play(self) -> Player:
        return Player()


class Environment(object):
    """The environment MuZero is interacting with."""

    def step(self, action) -> float:
        return 0


class Game(object):
    """A single episode of interaction with the environment."""

    def __init__(
        self, action_space_size: int, discount: float, env: Optional[Environment]
    ):
        self.environment = env
        self.history = []
        self.rewards = []
        self.child_visits = []
        self.root_values = []
        self.action_space_size = action_space_size
        self.discount = discount

    def terminal(self) -> bool:
        # Game specific termination rules.
        pass

    def legal_actions(self) -> List[Action]:
        # Game specific calculation of legal actions.
        return []

    def apply(self, action: Action):
        if self.environment:
            reward = self.environment.step(action)
            self.rewards.append(reward)
        self.history.append(action)

    def store_search_statistics(self, root: Node):
        sum_visits = sum(child.visit_count for child in root.children.values())
        action_space = (Action(index) for index in range(self.action_space_size))
        self.child_visits.append(
            [
                root.children[a].visit_count / sum_visits if a in root.children else 0
                for a in action_space
            ]
        )
        self.root_values.append(root.value())

    def make_image(self, state_index: int):
        # Game specific feature planes.
        return []

    def make_target(
        self, state_index: int, num_unroll_steps: int, td_steps: int, to_play: Player
    ):
        # The value target is the discounted root value of the search tree N steps
        # into the future, plus the discounted sum of all rewards until then.
        targets = []
        for current_index in range(state_index, state_index + num_unroll_steps + 1):
            bootstrap_index = current_index + td_steps
            if bootstrap_index < len(self.root_values):
                value = self.root_values[bootstrap_index] * self.discount**td_steps
            else:
                value = 0

            for i, reward in enumerate(self.rewards[current_index:bootstrap_index]):
                value += (
                    reward * self.discount**i
                )  # pytype: disable=unsupported-operands

            if current_index > 0 and current_index <= len(self.rewards):
                last_reward = self.rewards[current_index - 1]
            else:
                last_reward = None

            if current_index < len(self.root_values):
                targets.append((value, last_reward, self.child_visits[current_index]))
            else:
                # States past the end of games are treated as absorbing states.
                targets.append((0, last_reward, []))
        return targets

    def to_play(self) -> Player:
        return Player()

    def action_history(self) -> ActionHistory:
        return ActionHistory(self.history, self.action_space_size)

    def is_reanalyse(self) -> bool:
        return self.environment is None


class StoredGame(Game):
    """A stored Game that can be used for reanalyse."""

    def __init__(self, game: Game):
        super().__init__(game.action_space_size, game.discount, env=None)
        self._stored_history = game.history
        self._stored_rewards = game.rewards

    def terminal(self) -> bool:
        return not self._stored_history

    def apply(self, action: Action):
        # Ignore the action, instead replay the stored data.
        del action
        self.rewards.append(self._stored_rewards.pop(0))
        self.history.append(self._stored_history.pop(0))


class ReplayBuffer(object):

    def __init__(self, config: MuZeroConfig):
        self.window_size = config.window_size
        self.batch_size = config.batch_size
        self.buffer = []

    def save_game(self, game):
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        self.buffer.append(game)

    def sample_batch(self, num_unroll_steps: int, td_steps: int):
        games = [self.sample_game() for _ in range(self.batch_size)]
        game_pos = [(g, self.sample_position(g)) for g in games]
        return [
            (
                g.make_image(i),
                g.history[i : i + num_unroll_steps],
                g.make_target(i, num_unroll_steps, td_steps, g.to_play()),
            )
            for (g, i) in game_pos
        ]

    def sample_game(self) -> Game:
        # Sample game from buffer either uniformly or according to some priority.
        return self.buffer[0]

    def sample_position(self, game) -> int:
        # Sample position from game either uniformly or according to some priority.
        return -1


class ReanalyseBuffer(object):

    def __init__(self):
        self._games = []

    def save_game(self, game: Game):
        """Saves a new game to the reanalyse buffer, to be reanalysed later."""

    def sample_game(self) -> Game:
        """Samples a game that should be reanalysed."""
        games = random.choices(
            self._games, weights=[len(g.history) for g in self._games], k=1
        )
        return StoredGame(games[0])


class DemonstrationBuffer(ReanalyseBuffer):
    """A reanlayse buffer of a fixed set of demonstrations.
    Can be used to learn from existing policies, human demonstrations or for
    Offline RL.
    """

    def __init__(self, demonstrations: List[Game]):
        super().__init__()
        self._games.extend(demonstrations)


class MostRecentBuffer(ReanalyseBuffer):
    """A reanalyse buffer that keeps the most recent games to reanalyse."""

    def __init__(self, capacity: int):
        super().__init__()
        self._capacity = capacity

    def save_game(self, game: Game):
        self._games.append(game)
        while sum(len(g.history) for g in self._games):
            self._games.pop(0)


class HighestRewardBuffer(ReanalyseBuffer):
    """A reanalyse buffer that keeps games with highest rewards to reanalyse."""

    def __init__(self, capacity: int):
        super().__init__()
        self._capacity = capacity

    def save_game(self, game: Game):
        self._games.append(game)
        while sum(len(g.history) for g in self._games):
            self._games.sort(key=lambda g: sum(g.rewards), reverse=True)
            self._games.pop()


class NetworkOutput(typing.NamedTuple):
    value: float
    reward: float
    policy_logits: Dict[Action, float]
    hidden_state: List[float]


class Network(object):

    def __init__(
        self, params: Optional[hk.Params] = None, state: Optional[hk.State] = None
    ):
        pass

    def initial_inference(self, image) -> NetworkOutput:
        # representation + prediction function
        return NetworkOutput(0, 0, {}, [])

    def recurrent_inference(self, hidden_state, action) -> NetworkOutput:
        # dynamics + prediction function
        return NetworkOutput(0, 0, {}, [])

    def get_weights(self):
        # Returns the weights of this network.
        return []

    def training_steps(self) -> int:
        # How many steps / batches the network has been trained for.
        return 0


class SharedStorage(object):

    def __init__(self):
        self._networks = {}

    def latest_network(self) -> Network:
        if self._networks:
            return self._networks[max(self._networks.keys())]
        else:
            # policy -> uniform, value -> 0, reward -> 0
            return make_uniform_network()

    def save_network(self, step: int, network: Network):
        self._networks[step] = network


def bernoulli(p: float) -> bool:
    return random.random() < p


@dataclasses.dataclass
class GenerationStats:
    total_states: int = 0
    max_episode_length: int = 0
    mean_episode_length: Optional[float] = None

    def episode_length(self):
        if self.mean_episode_length is None:
            return self.max_episode_length
        else:
            return self.mean_episode_length


class ActingStats(object):

    def __init__(self, config: MuZeroConfig):
        self._reanalyse_fraction = config.reanalyse_fraction
        self._fresh = GenerationStats()
        self._reanalysed = GenerationStats()

    def state_added(self, game: Game):
        stats = self._get_stats(game.is_reanalyse())
        stats.total_states += 1
        stats.max_episode_length = max(stats.max_episode_length, len(game.history))

    def game_finished(self, game: Game):
        stats = self._get_stats(game.is_reanalyse())
        if stats.mean_episode_length is None:
            stats.mean_episode_length = len(game.history)
        else:
            alpha = 0.01
            stats.mean_episode_length = (
                alpha * len(game.history) + (1 - alpha) * stats.mean_episode_length
            )

    def should_reanalyse(self) -> bool:
        # Overshoot slightly to approach desired fraction.
        actual = self._reanalysed.total_states / (
            self._reanalysed.total_states + self._fresh.total_states
        )
        target = self._reanalyse_fraction + (self._reanalyse_fraction - actual) / 2
        target = max(0, min(1, target))

        # Correct for reanalysing only part of full episodes.
        fresh_fraction = 1 - target
        parts_per_episode = max(
            1, self._fresh.episode_length() / self._reanalysed.episode_length()
        )
        fresh_fraction /= parts_per_episode

        return bernoulli(1 - fresh_fraction)

    def _get_stats(self, reanalysed: bool) -> GenerationStats:
        return self._reanalysed if reanalysed else self._fresh


##### End Helpers ########
##########################


# MuZero training is split into two independent parts: Network training and
# self-play data generation.
# These two parts only communicate by transferring the latest network checkpoint
# from the training to the self-play, and the finished games from the self-play
# to the training.
def muzero(config: MuZeroConfig):
    storage = SharedStorage()
    replay_buffer = ReplayBuffer(config)
    reanalyse_buffer = ReanalyseBuffer()

    for _ in range(config.num_actors):
        launch_job(run_selfplay, config, storage, replay_buffer, reanalyse_buffer)

    train_network(config, storage, replay_buffer)

    return storage.latest_network()


##################################
####### Part 1: Self-Play ########


# Each self-play job is independent of all others; it takes the latest network
# snapshot, produces a game and makes it available to the training job by
# writing it to a shared replay buffer.
def run_selfplay(
    config: MuZeroConfig,
    storage: SharedStorage,
    replay_buffer: ReplayBuffer,
    reanalyse_buffer: ReanalyseBuffer,
):
    stats = ActingStats(config)
    while True:
        network = storage.latest_network()

        if stats.should_reanalyse():
            game = reanalyse_buffer.sample_game()
        else:
            game = config.new_game()

        game = play_game(config, game, network, stats)

        replay_buffer.save_game(game)
        if not game.is_reanalyse():
            reanalyse_buffer.save_game(game)


# Each game is produced by starting at the initial board position, then
# repeatedly executing a Monte Carlo Tree Search to generate moves until the end
# of the game is reached.
def play_game(
    config: MuZeroConfig, game: Game, network: Network, stats: ActingStats
) -> Game:

    while not game.terminal() and len(game.history) < config.max_moves:
        min_max_stats = MinMaxStats()

        # At the root of the search tree we use the representation function to
        # obtain a hidden state given the current observation.
        root = Node(0)
        current_observation = game.make_image(-1)
        network_output = network.initial_inference(current_observation)
        expand_node(root, game.to_play(), game.legal_actions(), network_output)
        backpropagate(
            [root], network_output.value, game.to_play(), config.discount, min_max_stats
        )
        add_exploration_noise(config, root)

        # We then run a Monte Carlo Tree Search using only action sequences and the
        # model learned by the network.
        run_mcts(config, root, game.action_history(), network, min_max_stats)
        action = select_action(config, len(game.history), root, network)
        game.apply(action)
        game.store_search_statistics(root)
        stats.state_added(game)

    stats.game_finished(game)
    return game


# Core Monte Carlo Tree Search algorithm.
# To decide on an action, we run N simulations, always starting at the root of
# the search tree and traversing the tree according to the UCB formula until we
# reach a leaf node.
def run_mcts(
    config: MuZeroConfig,
    root: Node,
    action_history: ActionHistory,
    network: Network,
    min_max_stats: MinMaxStats,
):
    for _ in range(config.num_simulations):
        history = action_history.clone()
        node = root
        search_path = [node]

        while node.expanded():
            action, node = select_child(config, node, min_max_stats)
            history.add_action(action)
            search_path.append(node)

        # Inside the search tree we use the dynamics function to obtain the next
        # hidden state given an action and the previous hidden state.
        parent = search_path[-2]
        network_output = network.recurrent_inference(
            parent.hidden_state, history.last_action()
        )
        expand_node(node, history.to_play(), history.action_space(), network_output)

        backpropagate(
            search_path,
            network_output.value,
            history.to_play(),
            config.discount,
            min_max_stats,
        )


def select_action(config: MuZeroConfig, num_moves: int, node: Node, network: Network):
    visit_counts = [
        (child.visit_count, action) for action, child in node.children.items()
    ]
    t = config.visit_softmax_temperature_fn(
        num_moves=num_moves, training_steps=network.training_steps()
    )
    _, action = softmax_sample(visit_counts, t)
    return action


# Select the child with the highest UCB score.
def select_child(config: MuZeroConfig, node: Node, min_max_stats: MinMaxStats):
    _, action, child = max(
        (ucb_score(config, node, child, min_max_stats), action, child)
        for action, child in node.children.items()
    )
    return action, child


# The score for a node is based on its value, plus an exploration bonus based on
# the prior.
def ucb_score(
    config: MuZeroConfig, parent: Node, child: Node, min_max_stats: MinMaxStats
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
    node: Node, to_play: Player, actions: List[Action], network_output: NetworkOutput
):
    node.to_play = to_play
    node.hidden_state = network_output.hidden_state
    node.reward = network_output.reward
    policy = {a: math.exp(network_output.policy_logits[a]) for a in actions}
    policy_sum = sum(policy.values())
    for action, p in policy.items():
        node.children[action] = Node(p / policy_sum)


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
def add_exploration_noise(config: MuZeroConfig, node: Node):
    actions = list(node.children.keys())
    noise = numpy.random.dirichlet([config.root_dirichlet_alpha] * len(actions))
    frac = config.root_exploration_fraction
    for a, n in zip(actions, noise):
        node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac


######### End Self-Play ##########
##################################

##################################
####### Part 2: Training #########


def train_network(
    config: MuZeroConfig, storage: SharedStorage, replay_buffer: ReplayBuffer
):
    network = Network()
    network = hk.transform_with_state(network)

    # Sample initial batch of random data, generated with uniform policy.
    batch = replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps)
    params, state = network.init(batch)

    def learning_rate(step: int):
        lr_decay = 0.5 * (1 + jnp.cos(jnp.pi * step / config.training_steps))
        return config.lr_init * lr_decay

    # Adam with decoupled weight decay.
    optimizer = optax.scale_by_adam(eps=1e-8)
    optimizer = optax.chain(
        optimizer, optax.add_decayed_weights(config.weight_decay / config.lr_init)
    )
    optimizer = optax.chain(
        optimizer, optax.scale_by_schedule(learning_rate), optax.scale(-1)
    )
    opt_state = optimizer.init(params)

    for step in range(config.training_steps):
        if step % config.checkpoint_interval == 0:
            storage.save_network(step, Network(params, state))

        batch = replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps)
        grads, state = jax.grad(loss_fn)(params, state, network.apply, batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

    storage.save_network(config.training_steps, Network(params, state))


def scale_gradient(x: jnp.ndarray, scale: float) -> jnp.ndarray:
    """Multiplies the gradient of `x` by `scale`."""

    @jax.custom_gradient
    def wrapped(x: jnp.ndarray):
        return x, lambda grad: (grad * scale,)

    return wrapped(x)


def softmax_cross_entropy(logits, labels):
    return -jnp.sum(labels * jax.nn.log_softmax(logits), axis=-1)


def loss_fn(params: hk.Params, state: hk.State, network: Network, batch):
    loss = 0
    for image, actions, targets in batch:
        # Initial step, from the real observation.
        network_output = network.initial_inference(image)
        hidden_state = network_output.hidden_state
        predictions = [(1.0, network_output)]

        # Recurrent steps, from action and previous hidden state.
        for action in actions:
            network_output = network.recurrent_inference(hidden_state, action)
            hidden_state = network_output.hidden_state
            predictions.append((1.0 / len(actions), network_output))

            hidden_state = scale_gradient(hidden_state, 0.5)

        for k, (prediction, target) in enumerate(zip(predictions, targets)):
            gradient_scale, network_output = prediction
            target_value, target_reward, target_policy = target

            l = softmax_cross_entropy(
                logits=network_output.policy_logits, labels=target_policy
            )
            l += scalar_loss(network_output.value, target_value)
            if k > 0:
                l += scalar_loss(network_output.reward, target_reward)

            loss += scale_gradient(l, gradient_scale)

    return loss / len(batch)


def scalar_loss(prediction, target) -> float:
    # MSE in board games, cross entropy between categorical values in Atari.
    return -1


######### End Training ###########
##################################

################################################################################
############################# End of pseudocode ################################
################################################################################


# Stubs to make the typechecker happy.
def softmax_sample(distribution, temperature: float):
    return 0, 0


def launch_job(f, *args):
    f(*args)


def make_uniform_network():
    return Network()
