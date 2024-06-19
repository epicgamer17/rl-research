import random
from collections import defaultdict
import copy
import math
from operator import itemgetter
import os
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import scipy
import pickle

from typing import Iterable, Tuple
from datetime import datetime

import torch
from torch import nn, Tensor

import numpy as np
import numpy.typing as npt

import itertools
from hyperopt import space_eval

import pandas as pd


# from ....replay_buffers.base_replay_buffer import Game
# from replay_buffers.segment_tree import SumSegmentTree


def normalize_policies(policies: torch.float32):
    # print(policies)
    policy_sums = policies.sum(axis=-1, keepdims=True)
    # print(policy_sums)
    policies = policies / policy_sums
    return policies


def action_mask(actions: Tensor, legal_moves, mask_value: float = 0) -> Tensor:
    """
    Mask actions that are not legal moves
    actions: Tensor, probabilities of actions or q-values
    """
    assert isinstance(legal_moves, list), "Legal moves should be a list"

    # add a dimension if the legal moves are not a list of lists
    # if len(legal_moves) != actions.shape[0]:
    #     legal_moves = [legal_moves]
    assert (
        len(legal_moves) == actions.shape[0]
    ), "Legal moves should be the same length as the batch size"

    mask = torch.zeros_like(actions, dtype=torch.bool)
    for i, legal in enumerate(legal_moves):
        mask[i, legal] = True
    # print(mask)
    # print(actions)
    # actions[mask == 0] = mask_value
    actions = torch.where(mask, actions, torch.tensor(mask_value))
    # print(mask)
    return actions


def get_legal_moves(info: dict | list[dict]):
    if isinstance(info, dict):
        return [info["legal_moves"] if "legal_moves" in info else None]
    else:
        return [(i["legal_moves"] if "legal_moves" in i else None) for i in info]


def normalize_images(image: Tensor) -> Tensor:
    """Preprocessing step to normalize image with 8-bit (0-255) color inplace.
    Modifys the original tensor

    Args:
        image (Tensor): An 8-bit color image

    Returns:
        Tensor: The tensor divided by 255
    """
    # Return a copy of the tensor divided by 255
    normalized_image = image.div_(255)
    return normalized_image


def make_stack(item: Tensor) -> Tensor:
    """Convert a tensor of shape (*) to (1, *). Does not copy the data; instead,
    returns a view of the original tensor.

    Args:
        item (Tensor):

    Returns:
        Tensor: A view of the original tensor.
    """
    #
    return item.view(1, *item.shape)


def update_per_beta(per_beta: float, per_beta_final: float, per_beta_steps: int):
    # could also use an initial per_beta instead of current (multiply below equation by current step)
    if per_beta < per_beta_final:
        clamp_func = min
    else:
        clamp_func = max
    per_beta = clamp_func(
        per_beta_final, per_beta + (per_beta_final - per_beta) / (per_beta_steps)
    )

    return per_beta


def update_linear_lr_schedule(
    learning_rate: float,
    final_value: float,
    total_steps: int,
    initial_value: float = None,
    current_step: int = None,
):
    # learning_rate = initial_value
    if initial_value < final_value or learning_rate < final_value:
        clamp_func = min
    else:
        clamp_func = max
    if initial_value is not None and current_step is not None:
        learning_rate = clamp_func(
            final_value,
            initial_value
            + (final_value - initial_value) * (current_step / total_steps),
        )
    else:
        learning_rate = clamp_func(
            final_value, learning_rate + (final_value - learning_rate) / total_steps
        )
    return learning_rate


def default_plot_func(
    axs, key: str, values: list[dict], targets: dict, row: int, col: int
):
    axs[row][col].set_title(
        "{} | rolling average: {}".format(key, np.mean(values[-10:]))
    )
    x = np.arange(1, len(values) + 1)
    axs[row][col].plot(x, values)
    if key in targets and targets[key] is not None:
        axs[row][col].axhline(y=targets[key], color="r", linestyle="--")


def plot_scores(axs, key: str, values: list[dict], targets: dict, row: int, col: int):
    if len(values) == 0:
        return
    scores = [value["score"] for value in values]
    x = np.arange(1, len(values) + 1)
    axs[row][col].plot(x, scores)

    has_max_scores = "max_score" in values[0]
    has_min_scores = "min_score" in values[0]
    assert (
        has_max_scores == has_min_scores
    ), "Both max_scores and min_scores must be provided or not provided"

    if has_max_scores:
        max_scores = [value["max_score"] for value in values]
        min_scores = [value["min_score"] for value in values]
        axs[row][col].fill_between(x, min_scores, max_scores, alpha=0.5)

    has_target_model_updates = "target_model_updated" in values[0]
    has_model_updates = "model_updated" in values[0]

    if has_target_model_updates:
        weight_updates = [value["target_model_updated"] for value in values]
        for i, weight_update in enumerate(weight_updates):
            if weight_update:
                axs[row][col].axvline(
                    x=i,
                    color="black",
                    linestyle="dotted",
                    # label="Target Model Weight Update",
                )

    if has_model_updates:
        weight_updates = [value["model_updated"] for value in values]
        for i, weight_update in enumerate(weight_updates):
            if weight_update:
                axs[row][col].axvline(
                    x=i,
                    color="gray",
                    linestyle="dotted",
                    # label="Model Weight Update",
                )

    axs[row][col].set_title(
        f"{key} | rolling average: {np.mean(scores[-10:])} | latest: {scores[-1]}"
    )

    axs[row][col].set_xlabel("Game")
    axs[row][col].set_ylabel("Score")

    axs[row][col].set_xlim(1, len(values))

    if len(scores) > 1:
        best_fit_x, best_fit_y = np.polyfit(x, scores, 1)
        axs[row][col].plot(
            x,
            best_fit_x * x + best_fit_y,
            color="g",
            label="Best Fit Line",
            linestyle="dotted",
        )

    if key in targets and targets[key] is not None:
        axs[row][col].axhline(
            y=targets[key],
            color="r",
            linestyle="dashed",
            label="Target Score: {}".format(targets[key]),
        )

    axs[row][col].legend()


def plot_loss(axs, key: str, values: list[dict], targets: dict, row: int, col: int):
    loss = [value["loss"] for value in values]
    x = np.arange(1, len(values) + 1)
    axs[row][col].plot(x, loss)

    has_target_model_updates = "target_model_updated" in values[0]
    has_model_updates = "model_updated" in values[0]

    if has_target_model_updates:
        weight_updates = [value["target_model_updated"] for value in values]
        for i, weight_update in enumerate(weight_updates):
            if weight_update:
                axs[row][col].axvline(
                    x=i,
                    color="black",
                    linestyle="dotted",
                    # label="Target Model Weight Update",
                )

    if has_model_updates:
        weight_updates = [value["model_updated"] for value in values]
        for i, weight_update in enumerate(weight_updates):
            if weight_update:
                axs[row][col].axvline(
                    x=i,
                    color="gray",
                    linestyle="dotted",
                    # label="Model Weight Update",
                )

    axs[row][col].set_title(
        f"{key} | rolling average: {np.mean(loss[-10:])} | latest: {loss[-1]}"
    )

    axs[row][col].set_xlabel("Time Step")
    axs[row][col].set_ylabel("Loss")

    axs[row][col].set_xlim(1, len(values))

    if key in targets and targets[key] is not None:
        axs[row][col].axhline(
            y=targets[key],
            color="r",
            linestyle="dashed",
            label="Target Score: {}".format(targets[key]),
        )

    axs[row][col].legend()


def plot_exploitability(
    axs, key: str, values: list[dict], targets: dict, row: int, col: int
):
    default_plot_func(axs, key, values, targets, row, col)


def plot_trials(scores: list, file_name: str, final_trial: int = 0):
    fig, axs = plt.subplots(
        1,
        1,
        figsize=(10, 5),
        squeeze=False,
    )
    if final_trial > 0:
        x = np.arange(1, final_trial + 1)
        scores = scores[:final_trial]
    else:
        x = np.arange(1, len(scores) + 1)
    axs[0][0].scatter(x, scores)
    best_fit_x, best_fit_y = np.polyfit(x, scores, 1)
    axs[0][0].plot(
        x,
        best_fit_x * x + best_fit_y,
        color="g",
        label="Best Fit Line",
        linestyle="dotted",
    )

    fig.suptitle("Score of Hyperopt trials over time for Rainbow DQN on CartPole-v1")
    axs[0][0].set_xlabel("Trial")
    axs[0][0].set_ylabel("Score")
    plt.savefig(f"./graphs/{file_name}.png")
    plt.show()
    plt.close(fig)


stat_keys_to_plot_funcs = {
    "test_score": plot_scores,
    "score": plot_scores,
    "policy_loss": plot_loss,
    "value_loss": plot_loss,
    "l2_loss": plot_loss,
    "loss": plot_loss,
    "rl_loss": plot_loss,
    "sl_loss": plot_loss,
    "exploitability": plot_exploitability,  # should this be plot_scores?
}


def plot_graphs(
    stats: dict,
    targets: dict,
    step: int,
    frames_seen: int,
    time_taken: float,
    model_name: str,
    dir: str = "./checkpoints/graphs",
):
    num_plots = len(stats)
    sqrt_num_plots = math.ceil(np.sqrt(num_plots))
    fig, axs = plt.subplots(
        sqrt_num_plots,
        sqrt_num_plots,
        figsize=(10 * sqrt_num_plots, 5 * sqrt_num_plots),
        squeeze=False,
    )

    hours = int(time_taken // 3600)
    minutes = int((time_taken % 3600) // 60)
    seconds = int(time_taken % 60)

    fig.suptitle(
        "training stats | training step {} | frames seen {} | time taken {} hours {} minutes {} seconds".format(
            step, frames_seen, hours, minutes, seconds
        )
    )

    for i, (key, values) in enumerate(stats.items()):
        row = i // sqrt_num_plots
        col = i % sqrt_num_plots

        if key in stat_keys_to_plot_funcs:
            stat_keys_to_plot_funcs[key](axs, key, values, targets, row, col)
        else:
            default_plot_func(axs, key, values, targets, row, col)

    for i in range(num_plots, sqrt_num_plots**2):
        row = i // sqrt_num_plots
        col = i % sqrt_num_plots
        fig.delaxes(axs[row][col])

    # plt.show()
    assert os.path.exists(dir), f"Directory {dir} does not exist"
    plt.savefig("{}/{}.png".format(dir, model_name))

    plt.close(fig)


def prepare_kernel_initializers(kernel_initializer: str, output_layer: bool = False):
    if kernel_initializer == "pytorch_default":
        return None
    if kernel_initializer == "glorot_uniform":
        return nn.init.xavier_uniform_
    elif kernel_initializer == "glorot_normal":
        return nn.init.xavier_normal_
    elif kernel_initializer == "he_uniform":
        return nn.init.kaiming_uniform_
    elif kernel_initializer == "he_normal":
        return nn.init.kaiming_normal_
    elif kernel_initializer == "variance_baseline":
        return VarianceScaling()
    elif kernel_initializer == "variance_0.1":
        return VarianceScaling(scale=0.1)
    elif kernel_initializer == "variance_0.3":
        return VarianceScaling(scale=0.3)
    elif kernel_initializer == "variance_0.8":
        return VarianceScaling(scale=0.8)
    elif kernel_initializer == "variance_3":
        return VarianceScaling(scale=3)
    elif kernel_initializer == "variance_5":
        return VarianceScaling(scale=5)
    elif kernel_initializer == "variance_10":
        return VarianceScaling(scale=10)
    # TODO
    # elif kernel_initializer == "lecun_uniform":
    #     return LecunUniform(seed=np.random.seed())
    # elif kernel_initializer == "lecun_normal":
    #     return LecunNormal(seed=np.random.seed())
    elif kernel_initializer == "orthogonal":
        return nn.init.orthogonal_

    raise ValueError(f"Invalid kernel initializer: {kernel_initializer}")


def prepare_activations(activation: str):
    # print("Activation to prase: ", activation)
    if activation == "linear":
        return nn.Identity()
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "relu6":
        return nn.ReLU6()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "softplus":
        return nn.Softplus()
    elif activation == "soft_sign":
        return nn.Softsign()
    elif activation == "silu" or activation == "swish":
        return nn.SiLU()
    elif activation == "tanh":
        return nn.Tanh()
    # elif activation == "log_sigmoid":
    #     return nn.LogSigmoid()
    elif activation == "hard_sigmoid":
        return nn.Hardsigmoid()
    # elif activation == "hard_silu" or activation == "hard_swish":
    #     return nn.Hardswish()
    # elif activation == "hard_tanh":
    #     return nn.Hardtanh()
    elif activation == "elu":
        return nn.ELU()
    # elif activation == "celu":
    #     return nn.CELU()
    elif activation == "selu":
        return nn.SELU()
    elif activation == "gelu":
        return nn.GELU()
    # elif activation == "glu":
    #     return nn.GLU()

    raise ValueError(f"Activation {activation} not recognized")


def epsilon_greedy_policy(
    q_values: list[float], epsilon: float, range=None, wrapper=np.argmax
):
    if np.random.rand() < epsilon:
        if range is not None:
            return np.random.randint(range)
        else:
            return np.random.randint(len(q_values))
    else:
        return wrapper(q_values)


def add_dirichlet_noise(
    policy: list[float], dirichlet_alpha: float, exploration_fraction: float
):
    # MAKE ALPHAZERO USE THIS
    noise = np.random.dirichlet([dirichlet_alpha] * len(policy))
    frac = exploration_fraction
    for i, n in enumerate(noise):
        policy[i] = (1 - frac) * policy[i] + frac * n
    return policy


def augment_board(
    self, game, flip_y: bool = False, flip_x: bool = False, rot90: bool = False
):
    # augmented_games[0] = rotate 90
    # augmented_games[1] = rotate 180
    # augmented_games[2] = rotate 270
    # augmented_games[3] = flip y (rotate 180 and flip x)
    # augmented_games[4] = rotate 90 and flip y (rotate 270 and flip x)
    # augmented_games[5] = rotate 180 and flip y (flip x)
    # augmented_games[6] = flip y and rotate 90 (rotate 270 and flip y) (rotate 90 and flip x)
    # augmented_games[7] = normal

    if (rot90 and flip_y) or (rot90 and flip_x):
        augemented_games = [copy.deepcopy(game) for _ in range(7)]
        for i in range(len(game.observation_history)):
            board = game.observation_history[i]
            policy = game.policy_history[i]
            augemented_games[0].observation_history[i] = np.rot90(board)
            augemented_games[0].policy_history[i] = np.rot90(policy)
            augemented_games[1].observation_history[i] = np.rot90(np.rot90(board))
            augemented_games[1].policy_history[i] = np.rot90(np.rot90(policy))
            augemented_games[2].observation_history[i] = np.rot90(
                np.rot90(np.rot90(board))
            )
            augemented_games[2].policy_history[i] = np.rot90(np.rot90(np.rot90(policy)))
            augemented_games[3].observation_history[i] = np.flipud(board)
            augemented_games[3].policy_history[i] = np.flipud(policy)
            augemented_games[4].observation_history[i] = np.flipud(np.rot90(board))
            augemented_games[4].policy_history[i] = np.flipud(np.rot90(policy))
            augemented_games[5].observation_history[i] = np.flipud(
                np.rot90(np.rot90(board))
            )
            augemented_games[5].policy_history[i] = np.flipud(
                np.rot90(np.rot90(policy))
            )
            augemented_games[6].observation_history[i] = np.rot90(np.flipud(board))
            augemented_games[6].policy_history[i] = np.rot90(np.flipud(policy))
    elif rot90 and not flip_y and not flip_x:
        augemented_games = [copy.deepcopy(game) for _ in range(3)]
        augemented_games[0].observation_history = [
            np.rot90(board) for board in game.observation_history
        ]
        augemented_games[0].policy_history = [
            np.rot90(policy) for policy in game.policy_history
        ]
        augemented_games[1].observation_history = [
            np.rot90(np.rot90(board)) for board in game.observation_history
        ]
        augemented_games[1].policy_history = [
            np.rot90(np.rot90(policy)) for policy in game.policy_history
        ]
        augemented_games[2].observation_history = [
            np.rot90(np.rot90(np.rot90(board))) for board in game.observation_history
        ]
        augemented_games[2].policy_history = [
            np.rot90(np.rot90(np.rot90(policy)) for policy in game.policy_history)
        ]
    elif flip_y and not rot90 and not flip_x:
        augemented_games = [copy.deepcopy(game)]
        augemented_games[0].observation_history = [
            np.flipud(board) for board in game.observation_history
        ]
        augemented_games[0].policy_history = [
            np.flipud(policy) for policy in game.policy_history
        ]

    elif flip_x and not rot90 and not flip_y:
        augemented_games = [copy.deepcopy(game) for _ in range(1)]
        augemented_games[0].observation_history = [
            np.fliplr(board) for board in game.observation_history
        ]
        augemented_games[0].policy_history = [
            np.fliplr(policy) for policy in game.policy_history
        ]

    augemented_games.append(game)
    return augemented_games


def calculate_observation_buffer_shape(max_size, observation_dimensions):
    raise DeprecationWarning(
        "This function is deprecated simply use (max_size,) + observation_dimensions"
    )
    # observation_buffer_shape = []
    # observation_buffer_shape += [max_size]
    # observation_buffer_shape += list(observation_dimensions)
    observation_buffer_shape = (max_size,) + observation_dimensions
    return list(observation_buffer_shape)


def sample_by_random_indices(
    max_index_or_1darray, batch_size: int, with_replacement=False
) -> npt.NDArray[np.int64]:
    """
    Sample from a numpy array using indices
    """
    return np.random.choice(max_index_or_1darray, batch_size, replace=with_replacement)


def sample_by_indices_probability(
    max_index_or_1darray, batch_size: int, probabilities: npt.NDArray[np.float64]
) -> npt.NDArray[np.int64]:
    """
    Sample from a numpy array using indices
    """
    return np.random.choice(max_index_or_1darray, batch_size, p=probabilities)


def sample_tree_proportional(
    tree, batch_size: int, max_size: int
) -> npt.NDArray[np.int64]:
    """
    tree: SumSegmentTree
    Sample proportionally from a sum segment tree. Used in prioritized experience replay
    """
    indices = np.zeros(batch_size, dtype=np.int64)
    total_priority = tree.sum(0, max_size - 1)
    priority_segment = total_priority / batch_size

    for i in range(batch_size):
        l = priority_segment * i
        h = priority_segment * (i + 1)
        upperbound = np.random.uniform(l, h)
        indices[i] = tree.retrieve(upperbound)
        # print(tree[indices[i]])

    return indices


def reward_clipping(reward: float, lower_bound: float = -1, upper_bound: float = 1):
    if reward < lower_bound:
        return lower_bound
    elif reward > upper_bound:
        return upper_bound
    return reward


def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def exploitability(env, average_policy, best_response_policies, num_episodes):
    # play average against best responses and measure the reward the expected value or value
    pass


def nash_convergence(env, average_policies, best_response_policies, num_episodes):
    # for every player (average policy) play against the corresponding best policies
    # sum the exploitability of each player and then divide by the number of players
    pass


def to_lists(l: list[Iterable]) -> list[Tuple]:
    """Convert a list of iterables to a zip of tuples

    Args:
        list (list[Iterable]): A list of iterables, e.g. [(1,1,1),(2,2,2),(3,3,3)]

    Returns:
        list[Tuple]: A list of tuples, i.e. [(1,2,3), (1,2,3), (1,2,3)]
    """

    return list(zip(*l))


def current_timestamp():
    return datetime.now().timestamp()


_epsilon = 1e-7


def categorical_crossentropy(predicted: torch.Tensor, target: torch.Tensor, axis=-1):
    predicted = predicted / torch.sum(predicted, dim=axis, keepdim=True)
    predicted = torch.clamp(predicted, _epsilon, 1.0 - _epsilon)
    log_prob = torch.log(predicted)
    return -torch.sum(log_prob * target, axis=axis)


class CategoricalCrossentropyLoss:
    def __init__(self, from_logits=False, axis=-1):
        self.from_logits = from_logits
        self.axis = axis

    def __call__(self, predicted, target):
        return categorical_crossentropy(predicted, target, self.axis)


def kl_divergence(predicted: torch.Tensor, target: torch.Tensor, axis=-1):
    predicted = predicted / torch.sum(predicted, dim=axis, keepdim=True)
    predicted = torch.clamp(predicted, _epsilon, 1.0)
    target = torch.clamp(target, _epsilon, 1.0)
    return torch.sum(target * torch.log(target / predicted), axis=axis)


class KLDivergenceLoss:
    def __init__(self, from_logits=False, axis=-1):
        self.from_logits = from_logits
        self.axis = axis

    def __call__(self, predicted, target):
        return kl_divergence(predicted, target, self.axis)


def huber(predicted: torch.Tensor, target: torch.Tensor, axis=-1, delta: float = 1.0):
    diff = torch.abs(predicted - target)
    return torch.where(
        diff < delta, 0.5 * diff**2, delta * (diff - 0.5 * delta)
    ).view(-1)


class HuberLoss:
    def __init__(self, axis=-1, delta: float = 1.0):
        self.axis = axis
        self.delta = delta

    def __call__(self, predicted, target):
        return huber(predicted, target, axis=self.axis, delta=self.delta)


from typing import Callable

Loss = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def calculate_padding(i: int, k: int, s: int) -> Tuple[int, int]:
    """Calculate both padding sizes along 1 dimension for a given input length, kernel length, and stride

    Args:
        i (int): input length
        k (int): kernel length
        s (int): stride

    Returns:
        (p_1, p_2): where p_1 = p_2 - 1 for uneven padding and p_1 == p_2 for even padding
    """

    p = (i - 1) * s - i + k
    p_1 = p // 2
    p_2 = (p + 1) // 2
    return (p_1, p_2)


def generate_layer_widths(widths: list[int], max_num_layers: int) -> list[Tuple[int]]:
    """Create all possible combinations of widths for a given number of layers"""
    width_combinations = []

    for i in range(0, max_num_layers):
        width_combinations.extend(itertools.combinations_with_replacement(widths, i))

    return width_combinations


def hyperopt_analysis(
    data_dir: str,
    file_name: str,
    viable_trial_threshold: int,
    step: int,
    final_trial: int = 0,
    eval_method: str = "final_score",
):
    trials = pickle.load(open(f"{data_dir}/{file_name}.p", "rb"))
    if final_trial > 0:
        print("Number of trials: {}".format(final_trial))
    else:
        print("Number of trials: {}".format(len(trials.trials)))
    # losses.sort()
    # print(len(os.listdir(f"{data_dir}/checkpoints")) - 1)
    # print(len(trials.trials))

    checkpoints = os.listdir(f"{data_dir}/checkpoints")
    checkpoints.remove("videos") if "videos" in checkpoints else None
    checkpoints.remove(".DS_Store") if ".DS_Store" in checkpoints else None
    checkpoints.sort(key=lambda x: int(x.split("_")[-1]))
    if final_trial > 0:
        checkpoints = checkpoints[:final_trial]

    viable_throughout_trials = []
    final_rolling_averages = []
    final_std_devs = []
    scores = []
    losses = []
    failed_trials = 0
    for i, trial in enumerate(trials.trials):
        losses.append(trial["result"]["loss"])
        if final_trial > 0 and i >= final_trial:
            break
        # print(trial["result"]["status"])
        if trial["result"]["status"] == "fail":
            failed_trials += 1
            final_rolling_averages.append(trial["result"]["loss"])
            scores.append(trial["result"]["loss"])
            final_std_devs.append(trial["result"]["loss"])
        else:
            # print(checkpoints[i - failed_trials])
            # print(failed_trials)
            # if os.path.exists(
            #     f"{data_dir}/checkpoints/{checkpoints[i - failed_trials]}/step_{step}/graphs_stats/stats.pkl"
            # ):
            stats = pickle.load(
                open(
                    f"{data_dir}/checkpoints/{checkpoints[i - failed_trials]}/step_{step}/graphs_stats/stats.pkl",
                    "rb",
                )
            )
            max_score = 0

            # print([stat_dict["score"] for stat_dict in stats["test_score"][-10:]])
            final_rolling_averages.append(
                np.around(
                    np.mean(
                        [stat_dict["score"] for stat_dict in stats["test_score"][-10:]]
                    ),
                    1,
                )
            )

            final_std_devs.append(
                np.around(
                    np.std(
                        [stat_dict["score"] for stat_dict in stats["test_score"][-10:]]
                    ),
                    1,
                )
            )

            for stat_dict in stats["test_score"]:
                if stat_dict["max_score"] > max_score:
                    max_score = stat_dict["max_score"]

            if max_score > viable_trial_threshold:
                viable_throughout_trials.append(max_score)

            if eval_method == "final_score":
                score = -trial["result"]["loss"]
            elif (
                eval_method == "rolling_average"
                or eval_method == "final_score_rolling_average"
            ):
                score = stats["test_score"][-1]["score"]
            scores.append(score)

    plot_trials(
        scores,
        file_name,
        final_trial=final_trial,
    )

    res = [
        list(x)
        for x in zip(
            *sorted(
                zip(losses, scores, final_rolling_averages, final_std_devs),
                key=itemgetter(0),
            )
        )
    ]
    losses = res[0]
    scores = res[1]
    final_rolling_averages = res[2]
    final_std_devs = res[3]
    viable_trials = [score for score in scores if score > viable_trial_threshold]

    print("Failed trials: ~{}%".format(round(failed_trials / len(scores) * 100)))

    print(
        "Viable trials (based on final score): ~{}%".format(
            round(len(viable_trials) / len(scores) * 100)
        )
    )
    print(
        "Viable trials (throughout training): ~{}%".format(
            round(len(viable_throughout_trials) / len(scores) * 100)
        )
    )

    print("Losses: {}".format(losses))
    print("Scores: {}".format(scores))
    print("Final rolling averages: {}".format(final_rolling_averages))
    print("Final standard deviations: {}".format(final_std_devs))

    print("Max loss: {}".format(max(losses)))
    print("Max score: {}".format(max(scores)))
    print("Max final rolling average: {}".format(max(final_rolling_averages)))
    print("Max final standard deviation: {}".format(max(final_std_devs)))

    print("Average loss: {}".format(np.mean(losses)))
    print("Average score: {}".format(np.mean(scores)))
    print("Average final rolling average: {}".format(np.mean(final_rolling_averages)))
    print("Average final standard deviation: {}".format(np.mean(final_std_devs)))

    viable_final_rolling_averages = [
        final_rolling_averages[i]
        for i, loss in enumerate(scores)
        if loss > viable_trial_threshold
    ]

    viable_std_devs = [
        final_std_devs[i]
        for i, loss in enumerate(scores)
        if loss > viable_trial_threshold
    ]

    print(
        "Average score of viable trials (based on final score): {}".format(
            np.mean(viable_trials)
        )
    )
    print(
        "Average final rolling average of viable trials (based on final score): {}".format(
            np.mean(viable_final_rolling_averages)
        )
    )
    print(
        "Average final standard deviation of viable trials (based on final score): {}".format(
            np.mean(viable_std_devs)
        )
    )


def graph_hyperparameter_importance(
    data_dir: str, trials_file: str, search_space_file: str, viable_trial_threshold: int
):
    with open(f"{data_dir}/{trials_file}", "rb") as f:
        trials = pickle.load(f)
    print(trials)

    search_space = pickle.load(open(f"./search_spaces/{search_space_file}", "rb"))

    values_dict = defaultdict(list)
    scores = []
    for trial in trials.trials:
        for key, value in space_eval(trial["misc"]["vals"], search_space).items():
            values_dict[key].append(value[0])
        scores.append(-trial["result"]["loss"])

    df = pd.DataFrame(values_dict)
    x_cols = df.columns
    df["scores"] = scores
    # print(df)
    df = df[df["scores"] > viable_trial_threshold]

    for col in x_cols:
        if col == "loss_function":
            continue
        plt = df.plot(x=col, y="scores", kind="scatter")
        grouped = df.groupby(col)["scores"]
        medians = grouped.median()
        means = grouped.mean()
        stddev = grouped.std()

        if not (col == "kernel_initializer" or col == "activation"):
            # plt.fill_between(medians.index, medians.values-stddev, medians.values+stddev, color="#00F0F0")
            plt.plot(means.index, means.values, color="#00FFFF")
        else:
            plt.scatter(means.index, means.values, c="#00FFFF")
        # plt.add_line


def calc_units(shape):
    shape = tuple(shape)
    if len(shape) == 1:
        return shape + shape
    if len(shape) == 2:
        # dense layer -> (in_channels, out_channels)
        return shape
    else:
        # conv_layer (Assuming convolution kernels (2D, 3D, or more).
        # kernel shape: (input_depth, depth, ...)
        in_units = shape[1]
        out_units = shape[0]
        c = 1
        for dim in shape[2:]:
            c *= dim
        return (c * in_units, c * out_units)


class VarianceScaling:
    def __init__(self, scale=0.1, mode="fan_in", distribution="uniform"):
        self.scale = scale
        self.mode = mode
        self.distribution = distribution

        assert mode == "fan_in" or mode == "fan_out" or mode == "fan_avg"
        assert distribution == "uniform", "only uniform distribution is supported"

    def __call__(self, tensor: Tensor) -> None:
        with torch.no_grad():
            scale = self.scale
            shape = tensor.shape
            in_units, out_units = calc_units(shape)
            if self.mode == "fan_in":
                scale /= in_units
            elif self.mode == "fan_out":
                scale /= out_units
            else:
                scale /= (in_units + out_units) / 2

            limit = math.sqrt(3.0 * scale)
            return tensor.uniform_(-limit, limit)


def isiterable(o):
    try:
        it = iter(o)
    except TypeError:
        return False
    return True


def tointlists(list):
    ret = []
    for x in list:
        if isiterable(x):
            ret.append(tointlists(x))
        else:
            ret.append(int(x))
    return ret


import time
from collections import deque


class StoppingCriteria:
    def __init__(self):
        pass

    def should_stop(self, details: dict) -> bool:
        return False


class TimeStoppingCriteria(StoppingCriteria):
    def __init__(self, max_runtime_sec=60 * 10):
        self.stop_time = time.time() + max_runtime_sec

    def should_stop(self, details: dict) -> bool:
        return time.time() > self.stop_time


class TrainingStepStoppingCritiera(StoppingCriteria):
    def __init__(self, max_training_steps=100000):
        self.max_training_steps = max_training_steps

    def should_stop(self, details: dict) -> bool:
        return details["training_step"] > self.max_training_steps


class EpisodesStoppingCriteria(StoppingCriteria):
    def __init__(self, max_episodes=100000):
        self.max_episodes = max_episodes

    def should_stop(self, details: dict) -> bool:
        return details["max_episodes"] > self.max_episodes


class AverageScoreStoppingCritera(StoppingCriteria):
    def __init__(self, min_avg_score: float, last_scores_length: int):
        self.min_avg_score = min_avg_score
        self.last_scores_length = last_scores_length
        self.last_scores = deque(maxlen=last_scores_length)

    def add_score(self, score: float):
        self.last_scores.append(score)

    def should_stop(self, details: dict) -> bool:
        if len(self.last_scores) < self.last_scores_length:
            return False

        return np.average(self.last_scores) < self.min_avg_score


class ApexLearnerStoppingCriteria(StoppingCriteria):
    def __init__(self):
        self.criterias: dict[str, StoppingCriteria] = {
            "time": TimeStoppingCriteria(max_runtime_sec=1.5 * 60 * 60),
            "training_step": TrainingStepStoppingCritiera(max_training_steps=10000),
            "avg_score": AverageScoreStoppingCritera(
                min_avg_score=15, last_scores_length=10
            ),
        }

    def should_stop(self, details: dict) -> bool:
        if self.criterias["time"].should_stop(details):
            return True

        if details["training_step"] < 10000:
            return False

        return self.criterias["training_step"].should_stop(details) or self.criterias[
            "avg_score"
        ].should_stop(details)

    def add_score(self, score: float):
        tc: AverageScoreStoppingCritera = self.criterias["avg_score"]
        tc.add_score(score)
