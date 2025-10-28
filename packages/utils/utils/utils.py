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


def legal_moves_mask(num_actions: int, legal_moves, device="cpu"):
    assert isinstance(
        legal_moves, list
    ), "Legal moves should be a list got {} of type {}".format(
        legal_moves, type(legal_moves)
    )
    # add a dimension if the legal moves are not a list of lists

    # assert (
    #     len(legal_moves) == actions.shape[0]
    # ), "Legal moves should be the same length as the batch size"
    legal_mask = torch.ones(num_actions)
    mask = torch.zeros_like(legal_mask, dtype=torch.bool).to(device)
    for i, legal in enumerate(legal_moves):
        mask[i, legal] = True
    legal_mask = torch.where(mask, legal_mask, torch.tensor(0).to(device)).to(device)

    return legal_mask


def action_mask(
    actions: Tensor, legal_moves, mask_value: float = 0, device="cpu"
) -> Tensor:
    """
    Mask actions that are not legal moves
    actions: Tensor, probabilities of actions or q-values
    """
    assert isinstance(
        legal_moves, list
    ), "Legal moves should be a list got {} of type {}".format(
        legal_moves, type(legal_moves)
    )
    # add a dimension if the legal moves are not a list of lists

    # assert (
    #     len(legal_moves) == actions.shape[0]
    # ), "Legal moves should be the same length as the batch size"

    mask = torch.zeros_like(actions, dtype=torch.bool).to(device)
    for i, legal in enumerate(legal_moves):
        mask[i, legal] = True
    # print(mask)
    # print(actions)
    # actions[mask == 0] = mask_value
    actions = torch.where(mask, actions, torch.tensor(mask_value).to(device)).to(device)
    # print(mask)

    return actions


def clip_low_prob_actions(actions: Tensor, low_prob: float = 0.01) -> Tensor:
    """
    Clip actions with probability lower than low_prob to 0
    actions: Tensor, probabilities of actions
    """
    # print("Actions in low prob func", actions)
    if low_prob == 0:
        return actions
    mask = actions < low_prob
    # print("Mask", mask)
    actions = torch.where(mask, 0.0, actions)
    # print("Actions after clipping", actions)
    return actions


def get_legal_moves(info: dict | list[dict]):
    # print(info)
    # if isinstance(info, dict):
    return [info["legal_moves"] if "legal_moves" in info else None]
    # else:
    #     return [(i["legal_moves"] if "legal_moves" in i else None) for i in info]


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


def action_mask_to_legal_moves(action_mask):
    # print(action_mask)
    legal_moves = [i for i, x in enumerate(action_mask) if x == 1]
    # print(legal_moves)
    return legal_moves


def update_per_beta(
    per_beta: float, per_beta_final: float, per_beta_steps: int, initial_per_beta: int
):
    # could also use an initial per_beta instead of current (multiply below equation by current step)
    if per_beta < per_beta_final:
        clamp_func = min
    else:
        clamp_func = max
    per_beta = clamp_func(
        per_beta_final,
        per_beta + (per_beta_final - initial_per_beta) / (per_beta_steps),
    )

    return per_beta


def update_linear_schedule(
    final_value: float,
    total_steps: int,
    initial_value: float,
    current_step: int,
):
    # learning_rate = initial_value
    if initial_value < final_value:
        clamp_func = min
    else:
        clamp_func = max
    value = clamp_func(
        final_value,
        initial_value + ((final_value - initial_value) * (current_step / total_steps)),
    )
    return value


def update_inverse_sqrt_schedule(
    initial_value: float = None,
    current_step: int = None,
):
    return initial_value / math.sqrt(current_step + 1)


def prepare_kernel_initializers(kernel_initializer: str, output_layer: bool = False):
    if kernel_initializer == "pytorch_default":
        return None
    if kernel_initializer == "glorot_uniform":
        return nn.init.xavier_uniform_
    elif kernel_initializer == "glorot_normal":
        return nn.init.xavier_normal_
    elif kernel_initializer == "he_uniform":
        # return lambda tensor: nn.init.kaiming_uniform_(tensor, nonlinearity="relu")
        return nn.init.kaiming_uniform_
    elif kernel_initializer == "he_normal":
        # return lambda tensor: nn.init.kaiming_normal_(tensor, nonlinearity="relu")
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
    q_values: list[float], info: dict, epsilon: float, wrapper=np.argmax
):
    if np.random.rand() < epsilon:
        # print("selecting a random move")
        if "legal_moves" in info:
            return random.choice(info["legal_moves"])
        else:
            q_values = q_values.reshape(-1)
            return random.choice(range(len(q_values)))
    else:
        # try:
        # print("using provided wrapper to select action")
        return wrapper(q_values, info)
    # except:
    #     return wrapper(q_values)


def add_dirichlet_noise(
    policy: list[float], dirichlet_alpha: float, exploration_fraction: float
):
    # MAKE ALPHAZERO USE THIS
    noise = np.random.dirichlet([dirichlet_alpha] * len(policy))
    frac = exploration_fraction
    for i, n in enumerate(noise):
        policy[i] = (1 - frac) * policy[i] + frac * n
    return policy


def augment_game(game, flip_y: bool = False, flip_x: bool = False, rot90: bool = False):
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


def augment_board(
    board, policy, flip_y: bool = False, flip_x: bool = False, rot90: bool = False
):
    if (rot90 and flip_y) or (rot90 and flip_x):
        augemented_boards = [copy.deepcopy(board) for _ in range(7)]
        augmented_policies = [copy.deepcopy(policy) for _ in range(7)]
        augemented_boards[0] = np.rot90(board)
        augmented_policies[0] = np.rot90(policy)
        augemented_boards[1] = np.rot90(np.rot90(board))
        augmented_policies[1] = np.rot90(np.rot90(policy))
        augemented_boards[2] = np.rot90(np.rot90(np.rot90(board)))
        augmented_policies[2] = np.rot90(np.rot90(np.rot90(policy)))
        augemented_boards[3] = np.flipud(board)
        augmented_policies[3] = np.flipud(policy)
        augemented_boards[4] = np.flipud(np.rot90(board))
        augmented_policies[4] = np.flipud(np.rot90(policy))
        augemented_boards[5] = np.flipud(np.rot90(np.rot90(board)))
        augmented_policies[5] = np.flipud(np.rot90(np.rot90(policy)))
        augemented_boards[6] = np.rot90(np.flipud(board))
        augmented_policies[6] = np.rot90(np.flipud(policy))
    elif rot90 and not flip_y and not flip_x:
        augemented_boards = [copy.deepcopy(board) for _ in range(3)]
        augmented_policies = [copy.deepcopy(policy) for _ in range(3)]
        augemented_boards[0] = np.rot90(board)
        augmented_policies[0] = np.rot90(policy)
        augemented_boards[1] = np.rot90(np.rot90(board))
        augmented_policies[1] = np.rot90(np.rot90(policy))
        augemented_boards[2] = np.rot90(np.rot90(np.rot90(board)))
        augmented_policies[2] = np.rot90(np.rot90(np.rot90(policy)))
    elif flip_y and not rot90 and not flip_x:
        augemented_boards = [copy.deepcopy(board)]
        augmented_policies = [copy.deepcopy(policy)]
        augemented_boards[0] = np.flipud(board)
        augmented_policies[0] = np.flipud(policy)
    elif flip_x and not rot90 and not flip_y:
        augemented_boards = [copy.deepcopy(board)]
        augmented_policies = [copy.deepcopy(policy)]
        augemented_boards[0] = np.fliplr(board)
        augmented_policies[0] = np.fliplr(policy)
    augemented_boards.append(board)
    augmented_policies.append(policy)
    return augemented_boards, augmented_policies


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
    assert torch.allclose(
        torch.sum(predicted, dim=axis, keepdim=True),
        torch.ones_like(torch.sum(predicted, dim=axis, keepdim=True)),
    )

    # print("Predicted:", predicted)
    # predicted = predicted / torch.sum(predicted, dim=axis, keepdim=True)
    # print("Normalized Predicted:", predicted)
    predicted = torch.clamp(predicted, _epsilon, 1.0 - _epsilon)
    # print("Clamped Predicted:", predicted)
    log_prob = torch.log(predicted)
    # print("Log Prob:", log_prob)
    return -torch.sum(log_prob * target, axis=axis)


class CategoricalCrossentropyLoss:
    def __init__(self, from_logits=False, axis=-1):
        self.from_logits = from_logits
        self.axis = axis

    def __call__(self, predicted, target):
        return categorical_crossentropy(predicted, target, self.axis)

    def __eq__(self, other):
        if not isinstance(other, CategoricalCrossentropyLoss):
            return False
        return self.from_logits == other.from_logits and self.axis == other.axis


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

    def __eq__(self, other):
        if not isinstance(other, KLDivergenceLoss):
            return False
        return self.from_logits == other.from_logits and self.axis == other.axis


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

    def __eq__(self, other):
        if not isinstance(other, HuberLoss):
            return False
        return self.axis == other.axis and self.delta == other.delta


def mse(predicted: torch.Tensor, target: torch.Tensor):
    # print(predicted)
    # print(target)
    return (predicted - target) ** 2


class MSELoss:
    def __init__(self):
        pass

    def __call__(self, predicted, target):
        return mse(predicted, target)

    def __eq__(self, other):
        return isinstance(other, MSELoss)


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

            # print([stat_dict["score"] for stat_dict in stats["test_score"][-5:]])
            final_rolling_averages.append(
                np.around(
                    np.mean(
                        [stat_dict["score"] for stat_dict in stats["test_score"][-5:]]
                    ),
                    1,
                )
            )

            final_std_devs.append(
                np.around(
                    np.std(
                        [stat_dict["score"] for stat_dict in stats["test_score"][-5:]]
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


import cv2
import numpy as np
import os
from typing import Optional, Dict, Any, Callable
from pettingzoo import AECEnv
from pettingzoo.utils import BaseWrapper


# For pickling and multiprocessing compatibility
class EpisodeTrigger:
    def __init__(self, period: int):
        self.period = period

    def __call__(self, episode_id: int) -> bool:
        return episode_id % self.period == 0


class RecordVideo(BaseWrapper):
    """
    Records video of PettingZoo AEC environment episodes.

    Args:
        env: The PettingZoo AEC environment to wrap
        video_folder: Directory to save videos
        episode_trigger: Function that takes episode_id and returns True if recording should start
        video_length: Maximum number of frames per video (0 for unlimited)
        name_prefix: Prefix for video filenames
        fps: Frames per second for the video
        codec: Video codec (fourcc format)
    """

    def __init__(
        self,
        env: AECEnv,
        video_folder: str = "videos",
        episode_trigger: Optional[Callable[[int], bool]] = None,
        video_length: int = 0,
        name_prefix: str = "episode",
        fps: int = 30,
        codec: str = "mp4v",
    ):
        super().__init__(env)

        self.video_folder = video_folder
        self.episode_trigger = episode_trigger or EpisodeTrigger(1000)
        self.video_length = video_length
        self.name_prefix = name_prefix
        self.fps = fps
        self.codec = codec

        # Create video directory
        os.makedirs(self.video_folder, exist_ok=True)

        # Video recording state
        self.recording = False
        self.video_writer = None
        self.frames_recorded = 0
        self.episode_id = 0
        self.episode_started = False

        # Ensure environment has render capability
        if not hasattr(env, "render"):
            raise ValueError("Environment must support rendering to record video")

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ):
        """Reset environment and potentially start recording new episode"""
        # Call parent reset
        super().reset(seed=seed, options=options)

        # Check if we should record this episode
        should_record = self.episode_trigger(self.episode_id)

        if should_record and not self.recording:
            self._start_recording()
        elif not should_record and self.recording:
            self._stop_recording()

        # Capture initial frame if recording
        if self.recording:
            self._capture_frame()

        self.episode_started = True
        return None  # PettingZoo AEC reset doesn't return anything

    def step(self, action):
        """Step environment and capture frame if recording"""
        # Call parent step - this updates env state but returns nothing
        super().step(action)

        # Capture frame if recording
        if self.recording:
            self._capture_frame()

        # Check if episode ended
        episode_ended = (
            not self.agents  # No more agents
            or all(self.terminations.values())
            or all(self.truncations.values())
        )

        # Stop recording if episode ended or max frames reached
        if self.recording and (
            episode_ended
            or (self.video_length > 0 and self.frames_recorded >= self.video_length)
        ):
            self._stop_recording()

        # If episode ended, increment episode counter
        if episode_ended and self.episode_started:
            self.episode_id += 1
            self.episode_started = False

        return None  # PettingZoo AEC step doesn't return anything

    def _start_recording(self):
        """Initialize video recording"""
        if self.recording:
            self._stop_recording()

        # Generate filename
        video_name = f"{self.name_prefix}_{self.episode_id:06d}.mp4"
        video_path = os.path.join(self.video_folder, video_name)

        # Get a frame to determine video dimensions
        frame = self._get_frame()
        if frame is None:
            print(
                f"Warning: Could not get frame for recording episode {self.episode_id}"
            )
            return

        height, width = frame.shape[:2]

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self.video_writer = cv2.VideoWriter(
            video_path, fourcc, self.fps, (width, height)
        )

        if not self.video_writer.isOpened():
            print(f"Warning: Could not open video writer for {video_path}")
            self.video_writer = None
            return

        self.recording = True
        self.frames_recorded = 0
        print(f"Started recording episode {self.episode_id} to {video_path}")

    def _stop_recording(self):
        """Stop video recording and save file"""
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            print(
                f"Stopped recording episode {self.episode_id}. Recorded {self.frames_recorded} frames."
            )

        self.recording = False
        self.frames_recorded = 0

    def _get_frame(self):
        """Get current frame from environment"""
        try:
            # Try to render as rgb_array
            frame = self.env.render()

            if frame is None:
                return None

            # Convert to numpy array if needed
            if not isinstance(frame, np.ndarray):
                return None

            # Ensure frame is in correct format (BGR for OpenCV)
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                # Convert RGB to BGR for OpenCV
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            elif len(frame.shape) == 3 and frame.shape[2] == 4:
                # Convert RGBA to BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            elif len(frame.shape) == 2:
                # Convert grayscale to BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                print(f"Warning: Unexpected frame shape: {frame.shape}")
                return None

            return frame

        except Exception as e:
            print(f"Warning: Error getting frame: {e}")
            return None

    def _capture_frame(self):
        """Capture and write current frame to video"""
        if not self.recording or self.video_writer is None:
            return

        frame = self._get_frame()
        if frame is not None:
            self.video_writer.write(frame)
            self.frames_recorded += 1

    def close(self):
        """Clean up video recording"""
        if self.recording:
            self._stop_recording()
        super().close()

    def __del__(self):
        """Ensure video recording is stopped on deletion"""
        if hasattr(self, "recording") and self.recording:
            self._stop_recording()


# Convenience function for common use cases
def record_video_wrapper(
    env: AECEnv,
    video_folder: str = "videos",
    record_every: int = 1000,
    max_frames: int = 0,
    fps: int = 30,
):
    """
    Convenience function to wrap environment with video recording.

    Args:
        env: PettingZoo environment
        video_folder: Where to save videos
        record_every: Record every N episodes (default: every 1000)
        max_frames: Maximum frames per video (0 for unlimited)
        fps: Video framerate
    """
    return RecordVideo(
        env=env,
        video_folder=video_folder,
        episode_trigger=EpisodeTrigger(record_every),
        video_length=max_frames,
        fps=fps,
    )


def numpy_dtype_to_torch_dtype(np_dtype):
    """Converts a NumPy dtype to its corresponding PyTorch dtype."""
    # Create a temporary NumPy array with the desired dtype
    temp_np_array = np.empty([], dtype=np_dtype)
    # Convert it to a PyTorch tensor and extract its dtype
    return torch.from_numpy(temp_np_array).dtype
