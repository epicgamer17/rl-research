import copy
import math
import os
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.initializers import (
    VarianceScaling,
    Orthogonal,
    GlorotUniform,
    GlorotNormal,
    HeNormal,
    HeUniform,
    LecunNormal,
    LecunUniform,
    Initializer,
)


from keras.activations import (
    relu,
    sigmoid,
    softplus,
    softsign,
    hard_sigmoid,
    elu,
    selu,
)

from tensorflow.nn import (
    silu,
    swish,
    gelu,
)


import numpy as np

# from ....replay_buffers.base_replay_buffer import Game


def normalize_policy(policy: np.float16):
    policy /= tf.reduce_sum(policy, axis=1)
    return policy


def action_mask(
    actions: list[int], legal_moves, num_actions: int, mask_value: float = 0
):
    mask = np.zeros(num_actions, dtype=np.int8)
    mask[legal_moves] = 1
    actions[mask == 0] = mask_value
    return actions


def get_legal_moves(info: dict):
    # info["legal_moves"] if self.config.game.has_legal_moves else None
    return info["legal_moves"] if "legal_moves" in info else None


def normalize_images(image: np.ndarray):
    image_copy = np.array(image)
    normalized_image = image_copy / 255.0
    return normalized_image


def make_stack(item: np.ndarray):
    new_shape = (1,) + item.shape
    return item.reshape(new_shape)


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
    # assert (
    #     "score" in values[0]
    # ), "Values must be a list of dicts with a 'score' key and optionally a max and min scores key. Values was {}".format(
    #     values
    # )
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
                    label="Target Model Weight Update {}".format(i),
                )

    if has_model_updates:
        weight_updates = [value["model_updated"] for value in values]
        for i, weight_update in enumerate(weight_updates):
            if weight_update:
                axs[row][col].axvline(
                    x=i,
                    color="gray",
                    linestyle="dotted",
                    label="Model Weight Update {}".format(i),
                )

    axs[row][col].set_title(
        f"{key} | rolling average: {np.mean(scores[-10:])} | latest: {scores[-1]}"
    )

    axs[row][col].set_xlabel("Game")
    axs[row][col].set_ylabel("Score")

    axs[row][col].set_xlim(1, len(values) + 1)

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
            label="Target Score {}".format(targets[key]),
        )


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
                    label="Target Model Weight Update {}".format(i),
                )

    if has_model_updates:
        weight_updates = [value["model_updated"] for value in values]
        for i, weight_update in enumerate(weight_updates):
            if weight_update:
                axs[row][col].axvline(
                    x=i,
                    color="gray",
                    linestyle="dotted",
                    label="Model Weight Update {}".format(i),
                )

    axs[row][col].set_title(
        f"{key} | rolling average: {np.mean(loss[-10:])} | latest: {loss[-1]}"
    )

    axs[row][col].set_xlabel("Time Step")
    axs[row][col].set_ylabel("Loss")

    axs[row][col].set_xlim(1, len(values) + 1)

    if key in targets and targets[key] is not None:
        axs[row][col].axhline(
            y=targets[key],
            color="r",
            linestyle="dashed",
            label="Target Score {}".format(targets[key]),
        )


def plot_exploitability(
    axs, key: str, values: list[dict], targets: dict, row: int, col: int
):
    default_plot_func(axs, key, values, targets, row, col)


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
    if not os.path.exists("{}/{}".format(dir, model_name)):
        os.makedirs("{}/{}".format(dir, model_name))
    plt.savefig("{}/{}/{}.png".format(dir, model_name, model_name))

    plt.close(fig)


def prepare_kernel_initializers(kernel_initializer: str):
    if kernel_initializer == "glorot_uniform":
        return GlorotUniform(seed=np.random.seed())
    elif kernel_initializer == "glorot_normal":
        return GlorotNormal(seed=np.random.seed())
    elif kernel_initializer == "he_uniform":
        return HeUniform(seed=np.random.seed())
    elif kernel_initializer == "he_normal":
        return HeNormal(seed=np.random.seed())
    elif kernel_initializer == "variance_baseline":
        return VarianceScaling(seed=np.random.seed())
    elif kernel_initializer == "variance_0.1":
        return VarianceScaling(scale=0.1, seed=np.random.seed())
    elif kernel_initializer == "variance_0.3":
        return VarianceScaling(scale=0.3, seed=np.random.seed())
    elif kernel_initializer == "variance_0.8":
        return VarianceScaling(scale=0.8, seed=np.random.seed())
    elif kernel_initializer == "variance_3":
        return VarianceScaling(scale=3, seed=np.random.seed())
    elif kernel_initializer == "variance_5":
        return VarianceScaling(scale=5, seed=np.random.seed())
    elif kernel_initializer == "variance_10":
        return VarianceScaling(scale=10, seed=np.random.seed())
    elif kernel_initializer == "lecun_uniform":
        return LecunUniform(seed=np.random.seed())
    elif kernel_initializer == "lecun_normal":
        return LecunNormal(seed=np.random.seed())
    elif kernel_initializer == "orthogonal":
        return Orthogonal(seed=np.random.seed())

    raise ValueError(f"Invalid kernel initializer: {kernel_initializer}")


def prepare_activations(activation: str):
    # print("Activation to prase: ", activation)
    if activation == "linear":
        return None
    elif activation == "relu":
        return relu
    elif activation == "relu6":
        return relu(max_value=6)
    elif activation == "sigmoid":
        return sigmoid
    elif activation == "softplus":
        return softplus
    elif activation == "soft_sign":
        return softsign
    elif activation == "silu":
        return silu
    elif activation == "swish":
        return swish
    # elif activation == "log_sigmoid":
    #     return log_sigmoid
    elif activation == "hard_sigmoid":
        return hard_sigmoid
    # elif activation == "hard_silu":
    #     return hard_silu
    # elif activation == "hard_swish":
    #     return hard_swish
    # elif activation == "hard_tanh":
    #     return hard_tanh
    elif activation == "elu":
        return elu
    # elif activation == "celu":
    #     return celu
    elif activation == "selu":
        return selu
    elif activation == "gelu":
        return gelu
    # elif activation == "glu":
    #     return glu

    raise ValueError(f"Activation {activation} not recognized")


def epsilon_greedy_policy(q_values: list[float], epsilon: float):
    if np.random.rand() < epsilon:
        return np.random.randint(len(q_values))
    else:
        return np.argmax(q_values)


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
    observation_buffer_shape = []
    observation_buffer_shape += [max_size]
    observation_buffer_shape += list(observation_dimensions)
    return list(observation_buffer_shape)
