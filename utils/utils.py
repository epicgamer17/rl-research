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

import numpy as np


def normalize_policy(policy):
    policy /= tf.reduce_sum(policy, axis=1)
    return policy


def action_mask(actions, legal_moves, num_actions, mask_value=0):
    mask = np.zeros(num_actions, dtype=np.int8)
    mask[legal_moves] = 1
    actions[mask == 0] = mask_value
    return actions


def get_legal_moves(info):
    # info["legal_moves"] if self.config.game.has_legal_moves else None
    return info["legal_moves"] if "legal_moves" in info else None


def normalize_images(image):
    image_copy = np.array(image)
    normalized_image = image_copy / 255.0
    return normalized_image


def make_stack(item):
    new_shape = (1,) + item.shape
    return item.reshape(new_shape)


def update_per_beta(per_beta, per_beta_final, per_beta_steps):
    # could also use an initial per_beta instead of current (multiply below equation by current step)
    per_beta = min(
        per_beta_final, per_beta + (per_beta_final - per_beta) / (per_beta_steps)
    )

    return per_beta


def update_linear_lr_schedule(
    learning_rate, final_value, total_steps, initial_value=None, current_step=None
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


def default_plot_func(axs, key, values, targets, row, col, *args, **kwargs):
    axs[row][col].set_title(
        "{} | rolling average: {}".format(key, np.mean(values[-10:]))
    )
    x = np.arange(0, len(values))
    axs[row][col].plot(x, values)
    if key in targets and targets[key] is not None:
        axs[row][col].axhline(y=targets[key], color="r", linestyle="--")


def plot_scores(axs, key, values, targets, row, col, *args, **kwargs):
    assert (
        "score" in values[0]
    ), "Values must be a list of dicts with a 'score' key and optionally a max and min scores key"

    has_max_scores = "max_score" in values[0]
    has_min_scores = "min_score" in values[0]
    scores = list()
    max_scores = list()
    min_scores = list()

    assert (
        has_max_scores == has_min_scores
    ), "Both max_scores and min_scores must be provided or not provided"

    for score_dict in values:
        scores.append(score_dict["score"])
        if has_max_scores:
            max_scores.append(score_dict["max_score"])
        if has_min_scores:
            min_scores.append(score_dict["min_score"])

    axs[row][col].set_title(
        f"{key} | rolling average: {np.mean(scores[-10:])} | latest test score: {scores[-1]}")

    axs[row][col].set_xlabel("Test Game")
    axs[row][col].set_ylabel("Test Score")

    axs[row][col].set_xlim(0, len(values))

    x = np.arange(len(values))

    axs[row][col].plot(x, scores)
    if has_max_scores:
        axs[row][col].fill_between(x, min_scores, max_scores, alpha=0.5)

    best_fit = np.polyfit(x, scores, 1)
    axs[row][col].plot(
        x,
        best_fit[0] * x + best_fit[1],
        color="g",
        label="Best Fit Line",
        linestyle="dotted",
    )

    if "target_model_weight_update" in kwargs:
        weight_updates = kwargs["target_model_weight_update"]
        for i, weight_update in enumerate(weight_updates):
            axs[row][col].axvline(
                x=weight_update,
                color="r",
                linestyle="dashed",
                label="Target Model Weight Update {}".format(i),
            )

    if "model_weight_update" in kwargs:
        weight_updates = kwargs["model_weight_update"]
        for i, weight_update in enumerate(weight_updates):
            axs[row][col].axvline(
                x=weight_update,
                color="r",
                linestyle="--",
                label="Model Weight Update {}".format(i),
            )

    if key in targets and targets[key] is not None:
        axs[row][col].axhline(
            y=targets[key],
            color="r",
            linestyle="--",
            label="Target Score {}".format(targets[key]),
        )


def plot_loss(axs, key, values, targets, row, col, **kwargs):
    default_plot_func(axs, key, values, targets, row, col)


def plot_exploitability(axs, key, values, targets, row, col, **kwargs):
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


from typing import NamedTuple

class Statictic(NamedTuple):
    score: float
    min: float
    max:float

def plot_graphs(stats: dict, targets, step, frames_seen, time_taken, model_name):
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
    if not os.path.exists("./training_graphs"):
        os.makedirs("./training_graphs")
    if not os.path.exists("./training_graphs/{}".format(model_name)):
        os.makedirs("./training_graphs/{}".format(model_name))
    plt.savefig("./training_graphs/{}/{}.png".format(model_name, model_name))

    plt.close(fig)


def prepare_kernel_initializers(kernel_initializer):
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

