import os
import itertools
import pickle
import pandas
import gymnasium as gym
from hyperopt import tpe, hp, fmin, space_eval
from agent_configs import RainbowConfig
import gc

import sys

sys.path.append("../..")
from dqn.rainbow.rainbow_agent import RainbowAgent
from game_configs import CartPoleConfig


def run_training(args):
    m = RainbowAgent(
        env=args[1],
        config=RainbowConfig(args[0], CartPoleConfig()),
        name="{}_{}".format(args[2], args[1].unwrapped.spec.id),
    )
    m.train()
    print("Training complete")
    return -m.test(num_trials=10, step=5000, dir="./checkpoints/")["score"]


def objective(params):
    gc.collect()
    print("Params: ", params)
    print("Making environments")
    env = gym.make("CartPole-v1", render_mode="rgb_array")

    if os.path.exists("./CartPole_trials.p"):
        trials = pickle.load(open("./CartPole_trials.p", "rb"))
        name = "CartPole_{}".format(len(trials.trials) + 1)
    else:
        name = "CartPole_1"
    params["model_name"] = name
    entry = pandas.DataFrame.from_dict(
        params,
        orient="index",
    ).T

    entry.to_csv(
        "classiccontrol_results.csv",
        mode="a",
        header=False,
    )

    score = -run_training([params, env, name])
    print("parallel programs done")
    return score  # np.mean(scores_list)


widths = [32, 64, 128, 256, 512, 1024]
width_combinations = []

for i in range(0, 5):
    width_combinations.extend(itertools.combinations_with_replacement(widths, i))

from hyperopt import hp


def create_search_space():
    search_space = {
        "kernel_initializer": hp.choice(
            "kernel_initializer",
            [
                "he_uniform",
                "he_normal",
                "glorot_uniform",
                "glorot_normal",
                "orthogonal",
            ],
        ),
        "learning_rate": hp.choice("learning_rate", [10, 5, 2, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]),
        "adam_epsilon": hp.choice("adam_epsilon", [0.3125, 0.03125, 0.003125, 0.0003125]),
        # NORMALIZATION?
        "ema_beta": hp.uniform("ema_beta", 0.95, 0.999),
        "transfer_interval": hp.choice("transfer_interval", [10, 25, 50, 100, 200, 400, 800, 1600, 2000]),
        "replay_interval": hp.choice("replay_interval", [1, 2, 3, 4, 5, 8, 10, 12]),
        "minibatch_size": hp.choice("minibatch_size", [2**i for i in range(4, 8)]),  ###########
        "replay_buffer_size": hp.choice(
            "replay_buffer_size",
            [2000, 3000, 5000, 7500, 10000, 15000, 20000, 25000, 50000],
        ),  #############
        "min_replay_buffer_size": hp.choice(
            "min_replay_buffer_size",
            [125, 250, 375, 500, 625, 750, 875, 1000, 1500, 2000],
        ),  # 125, 250, 375, 500, 625, 750, 875, 1000, 1500, 2000
        "n_step": hp.choice("n_step", [3, 4, 5, 8, 10]),
        "discount_factor": hp.choice("discount_factor", [0.9, 0.99, 0.995, 0.999]),
        "atom_size": hp.choice("atom_size", [51, 61, 71, 81]),  #
        "conv_layers": hp.choice("conv_layers", [[]]),
        "dense_layers_widths": hp.choice("dense_layers_widths", width_combinations),
        "advantage_hidden_layers_widths": hp.choice("advantage_hidden_layers_widths", width_combinations),  #
        "value_hidden_layers_widths": hp.choice("value_hidden_layers_widths", width_combinations),  #
        "training_steps": hp.choice("training_steps", [5000]),
        "per_epsilon": hp.choice("per_epsilon", [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1]),
        "per_alpha": hp.choice("per_alpha", [0.05 * i for i in range(1, 21)]),
        "per_beta": hp.choice("per_beta", [0.05 * i for i in range(1, 21)]),
        "save_intermediate_weights": hp.choice("save_intermediate_weights", [False]),
    }
    initial_best_config = [{}]

    return search_space, initial_best_config


if __name__ == "__main__":
    search_space, initial_best_config = create_search_space()
    max_trials = 2
    trials_step = 2  # how many additional trials to do after loading the last ones

    try:  # try to load an already saved trials object, and increase the max
        trials = pickle.load(open("./classiccontrol_trials.p", "rb"))
        print("Found saved Trials! Loading...")
        max_trials = len(trials.trials) + trials_step
        print("Rerunning from {} trials to {} (+{}) trials".format(len(trials.trials), max_trials, trials_step))
    except:  # create a new trials object and start searching
        # trials = Trials()
        trials = None

    best = fmin(
        fn=objective,  # Objective Function to optimize
        space=search_space,  # Hyperparameter's Search Space
        algo=tpe.suggest,  # Optimization algorithm (representative TPE)
        max_evals=max_trials,  # Number of optimization attempts
        trials=trials,  # Record the results
        # early_stop_fn=no_progress_loss(5, 1),
        trials_save_file="./classiccontrol_trials.p",
        # points_to_evaluate=initial_best_config,
        show_progressbar=False,
    )

    print(best)
    best_trial = space_eval(search_space, best)
    # gc.collect()
