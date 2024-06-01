import os
import pickle
import numpy as np
import pandas
import gymnasium as gym
from hyperopt import tpe, fmin, space_eval, STATUS_OK, STATUS_FAIL
from agent_configs import RainbowConfig
import gc

import sys

sys.path.append("../..")
from dqn.rainbow.rainbow_agent import RainbowAgent

# from rainbow_agent import RainbowAgent
from game_configs import CartPoleConfig

global file_name
global eval_method


def run_training(args):
    m = RainbowAgent(
        env=args[1],
        config=RainbowConfig(args[0], CartPoleConfig()),
        name="{}".format(args[2]),
    )
    m.train()
    print("Training complete")
    if eval_method == "final_score":
        return -m.test(num_trials=10, step=5000, dir=f"./checkpoints/")["score"]
    elif eval_method == "rolling_average":
        return -np.around(
            np.mean([stat_dict["score"] for stat_dict in m.stats["test_score"][-10:]]),
            1,
        )
    elif eval_method == "final_score_rolling_average":
        return (
            -m.test(num_trials=10, step=5000, dir=f"./checkpoints/")["score"]
            - np.around(
                np.mean(
                    [stat_dict["score"] for stat_dict in m.stats["test_score"][-10:-1]]
                ),
                1,
            )
        ) / 2


def objective(params):
    gc.collect()
    print("Params: ", params)
    print("Making environments")
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    if os.path.exists(f"./{file_name}_trials.p"):
        trials = pickle.load(open(f"./{file_name}_trials.p", "rb"))
        name = "{}_{}".format(file_name, len(trials.trials) + 1)
    else:
        name = f"{file_name}_1"
    # name = datetime.datetime.now().timestamp()
    params["model_name"] = name
    entry = pandas.DataFrame.from_dict(
        params,
        orient="index",
    ).T

    entry.to_csv(
        f"./{file_name}_results.csv",
        mode="a",
        header=False,
    )

    status = STATUS_OK
    try:
        # add other illegal hyperparameter combinations here
        assert params["min_replay_buffer_size"] >= params["minibatch_size"]
        assert params["replay_buffer_size"] > params["min_replay_buffer_size"]
    except AssertionError as e:
        status = STATUS_FAIL
        print(f"exited due to invalid hyperparameter combination: {e}")
        return {"status": status, "loss": 0}

    if status != STATUS_FAIL:
        score = run_training([params, env, name])

    # num_workers = len(environments_list)
    # args_list = np.array(
    #     [
    #         [params for env in environments_list],
    #         environments_list,
    #         [name for env in environments_list],
    #     ]
    # ).T
    # with contextlib.closing(multiprocessing.Pool()) as pool:
    #     scores_list = pool.map_async(
    #         globalized_training_func, (args for args in args_list)
    #     ).get()
    #     print(scores_list)
    print("parallel programs done")
    return {"status": status, "loss": score}  # np.mean(scores_list)


if __name__ == "__main__":
    search_space_path, initial_best_config_path = sys.argv[1], sys.argv[2]
    search_space = pickle.load(open(search_space_path, "rb"))
    initial_best_config = pickle.load(open(initial_best_config_path, "rb"))
    file_name = sys.argv[3]
    eval_method = sys.argv[4]
    assert (
        eval_method == "final_score"
        or eval_method == "rolling_average"
        or eval_method == "final_score_rolling_average"
    )
    max_trials = 64
    trials_step = 64  # how many additional trials to do after loading the last ones

    try:  # try to load an already saved trials object, and increase the max
        trials = pickle.load(open(f"./{file_name}_trials.p", "rb"))
        print("Found saved Trials! Loading...")
        max_trials = len(trials.trials) + trials_step
        print(
            "Rerunning from {} trials to {} (+{}) trials".format(
                len(trials.trials), max_trials, trials_step
            )
        )
    except:  # create a new trials object and start searching
        trials = None

    best = fmin(
        fn=objective,  # Objective Function to optimize
        space=search_space,  # Hyperparameter's Search Space
        algo=tpe.suggest,  # Optimization algorithm (representative TPE)
        max_evals=max_trials,  # Number of optimization attempts
        trials=trials,  # Record the results
        # early_stop_fn=no_progress_loss(5, 1),
        trials_save_file=f"./{file_name}_trials.p",
        # points_to_evaluate=initial_best_config,
        show_progressbar=False,
    )

    print(best)
    best_trial = space_eval(search_space, best)
    # gc.collect()
