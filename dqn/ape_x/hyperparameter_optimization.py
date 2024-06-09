import os
import pickle
import subprocess
from subprocess import Popen
import pathlib
from pathlib import Path
import multiprocessing
import time
import gc
import argparse

# os.environ["OMP_NUM_THREADS"] = f"{1}"
# os.environ['TF_NUM_INTEROP_THREADS'] = f"{1}"
# os.environ['TF_NUM_INTRAOP_THREADS'] = f"{1}"

import tensorflow as tf
import numpy as np
import pandas as pd
import gymnasium as gym
from hyperopt import tpe, hp, fmin, space_eval, STATUS_OK, STATUS_FAIL

from agent_configs import ApeXActorConfig, ApeXLearnerConfig
from agent_configs import ReplayBufferConfig
from game_configs.cartpole_config import CartPoleConfig
from learner import ApeXLearner

# tf.config.threading.set_intra_op_parallelism_threads(1)
# tf.config.threading.set_inter_op_parallelism_threads(1)


SIGTERM = 15
SSH_USERNAME = "ehuang"

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


ctx = multiprocessing.get_context("fork")

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler("hyperopt.log", mode="w")
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(message)s"))

logger.addHandler(fh)
logger.addHandler(ch)

logging.basicConfig(
    level=logging.DEBUG,
    handlers=[fh, ch],
    format="%(asctime)s %(name)s %(threadName)s %(levelname)s: %(message)s",
)


def get_current_host():
    host_output = subprocess.check_output("hostname | sed 's/[^0-9]*//'", shell=True)
    logger.info(f"current host: {host_output.strip()}")
    try:
        current_host = int(host_output.strip())
    except ValueError:
        # assume we are not on open-gpu-x
        current_host = 0
    return current_host


def run_training(config, env: gym.Env, name):
    global SSH_USERNAME
    print("=================== Run Training ======================")

    distributed_config_placeholder = {
        "rank": 0,
        "worker_name": '',
        "world_size": 0,
        "rpc_port": 0,
        "pg_port": 0,
        "master_addr": '',
        "replay_addr": '',
        "storage_addr": '',
    }

    # combined learner and actor config
    conf = (config | distributed_config_placeholder) | {
        "num_actors": 1,
        "training_steps": 2000,
        # save on mimi disk quota
        "save_intermediate_weights": False,
        # set for learner, will be overwritten by cli flags in actors
        "noisy_sigma": config["learner_noisy_sigma"],
    }

    generated_dir = "generated"
    os.makedirs(generated_dir, exist_ok=True)

    learner_config_path = Path(Path.cwd(), "configs", "learner_config.yaml")
    actor_config_path = Path(Path.cwd(), "configs", "actor_config.yaml")

    hosts_file_path = Path(Path.cwd(), generated_dir, "hosts.yaml")
    learner_output_path = Path(Path.cwd(), generated_dir, "learner_output.yaml")
    actor_output_path = Path(Path.cwd(), generated_dir, "actor_output.yaml")
    distributed_output_path = Path(Path.cwd(), generated_dir, "distributed_output.yaml")

    learner_config = ApeXLearnerConfig(conf, game_config=CartPoleConfig())
    actor_config = ApeXActorConfig(conf, game_config=CartPoleConfig())

    learner_config.dump(learner_config_path)
    actor_config.dump(actor_config_path)

    current_host = get_current_host()

    machines = config["num_actors"] +3 # +mongo, replay, spectator
    cmd = f"./bin/find_servers -exclude={current_host} -output={hosts_file_path} -ssh_username={SSH_USERNAME} -machines={machines}"
    print("running cmd:", cmd)
    proc = subprocess.run(cmd.split(" "), capture_output=True, text=True)

    # not enough actors to run or other issue generating hosts
    if proc.returncode != 0:
        return {"status": STATUS_FAIL, "loss": 0}

    cmd = f'./bin/write_configs -learner_config={learner_config_path} -actor_config={actor_config_path} -hosts_file={hosts_file_path} -learner_output={learner_output_path} -actor_output={actor_output_path} -distributed_output={distributed_output_path} -ssh_username={SSH_USERNAME} -actors_initial_sigma={config["actors_initial_sigma"]} -actors_sigma_alpha={config["actors_sigma_alpha"]}'
    print("running cmd: ", cmd)
    out = subprocess.run(cmd.split(" "), capture_output=True, text=True)
    logger.debug(f"write_configs stdout: {out.stdout}")
    logger.debug(f"write_configs stderr: {out.stderr}")
    try:
        cmd = f"./bin/hyperopt -distributed_config={distributed_output_path} -learner_name={name} -ssh_username={SSH_USERNAME}"
        print("running cmd:", cmd)
        go_proc = Popen(cmd.split(" "), stdin=subprocess.PIPE, text=True)
        time.sleep(5)

        learner_generated_config = ApeXLearnerConfig.load(learner_output_path)
        learner = ApeXLearner(env, learner_generated_config, name=name)
        logger.info("        === Running learner")
        learner.run()
        logger.info("Training complete")
        loss = -learner.test(num_trials=10, step=0)
        return {"status": STATUS_OK, "loss": loss}
    except KeyboardInterrupt:
        logger.info("learner interrupted, cleaning up")
        loss = -learner.test(num_trials=10, step=0)
        return {"status": STATUS_OK, "loss": loss}
    except Exception as e:
        logger.exception(f"learner failed due to error {e}")
        return {
            "status": STATUS_FAIL,
            "loss": 100000,
        }  # make this high since some games have negative rewards (mountain car and acrobot) and 0 would actually be a perfect score
    finally:
        go_proc.send_signal(SIGTERM)
        while go_proc.poll() == None:
            logger.debug("process not terminated yet, waiting")
            time.sleep(1)
        logger.info("cleaning up finished")


def objective(params):
    logger.info(f"Params: {params}")
    logger.info("Making environments")
    environments_list = [
        gym.make("CartPole-v1", render_mode="rgb_array"),
        # gym.make("Acrobot-v1", render_mode="rgb_array"),
        # gym.make("MountainCar-v0", render_mode="rgb_array"),
    ]

    if os.path.exists("./classiccontrol_trials.p"):
        trials = pickle.load(open("./classiccontrol_trials.p", "rb"))
        name = "classiccontrol_{}".format(len(trials.trials) + 1)
    else:
        name = "classiccontrol_1"
    # name = datetime.datetime.now().timestamp()
    params["model_name"] = name
    entry = pd.DataFrame.from_dict(
        params,
        orient="index",
    ).T

    entry.to_csv(
        "classiccontrol_results.csv",
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
        logger.info(f"exited due to invalid hyperparameter combination: {e}")
        return {"status": status, "loss": 0}

    if status != STATUS_FAIL:
        loss_list = list()
        for env in environments_list:
            res_dict = run_training(params, env, name)
            if res_dict["status"] == STATUS_FAIL:
                return res_dict
            else:
                loss_list.append(res_dict["loss"])

    print("training done with loss {} and status {}".format(np.sum(loss_list), status))
    return {"loss": np.sum(loss_list), "status": status}


def create_search_space():
    search_space = {
        "activation": hp.choice("activation", ["relu"]),
        "kernel_initializer": hp.choice(
            "kernel_initializer",
            [
                "he_uniform",
                "he_normal",
                "glorot_uniform",
                "glorot_normal",
                # "lecun_uniform",
                # "lecun_normal",
                "orthogonal",
                "variance_baseline",
                "variance_0.1",
                "variance_0.3",
                "variance_0.8",
                "variance_3",
                "variance_5",
                "variance_10",
            ],
        ),
        "learning_rate": hp.choice(
            "learning_rate", [0.1, 0.01, 0.001, 0.0001, 0.00001]
        ),  #
        "adam_epsilon": hp.choice(
            "adam_epsilon",
            [0.3125, 0.03125, 0.003125, 0.0003125, 0.00003125, 0.000003125],
        ),
        "clipnorm": hp.choice(
            "clipnorm", [None, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]
        ),
        # NORMALIZATION?
        "transfer_interval": hp.choice(
            "transfer_interval", [10, 25, 50, 100, 200, 400, 800, 1600, 2000]
        ),
        "minibatch_size": hp.choice("minibatch_size", [2**i for i in range(3, 8)]),
        "replay_buffer_size": hp.choice(
            "replay_buffer_size",
            [2000, 3000, 5000, 7500, 10000, 15000, 20000, 25000, 50000, 100000],
        ),
        "actor_buffer_size": hp.choice(
            "actor_buffer_size", [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        ),
        "min_replay_buffer_size": hp.choice(
            "min_replay_buffer_size",
            [125, 250, 375, 500, 625, 750, 875, 1000, 1500, 2000],
        ),
        "n_step": hp.choice("n_step", [1, 2, 3, 4, 5, 8, 10]),
        "discount_factor": hp.choice("discount_factor", [0.9, 0.99, 0.995, 0.999]),
        "atom_size": hp.choice("atom_size", [41, 51, 61, 71, 81]),  #
        "width": hp.choice("width", [32, 64, 128, 256, 512, 1024]),
        "dense_layers": hp.choice("dense_layers", [0, 1, 2, 3, 4]),
        # REWARD CLIPPING
        "loss_function": hp.choice(
            "loss_function",
            [tf.keras.losses.CategoricalCrossentropy(), tf.keras.losses.KLDivergence()],
        ),
        "advantage_hidden_layers": hp.choice(
            "advantage_hidden_layers", [0, 1, 2, 3, 4]
        ),  #
        "value_hidden_layers": hp.choice("value_hidden_layers", [0, 1, 2, 3, 4]),  #
        "per_epsilon": hp.choice(
            "per_epsilon", [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1]
        ),
        "per_alpha": hp.choice("per_alpha", [0.05 * i for i in range(0, 21)]),
        "per_beta": hp.choice("per_beta", [0.05 * i for i in range(1, 21)]),
        "push_params_interval": hp.choice(
            "push_params_interval", [2, 3, 4, 5, 8, 10, 12]
        ),
        "updates_queue_size": hp.choice(
            "updates_queue_size", [1, 2, 3, 4, 5, 8, 10, 12]
        ),
        "samples_queue_size": hp.choice(
            "samples_queue_size", [1, 2, 3, 4, 5, 8, 10, 12]
        ),
        # "remove_old_experiences_interval": hp.choice(
        #     "remove_old_experiences_interval", [1000, 2000, 3000, 4000, 5000, 8000, 10000]
        # ),
        "poll_params_interval": hp.choice("poll_params_interval", [50, 100, 200, 300]),
        "actors_initial_sigma": hp.choice(
            "actors_initial_sigma", [0.1 * i for i in range(1, 10)]
        ),
        "actors_sigma_alpha": hp.choice(
            "actors_sigma_alpha", [i for i in range(1, 20)]
        ),
        "learner_noisy_sigma": hp.choice(
            "learner_noisy_sigma", [0.1 * i for i in range(1, 10)]
        ),
        "num_actors": hp.choice("num_actors", [i for i in range(1, 16 + 1)]),
        # 'per_beta_increase': hp.uniform('per_beta_increase', 0, 0.015),
        # 'search_max_depth': 5,
        # 'search_max_time': 10,
    }
    initial_best_config = []

    return search_space, initial_best_config


def main():
    global SSH_USERNAME
    parser = argparse.ArgumentParser()
    parser.add_argument("ssh_username", type=str, help="an integer to be summed")
    args = parser.parse_args()

    SSH_USERNAME = args.ssh_username

    search_space, initial_best_config = create_search_space()
    max_trials = 64
    trials_step = 64  # how many additional trials to do after loading the last ones

    try:  # try to load an already saved trials object, and increase the max
        trials = pickle.load(open("./classiccontrol_trials.p", "rb"))
        logger.info("Found saved Trials! Loading...")
        max_trials = max(len(trials.trials) + trials_step, max_trials + trials_step)
        logger.info(
            f"Rerunning from {len(trials.trials)} trials to {max_trials} (+{trials_step}) trials"
        )
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
        points_to_evaluate=initial_best_config,
        show_progressbar=False,
    )

    logger.info(best)
    best_trial = space_eval(search_space, best)


# objective function - needs to launch

if __name__ == "__main__":
    main()
