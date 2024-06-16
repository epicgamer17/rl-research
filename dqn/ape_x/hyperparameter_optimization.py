import os
import pickle
import subprocess
from subprocess import Popen
from pathlib import Path
import time
import argparse

import numpy as np
import pandas as pd
import gymnasium as gym
from hyperopt import tpe, hp, fmin, space_eval, STATUS_OK, STATUS_FAIL

from agent_configs import ApeXActorConfig, ApeXLearnerConfig
from game_configs.cartpole_config import CartPoleConfig
from learner import ApeXLearner
import utils

SIGTERM = 15
SSH_USERNAME = "ehuang"
def recv_stop_msg(msg):
    global stop_chan
    stop_chan.put(msg)


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
        "worker_name": "",
        "world_size": 0,
        "rpc_port": 0,
        "master_addr": "",
        "replay_addr": "",
        "storage_addr": "",
    }

    # combined learner and actor config
    conf = (config | distributed_config_placeholder) | {
        "training_steps": 10000,
        # save on mimi disk quota
        "save_intermediate_weights": False,
        # set for learner, will be overwritten by learner when creating actors
        "noisy_sigma": 0# config["learner_noisy_sigma"],
    }

    generated_dir = "generated"
    os.makedirs(generated_dir, exist_ok=True)

    learner_config_path = Path(Path.cwd(), "configs", "learner_config.yaml")
    actor_config_path = Path(Path.cwd(), "configs", "actor_config.yaml")

    hosts_file_path = Path(Path.cwd(), generated_dir, "hosts.yaml")
    learner_output_path = Path(Path.cwd(), generated_dir, "learner_output.yaml")
    actor_output_path = Path(Path.cwd(), generated_dir, "actor_output.yaml")
    distributed_output_path = Path(Path.cwd(), generated_dir, "distributed_output.yaml")

    actor_config = ApeXActorConfig(conf, game_config=CartPoleConfig())
    actor_config.dump(actor_config_path)

    conf["distributed_actor_config_file"] = str(actor_config_path.absolute())

    learner_config = ApeXLearnerConfig(conf, game_config=CartPoleConfig())
    learner_config.dump(learner_config_path)

    current_host = get_current_host()

    machines = learner_config.num_actors + 3
    cmd = f"./bin/find_servers -exclude={current_host} -output={hosts_file_path} -ssh_username={SSH_USERNAME} -machines={machines}"
    print("running cmd:", cmd)
    proc = subprocess.run(cmd.split(" "), capture_output=True, text=True)

    # not enough actors to run or other issue generating hosts
    if proc.returncode != 0:
        return {"status": STATUS_FAIL, "loss": 100000}

    cmd = f"./bin/write_configs -learner_config={learner_config_path} -actor_config={actor_config_path} -hosts_file={hosts_file_path} -learner_output={learner_output_path} -actor_output={actor_output_path} -distributed_output={distributed_output_path} -ssh_username={SSH_USERNAME}"
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
        learner = ApeXLearner(env, learner_generated_config, name=name, stop_fn=recv_stop_msg)
        logger.info("        === Running learner")
        learner.run()
        logger.info("Training complete")
        loss = -learner.test(num_trials=10, step=0)["score"]
        return {"status": STATUS_OK, "loss": loss}
    except KeyboardInterrupt:
        logger.info("learner interrupted, cleaning up")
        loss = -learner.test(num_trials=10, step=0)["score"]
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


from hyperopt.pyll.base import scope
import math


def create_search_space():
    search_space = {
        # "activation": hp.choice("activation", ["relu"]),
        "kernel_initializer": hp.choice(
            "kernel_initializer",
            [
                "pytorch_default",
                "he_uniform",
                "he_normal",
                "glorot_uniform",
                "glorot_normal",
                "orthogonal",
                "variance_baseline",
                "variance_0.1",
                "variance_0.3",
                "variance_0.8",
                "variance_3",
                "variance_5",
                "variance_10",
                # "lecun_uniform",
                # "lecun_normal",
            ],
        ),
        #### not actually used in apex, just to prevent the configs from throwing errors
        "loss_function": hp.choice(
            "loss_function",
            [utils.CategoricalCrossentropyLoss()],#, utils.KLDivergenceLoss()],
        ),
        ###
        "learning_rate": hp.loguniform("learning_rate", math.log(1e-5), math.log(1e-1)),
        "adam_epsilon": hp.loguniform("adam_epsilon", math.log(1e-9), math.log(1e-6)),
        "clipnorm": hp.loguniform("clipnorm", math.log(0.1), math.log(1000)),
        "transfer_interval": scope.int(hp.quniform("transfer_interval", 20, 1000, 20)),
        "minibatch_size": scope.int(
            hp.loguniform("minibatch_size", math.log(2**3), math.log(2**10))
        ),
        "replay_buffer_size": scope.int(
            hp.loguniform("replay_buffer_size", math.log(1e5), math.log(1e7))
        ),
        "actor_buffer_size": scope.int(hp.quniform("actor_buffer_size", 100, 1000, 25)),
        "min_replay_buffer_size": scope.int(
            hp.quniform("min_replay_buffer_size", 100, 2000, 100)
        ),
        "n_step": scope.int(hp.quniform("n_step", 3, 10, 1)),
        "discount_factor": hp.loguniform(
            "discount_factor", math.log(0.9), math.log(0.999)
        ),
        "atom_size": hp.choice("atom_size", [41, 51, 61, 71, 81]),
        "dense_layers_widths": hp.choice(
            "dense_layers_widths", [[32], [64], [128], [256], [512], [1024]]
        ),
        "advantage_hidden_layers_widths": hp.choice(
            "advantage_hidden_layers_widths", [[32], [64], [128], [256], [512], [1024]]
        ),
        "value_hidden_layers_widths": hp.choice(
            "value_hidden_layers_widths", [[32], [64], [128], [256], [512], [1024]]
        ),
        "per_epsilon": hp.loguniform("per_epsilon", math.log(1e-8), math.log(1e-1)),
        "per_alpha": hp.quniform("per_alpha", 0.05, 1, 0.05),
        "per_beta": hp.quniform("per_beta", 0.05, 1, 0.05),
        "push_params_interval": scope.int(hp.quniform("push_params_interval", 2, 12, 1)),
        "updates_queue_size": scope.int(hp.quniform("updates_queue_size", 2, 12, 1)),
        "samples_queue_size": scope.int(hp.quniform("samples_queue_size", 2, 12, 1)),
        "poll_params_interval": scope.int(
            hp.quniform("poll_params_interval", 50, 500, 10)
        ),
        "num_actors": scope.int(hp.quniform("num_actors", 5, 17, 1)),
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
