import os
import time

# os.environ["OMP_NUM_THREADS"] = f"{1}"
# os.environ['TF_NUM_INTEROP_THREADS'] = f"{1}"
# os.environ['TF_NUM_INTRAOP_THREADS'] = f"{1}"

from agent_configs import ApeXActorConfig, ApeXLearnerConfig
from agent_configs import ReplayBufferConfig
from game_configs.cartpole_config import CartPoleConfig
import tensorflow as tf

# tf.config.threading.set_intra_op_parallelism_threads(1)
# tf.config.threading.set_inter_op_parallelism_threads(1)

import gc

from learner import ApeXLearner

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

import multiprocessing
import sys
import pandas
import pickle
import gymnasium as gym
from hyperopt import tpe, hp, fmin, space_eval
import subprocess
from subprocess import Popen
import pathlib
import contextlib
import numpy as np

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


def run_training(config, env: gym.Env, name):
    print("=================== Run Training ======================")
    with open(f"{pathlib.Path.home()}/mongodb/mongodb_admin_password", "r") as f:
        password = f.read()

    distributed_config = {
        "actor_replay_port": 5554,
        "learner_replay_port": 5555,
        "replay_addr": "127.0.0.1",
        "storage_hostname": "127.0.0.1",
        "storage_port": 5553,
        "storage_username": "ezra",
        "storage_password": password.strip(),
    }

    conf = {
        **config,
        **distributed_config,
        "num_actors": 1,
        "training_steps": 100,
    }

    replay_conf = dict(
        observation_dimensions=env.observation_space.shape,
        max_size=conf["replay_buffer_size"],
        min_size=conf["min_replay_buffer_size"],
        batch_size=conf["minibatch_size"],
        max_priority=1.0,
        per_alpha=config["per_alpha"],
        n_step=1,  # we don't need n-step because the actors give n-step transitions already
        gamma=config["discount_factor"],
    )

    learner_config = ApeXLearnerConfig(conf, game_config=CartPoleConfig())
    actor_config = ApeXActorConfig(conf, game_config=CartPoleConfig())

    replay_conf = ReplayBufferConfig(replay_conf)

    actor_config_filename = "configs/apex_actor_config.yaml"
    actor_config.dump(actor_config_filename)

    replay_config_filename = "configs/replay_config.yaml"
    replay_conf.dump(replay_config_filename)

    def make_replay_args():
        learner_port = distributed_config["learner_replay_port"]
        actors_port = distributed_config["actor_replay_port"]
        args = f"distributed_replay_buffer.py --learner_port {learner_port} --actors_port {actors_port} --config_file {replay_config_filename}".split(
            " "
        )
        return [sys.executable, *args]

    def make_mongo_args():
        mongo_executable = subprocess.check_output(["which", "mongod"]).rstrip(b"\n")
        mongo_port = distributed_config["storage_port"]
        args = f"--dbpath /home/ezrahuang/mongodb/data --logpath /home/ezrahuang/mongodb/logs/mongod.log --port {mongo_port} --auth".split(
            " "
        )
        return [mongo_executable, *args]

    def make_actor_i_args(i):
        epsilon = 0  # not used yet
        args = f"main_actor.py --name {i} --config_file {actor_config_filename}".split(
            " "
        )
        return [sys.executable, *args]

    def make_spectator_actor():
        args = f"main_actor.py --name spectator --config_file {actor_config_filename} --spectator".split(
            " "
        )
        p = Popen([sys.executable, *args])
        return p

    with contextlib.closing(ctx.Pool()) as pool:
        # pool.apply(Popen, (make_mongo_args(),))
        Popen(make_mongo_args())
        replay_proc = Popen(make_replay_args())
        time.sleep(5)

        actor_procs: list[Popen] = list()
        for i in range(conf["num_actors"]):
            actor_procs.append(Popen(make_actor_i_args(i)))

        learner = ApeXLearner(env, learner_config, name=name)
        logger.info("        === Running learner")
        learner.run()
        logger.info("        === Learner done")

        logger.info("        === Terminiating pool")
        replay_proc.terminate()
        for proc in actor_procs:
            proc.terminate()
        pool.terminate()

    logger.info("Training complete")
    return -learner.test(num_trials=10, step=0)


def objective(params):
    gc.collect()
    print("Params: ", params)
    print("Making environments")
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
    entry = pandas.DataFrame.from_dict(
        params,
        orient="index",
    ).T

    entry.to_csv(
        "classiccontrol_results.csv",
        mode="a",
        header=False,
    )

    args_list = np.array(
        [
            [params for env in environments_list],
            environments_list,
            [name for env in environments_list],
        ]
    ).T

    scores_list = list()
    for args in args_list:
        score = run_training(args[0], args[1], args[2])
        print("score: ", score)
        scores_list.append(score)

    print("programs done")
    return np.sum(scores_list)


from hyperopt import hp
import tensorflow as tf
from hyperopt.pyll import scope


def create_search_space():
    search_space = {
        "activation": hp.choice(
            "activation",
            [
                # "linear",
                "relu",
                # 'relu6',
                # "sigmoid",
                # "softplus",
                # "soft_sign",
                # "silu",
                # "swish",
                # "log_sigmoid",
                # "hard_sigmoid",
                # 'hard_silu',
                # 'hard_swish',
                # 'hard_tanh',
                # "elu",
                # 'celu',
                # "selu",
                # "gelu",
                # 'glu'
            ],
        ),
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
            "learning_rate", [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
        ),  #
        "adam_epsilon": hp.choice(
            "adam_epsilon",
            [0.5, 0.3125, 0.03125, 0.003125, 0.0003125, 0.00003125, 0.000003125],
        ),
        "clipnorm": hp.choice(
            "clipnorm", [None, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]
        ),
        # NORMALIZATION?
        "transfer_interval": hp.choice(
            "transfer_interval", [10, 25, 50, 100, 200, 400, 800, 1600, 2000]
        ),
        "replay_interval": hp.choice("replay_interval", [1, 2, 3, 4, 5, 8, 10, 12]),
        "minibatch_size": hp.choice(
            "minibatch_size", [2**i for i in range(3, 8)]
        ),  ###########
        "replay_buffer_size": hp.choice(
            "replay_buffer_size",
            [2000, 3000, 5000, 7500, 10000, 15000, 20000, 25000, 50000, 100000],
        ),  #############
        "actor_buffer_size": hp.choice(
            "actor_buffer_size", [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        ),
        "min_replay_buffer_size": hp.choice(
            "min_replay_buffer_size",
            [0, 125, 250, 375, 500, 625, 750, 875, 1000, 1500, 2000],
        ),  # 125, 250, 375, 500, 625, 750, 875, 1000, 1500, 2000
        "n_step": hp.choice("n_step", [1, 2, 3, 4, 5, 8, 10]),
        "discount_factor": hp.choice(
            "discount_factor", [0.1, 0.5, 0.9, 0.99, 0.995, 0.999]
        ),
        "atom_size": hp.choice("atom_size", [11, 21, 31, 41, 51, 61, 71, 81]),  #
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
        "actors_sigma_alpha": hp.choice("actors_sigma_alpha", [range(1, 20)]),
        "learner_noisy_sigma": hp.choice(
            "learner_noisy_sigma", [0.1 * i for i in range(1, 10)]
        ),
        # 'per_beta_increase': hp.uniform('per_beta_increase', 0, 0.015),
        # 'search_max_depth': 5,
        # 'search_max_time': 10,
    }
    initial_best_config = []

    return search_space, initial_best_config


def main():
    search_space, initial_best_config = create_search_space()
    max_trials = 16
    trials_step = 1  # how many additional trials to do after loading the last ones

    try:  # try to load an already saved trials object, and increase the max
        trials = pickle.load(open("./classiccontrol_trials.p", "rb"))
        print("Found saved Trials! Loading...")
        max_trials = len(trials.trials) + trials_step
        print(
            "Rerunning from {} trials to {} (+{}) trials".format(
                len(trials.trials), max_trials, trials_step
            )
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

    print(best)
    best_trial = space_eval(search_space, best)
    gc.collect()


# objective function - needs to launch

if __name__ == "__main__":
    main()