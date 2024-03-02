import os
# os.environ["OMP_NUM_THREADS"] = f"{1}"
# os.environ['TF_NUM_INTEROP_THREADS'] = f"{1}"
# os.environ['TF_NUM_INTRAOP_THREADS'] = f"{1}"

import tensorflow as tf
# tf.config.threading.set_intra_op_parallelism_threads(1)
# tf.config.threading.set_inter_op_parallelism_threads(1)

import gc

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

import concurrent.futures
import multiprocessing
from multiprocessing import Pool
import sys
# from rainbow_dqn import RainbowDQN
import numpy as np
import pandas
import pickle
import gymnasium as gym
from hyperopt import tpe, hp, fmin, space_eval
import contextlib
from rainbow_dqn import RainbowDQN

# MAGIC CODE DO NOT TOUCH
def globalize(func):
    def result(*args, **kwargs):
        return func(*args, **kwargs)
    result.__name__ = result.__qualname__ = (
        os.path.abspath(func.__code__.co_filename).replace('.', '') + '\0' +
        str(func.__code__.co_firstlineno))
    setattr(sys.modules[result.__module__], result.__name__, result)
    return result

def make_func():
    def run_training(args):
        gc.collect()
        m = RainbowDQN(
            env=args[1],
            model_name="{}_{}".format(args[2], args[1].unwrapped.spec.id),
            config=args[0]
        )
        m.train()
        print("Training complete")
        return -m.test(num_trials=10)
    return run_training

func1 = globalize(make_func())

def objective(params):
    gc.collect()
    environments_list = [gym.make("CartPole-v1", render_mode="rgb_array"), gym.make("Acrobot-v1", render_mode="rgb_array"), gym.make("MountainCar-v0", render_mode="rgb_array"), ]

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

    num_workers = len(environments_list)
    args_list = np.array([[params for env in environments_list], environments_list, [name for env in environments_list]]).T
    with contextlib.closing(multiprocessing.Pool(8)) as pool:
        # print(pool.map(func1, range(10)))
        scores_list = pool.map_async(func1, (args for args in args_list)).get()
        print(scores_list)
    print("parallel programs done")
    # with concurrent.futures.ThreadPoolExecutor(8) as executor:
    #     # print(list(executor.map(func1, range(10))))
    #     scores_list = list(executor.map(func1, (args for args in args_list)))
    #     print(list(executor.map(func1, (args for args in args_list))))
    return np.sum(scores_list)



func2 = globalize(objective)

from hyperopt import hp
import tensorflow as tf
from hyperopt.pyll import scope

def create_search_space():
    search_space = {
        'activation': hp.choice('activation', [
            'linear',
            'relu',
            # 'relu6',
            'sigmoid',
            'softplus',
            'soft_sign',
            'silu',
            'swish',
            'log_sigmoid',
            'hard_sigmoid',
            # 'hard_silu',
            # 'hard_swish',
            # 'hard_tanh',
            'elu',
            # 'celu',
            'selu',
            'gelu',
            # 'glu'
        ]),
        'kernel_initializer': hp.choice('kernel_initializer', ['he_uniform', 'he_normal', 'glorot_uniform', 'glorot_normal', 'lecun_uniform', 'lecun_normal', 'orthogonal', 'variance_baseline', 'variance_0.1', 'variance_0.3', 'variance_0.8', 'variance_3', 'variance_5', 'variance_10']),
        'optimizer_function': hp.choice('optimizer_function', [tf.keras.optimizers.legacy.Adam]), # NO SGD OR RMSPROP FOR NOW SINCE IT IS FOR RAINBOW DQN
        'learning_rate': hp.choice('learning_rate', [10, 5, 2, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]), #
        'adam_epsilon': hp.choice('adam_epsilon', [1, 0.5, 0.3125, 0.03125, 0.003125, 0.0003125, 0.00003125, 0.000003125]),
        # NORMALIZATION?
        'soft_update': hp.choice('soft_update', [False]), # seems to always be false, we can try it with tru
        'ema_beta': hp.uniform('ema_beta', 0.95, 0.999),
        'transfer_frequency': hp.choice('transfer_frequency', [10, 25, 50, 100, 200, 400, 800, 1600, 2000]),
        'replay_period': hp.choice('replay_period', [1, 2, 3, 4, 5, 8, 10, 12]),
        'replay_batch_size': hp.choice('replay_batch_size', [2 ** i for i in range(0, 8)]), ###########
        'memory_size': hp.choice('memory_size', [2000, 3000, 5000, 7500, 10000, 15000, 20000, 25000]), #############
        'min_memory_size': hp.choice('min_memory_size', [0]), #125, 250, 375, 500, 625, 750, 875, 1000, 1500, 2000
        'n_step': hp.choice('n_step', [1, 2, 3, 4, 5, 8, 10]),
        'discount_factor': hp.choice('discount_factor', [0.1, 0.5, 0.9, 0.99, 0.995, 0.999]),
        'atom_size': hp.choice('atom_size', [11, 21, 31, 41, 51, 61, 71, 81]), #
        'conv_layers': hp.choice('conv_layers', [[]]),
        'conv_layers_noisy': hp.choice('conv_layers_noisy', [False]),
        'width': hp.choice('width', [32, 64, 128, 256, 512, 1024]),
        'dense_layers': hp.choice('dense_layers', [0, 1, 2, 3, 4]),
        'dense_layers_noisy': hp.choice('dense_layers_noisy', [True]), # i think this is always true for rainbow
        # REWARD CLIPPING
        'noisy_sigma': hp.choice('noisy_sigma', [0.5]), #
        'loss_function': hp.choice('loss_function', [tf.keras.losses.CategoricalCrossentropy(), tf.keras.losses.KLDivergence()]),
        'dueling': hp.choice('dueling', [True]),
        'advantage_hidden_layers': hp.choice('advantage_hidden_layers', [0, 1, 2, 3, 4]), #
        'value_hidden_layers': hp.choice('value_hidden_layers', [0, 1, 2, 3, 4]), #
        'num_training_steps': hp.choice('num_training_steps', [25000]),
        'per_epsilon': hp.choice('per_epsilon', [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1]),
        'per_alpha': hp.choice('per_alpha', [0.05 * i for i in range(0, 21)]),
        'per_beta': hp.choice('per_beta', [0.05 * i for i in range(1, 21)]),
        # 'per_beta_increase': hp.uniform('per_beta_increase', 0, 0.015),
        'v_min': hp.choice('v_min', [-500.0]), # MIN GAME SCORE
        'v_max': hp.choice('v_max', [500.0]), # MAX GAME SCORE
        # 'search_max_depth': 5,
        # 'search_max_time': 10,
    }

    # search_space = {
    #     'conv_layers': hp.choice('conv_layers', [[]]),
    #     'conv_layers_noisy': hp.choice('conv_layers_noisy', [False]), #
    #     'dense_layers': hp.choice('dense_layers', [
    #         [], [32], [32, 32], [32, 32, 32], [64], [64, 64], [64, 64, 64], [128], [128, 128], [128, 128, 128], [256], [256, 256], [256, 256, 256], [512], [512, 512], [512, 512, 512], [1024], [1024, 1024], [1024, 1024, 1024]
    #     ]),
    #     'dense_layers_noisy': hp.choice('dense_layers_noisy', [True, False]), #
    #     'noisy_sigma': hp.uniform('noisy_sigma', 0.1, 1.0), #
    #     'activation': hp.choice('activation', ['relu', 'sigmoid']),
    #     'kernel_initializer': hp.choice('kernel_initializer', ['he_uniform', 'he_normal', 'glorot_uniform', 'glorot_normal', 'lecun_uniform', 'lecun_normal']),
    #     'optimizer_function': hp.choice('optimizer_function', [tf.keras.optimizers.legacy.Adam, tf.keras.optimizers.legacy.SGD]),
    #     'adam_epsilon': hp.uniform('adam_epsilon', 0.00001, 1.0), #
    #     'learning_rate': hp.uniform('learning_rate', 0.000001, 0.0025),
    #     'loss_function': hp.choice('loss_function', [tf.keras.losses.CategoricalCrossentropy(), tf.keras.losses.KLDivergence()]),
    #     'dueling': hp.choice('dueling', [True]),
    #     'advantage_hidden_layers': [], #
    #     'value_hidden_layers': [], #
    #     'num_training_steps': scope.int(hp.quniform('num_training_steps', 5000, 10000, 100)), #
    #     'discount_factor': hp.uniform('discount_factor', 0.85, 0.999),
    #     'soft_update': hp.choice('soft_update', [True, False]),
    #     'ema_beta': hp.uniform('ema_beta', 0.95, 0.999),
    #     'transfer_frequency': scope.int(hp.uniform('transfer_frequency', 0, 200)), #
    #     'per_epsilon': hp.uniform('per_epsilon', 0.000001, 0.1),
    #     'per_alpha': hp.choice('per_alpha', [0.05 * i for i in range(0, 21)]),
    #     'per_beta': hp.choice('per_beta', [0.05 * i for i in range(0, 21)]),
    #     'per_beta_increase': hp.uniform('per_beta_increase', 0, 0.015),
    #     'replay_batch_size': hp.choice('replay_batch_size', [2 ** i for i in range(0, 8)]),
    #     'replay_period': scope.int(hp.quniform('replay_period', 1, 10, 1)),
    #     'memory_size': scope.int(hp.quniform('memory_size', 1, 100000, 1)), #
    #     'min_memory_size': hp.uniform('min_memory_size', 0, 3000), #
    #     'n_step': scope.int(hp.quniform('n_step', 1, 5, 1)), #
    #     'v_min': hp.choice('v_min', [0.0]), #
    #     'v_max': scope.int(hp.quniform('v_max', 100.0, 1000.0, 100.0)),
    #     'atom_size': hp.choice('atom_size', [51]), #
    #     # 'search_max_depth': 5,
    #     # 'search_max_time': 10,
    # }
    # Current best setting
    # For hp.uniform specify the exact value
    # For hp.choice specify the index (0 based indexing) in the array
    initial_best_config = []

    return search_space, initial_best_config



if __name__ == '__main__':
    config = {
        'activation': 'relu',
        'kernel_initializer': 'he_uniform',
        'optimizer_function': tf.keras.optimizers.legacy.Adam, # NO SGD OR RMSPROP FOR NOW SINCE IT IS FOR RAINBOW DQN
        'learning_rate': 0.001, #
        'adam_epsilon': 0.00003125,
        # NORMALIZATION?
        'soft_update': False, # seems to always be false, we can try it with tru
        'ema_beta': 0.95,
        'transfer_frequency': 100,
        'replay_period': 1,
        'replay_batch_size': 128,
        'memory_size': 10000, #############
        'min_memory_size': 500,
        'n_step': 3,
        'discount_factor': 0.99,
        'atom_size': 51, #
        'conv_layers': [],
        'conv_layers_noisy': False,
        'width': 512,
        'dense_layers': 2,
        'dense_layers_noisy': True, # i think this is always true for rainbow
        # REWARD CLIPPING
        'noisy_sigma': 0.5, #
        'loss_function': tf.keras.losses.KLDivergence(),
        'dueling': True,
        'advantage_hidden_layers': 1, #
        'value_hidden_layers': 1, #
        'num_training_steps': 25000,
        'per_epsilon': 0.001,
        'per_alpha': 0.5,
        'per_beta': 0.5,
        # 'per_beta_increase': hp.uniform('per_beta_increase', 0, 0.015),
        'v_min': -500.0, # MIN GAME SCORE
        'v_max': 500.0, # MAX GAME SCORE
        # 'search_max_depth': 5,
        # 'search_max_time': 10,
    }

    search_space, initial_best_config = create_search_space()


    max_trials = 2
    trials_step = 100  # how many additional trials to do after loading the last ones

    try:  # try to load an already saved trials object, and increase the max
        trials = pickle.load(open("./classiccontrol_trials.p", "rb"))
        print("Found saved Trials! Loading...")
        max_trials = len(trials.trials) + trials_step
        print("Rerunning from {} trials to {} (+{}) trials".format(len(trials.trials), max_trials, trials_step))
    except:  # create a new trials object and start searching
        # trials = Trials()
        trials = None

    best = fmin(
        fn=func2, # Objective Function to optimize
        space=search_space, # Hyperparameter's Search Space
        algo=tpe.suggest, # Optimization algorithm (representative TPE)
        max_evals=max_trials, # Number of optimization attempts
        trials=trials, # Record the results
        # early_stop_fn=no_progress_loss(5, 1),
        trials_save_file="./classiccontrol_trials.p",
        points_to_evaluate=initial_best_config,
        show_progressbar=False
    )

    print(best)
    best_trial = space_eval(search_space, best)
    gc.collect()
