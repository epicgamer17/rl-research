import subprocess
import tensorflow as tf
import gym
import numpy as np
import threading
import copy

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler("main.log", mode="w")
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(message)s"))

logger.addHandler(fh)
logger.addHandler(ch)

logging.basicConfig(
    level=logging.DEBUG,
    handlers=[fh, ch],
    format="%(asctime)s %(name)s %(threadName)s %(levelname)s: %(message)s",
)

import sys

sys.path.append("../")

from learner import SingleMachineLearner
from actor import SingleMachineActor


class ClipReward(gym.RewardWrapper):
    def __init__(self, env, min_reward, max_reward):
        super().__init__(env)
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.reward_range = (min_reward, max_reward)

    def reward(self, reward):
        return np.clip(reward, self.min_reward, self.max_reward)


def make_pacman_env():
    # as recommended by the original paper, should already include max pooling
    env = ClipReward(
        gym.wrappers.AtariPreprocessing(
            gym.make("MsPacmanNoFrameskip-v4", render_mode="rgb_array"),
            terminal_on_life_loss=True,
        ),
        -1,
        1,
    )  # as recommended by the original paper, should already include max pooling
    env = gym.wrappers.FrameStack(env, 4)
    return env


def make_cartpole_env():
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    return env


pacman_config = {
    "remove_old_experiences_interval": 10,
    "poll_params_interval": 400,  # number of steps between when an actor copies the latest network params from the learner
    "buffer_size": 100,  # size of local replay buffer size
    "batch_size": 50,  # number of experiences to push to remote replay buffer in one batch
    "activation": "relu",
    "kernel_initializer": "he_uniform",
    "optimizer": tf.keras.optimizers.legacy.Adam,  # NO SGD OR RMSPROP FOR NOW SINCE IT IS FOR RAINBOW DQN
    "learning_rate": 0.001,  #
    "adam_epsilon": 0.00003125,
    # NORMALIZATION?
    "soft_update": False,  # seems to always be false, we can try it with tru
    "ema_beta": 0.95,
    "transfer_frequency": 100,
    "replay_period": 50,
    "replay_batch_size": 128,
    "replay_buffer_size": 10000,  #############
    "min_replay_buffer_size": 500,
    "n_step": 3,
    "discount_factor": 0.99,
    "atom_size": 51,  #
    "conv_layers": [(32, 8, (4, 4)), (64, 4, (2, 2)), (64, 3, (1, 1))],
    "conv_layers_noisy": False,
    "width": 512,
    "dense_layers": 2,
    "dense_layers_noisy": True,  # i think this is always true for rainbow
    # REWARD CLIPPING
    "noisy_sigma": 0.5,  #
    "loss_function": tf.keras.losses.KLDivergence(),
    "dueling": True,
    "advantage_hidden_layers": 1,  #
    "value_hidden_layers": 1,  #
    "num_training_steps": 1000,
    "per_epsilon": 0.001,
    "per_alpha": 0.5,
    "per_beta": 0.5,
    "clipnorm": 0.5,
    # 'per_beta_increase': hp.uniform('per_beta_increase', 0, 0.015),
    "v_min": -500.0,  # MIN GAME SCORE
    "v_max": 500.0,  # MAX GAME SCORE
    # 'search_max_depth': 5,
    # 'search_max_time': 10,
}


def main():
    config = {
        "activation": "relu",
        "kernel_initializer": "orthogonal",
        "optimizer": tf.keras.optimizers.legacy.Adam,
        "learning_rate": 0.01,
        "adam_epsilon": 0.0003125,
        "soft_update": False,
        "ema_beta": 0.95,
        "transfer_frequency": 100,
        "replay_period": 2,
        "replay_batch_size": 2**7,
        "replay_buffer_size": 50000,
        "min_replay_buffer_size": 625,
        "n_step": 3,
        "discount_factor": 0.995,
        "atom_size": 51,
        "conv_layers": [],
        "conv_layers_noisy": False,
        "width": 1024,
        "dense_layers": 2,
        "dense_layers_noisy": True,
        "noisy_sigma": 0.5,
        "loss_function": tf.keras.losses.CategoricalCrossentropy(),
        "dueling": True,
        "advantage_hidden_layers": 0,
        "value_hidden_layers": 0,
        "num_training_steps": 1000,
        "per_epsilon": 0.001,
        "per_alpha": 0.05 * 10,
        "per_beta": 0.05 * 7,
        "clipnorm": 0.5,
        "v_min": -500.0,  # MIN GAME SCORE
        "v_max": 500.0,  # MAX GAME SCORE
        # 'search_max_depth': 5,
        # 'search_max_time': 10,
    }
    learner_config = copy.deepcopy(config)
    learner_config["num_training_steps"] = 100
    learner_config["remove_old_experiences_interval"] = 10
    learner = SingleMachineLearner(
        env=make_cartpole_env(),
        config=learner_config,
    )

    actor_config = copy.deepcopy(config)
    actor_config["poll_params_interval"] = 150
    actor_config["buffer_size"] = 100
    actor_config["num_training_steps"] = 10000
    actor = SingleMachineActor(
        id="actor",
        env=make_cartpole_env(),
        config=actor_config,
        single_machine_learner=learner,
    )

    learner_thread = threading.Thread(target=learner.run, name="learner")
    learner_thread.start()

    actor_thread = threading.Thread(target=actor.run, name="actor")
    actor_thread.start()

    # num_actors = 5

    # processes = list()
    # for i in range(num_actors):
    #     id = i
    #     process = subprocess.Popen(
    #         ["python", "main_actor.py", str(id)],
    #     )
    #     processes.append(process)

    # for process in processes:
    #     process.wait()

    # learner_thread = threading.Thread(target=learner.run, name="learner")
    # learner_thread.start()

    # for actor_thread in actor_threads:
    #     actor_thread.start()

    learner_thread.join()

    print("====FINISHED====")
    # logger.debug(f"learner weights: {learner.get_weights()}")


# def main_2():
#     config = {
#         "remove_old_experiences_interval": 10,
#         "poll_params_interval": 400,  # number of steps between when an actor copies the latest network params from the learner
#         "buffer_size": 100,  # size of local replay buffer size
#         "batch_size": 50,  # number of experiences to push to remote replay buffer in one batch
#         "activation": "relu",
#         "kernel_initializer": "he_uniform",
#         "optimizer": tf.keras.optimizers.legacy.Adam,  # NO SGD OR RMSPROP FOR NOW SINCE IT IS FOR RAINBOW DQN
#         "learning_rate": 0.001,  #
#         "adam_epsilon": 0.00003125,
#         # NORMALIZATION?
#         "soft_update": False,  # seems to always be false, we can try it with tru
#         "ema_beta": 0.95,
#         "transfer_frequency": 100,
#         "replay_period": 50,
#         "replay_batch_size": 128,
#         "replay_buffer_size": 10000,  #############
#         "min_replay_buffer_size": 500,
#         "n_step": 3,
#         "discount_factor": 0.99,
#         "atom_size": 51,  #
#         "conv_layers": [(32, 8, (4, 4)), (64, 4, (2, 2)), (64, 3, (1, 1))],
#         "conv_layers_noisy": False,
#         "width": 512,
#         "dense_layers": 2,
#         "dense_layers_noisy": True,  # i think this is always true for rainbow
#         # REWARD CLIPPING
#         "noisy_sigma": 0.5,  #
#         "loss_function": tf.keras.losses.KLDivergence(),
#         "dueling": True,
#         "advantage_hidden_layers": 1,  #
#         "value_hidden_layers": 1,  #
#         "num_training_steps": 25000,
#         "per_epsilon": 0.001,
#         "per_alpha": 0.5,
#         "per_beta": 0.5,
#         "clipnorm": 0.5,
#         # 'per_beta_increase': hp.uniform('per_beta_increase', 0, 0.015),
#         "v_min": -500.0,  # MIN GAME SCORE
#         "v_max": 500.0,  # MAX GAME SCORE
#         # 'search_max_depth': 5,
#         # 'search_max_time': 10,
#     }
#
#     l = SingleMachineLearner(
#         env=make_pacman_env(),
#         config=config,
#     )
#
#     print(l.get_weights())
#
#
if __name__ == "__main__":
    main()
