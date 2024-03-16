import sys

sys.path.append("../")

import gym
import numpy as np
from ape_x.actor import SingleMachineActor
from ape_x.learner import SingleMachineLearner
import tensorflow as tf


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


def main():
    config = {
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
        "num_training_steps": 25000,
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

    l = SingleMachineLearner(
        env=make_pacman_env(),
        config=config,
    )

    a = SingleMachineActor(
        0,
        env=make_pacman_env(),
        config=config,
        single_machine_learner=l,
    )

    print("running actor")
    a.run()


if __name__ == "__main__":
    main()
