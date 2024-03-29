import tensorflow as tf
import copy
from actor import DistributedActor
import argparse
import gymnasium as gym

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fh = logging.FileHandler("main_actor.log", mode="w")
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(message)s"))

logger.addHandler(fh)
logger.addHandler(ch)

logging.basicConfig(
    level=logging.INFO,
    handlers=[fh, ch],
    format="%(asctime)s %(name)s %(threadName)s %(levelname)s: %(message)s",
)

base_config = {
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


def make_cartpole_env():
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    return env


def main():
    parser = argparse.ArgumentParser(description="Run a distributed Ape-X actor")
    parser.add_argument("id", type=str)
    parser.add_argument("learner_addr", type=str)
    parser.add_argument("learner_port", type=str)
    parser.add_argument("replay_addr", type=str)
    parser.add_argument("replay_port", type=str)
    args = parser.parse_args()

    actor_config = copy.deepcopy(base_config)
    actor_config["poll_params_interval"] = 100
    actor_config["buffer_size"] = 128
    actor_config["num_training_steps"] = 50000
    actor_config["learner_addr"] = args.learner_addr
    actor_config["learner_port"] = args.learner_port
    actor_config["replay_addr"] = args.replay_addr
    actor_config["replay_port"] = args.replay_port

    actor = DistributedActor(
        id=args.id,
        env=make_cartpole_env(),
        config=actor_config,
    )
    actor.run()


if __name__ == "__main__":
    main()
