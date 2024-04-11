import tensorflow as tf
from tensorflow import keras
from keras import losses
import argparse
import gymnasium as gym

import logging

from agent_configs import ApeXActorConfig
from game_configs import CartPoleConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fh = logging.FileHandler("main_actor.log", mode="a")
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(message)s"))

logger.addHandler(fh)
logger.addHandler(ch)

logging.basicConfig(
    level=logging.INFO,
    handlers=[fh, ch],
    format="%(asctime)s %(name)s %(threadName)s %(levelname)s: %(message)s",
)

import sys

sys.path.append("..")
from actor import ApeXActor

distributed_config = {
    "learner_addr": "127.0.0.1",
    "learner_port": 5556,
    "replay_port": 5554,
    "replay_addr": "127.0.0.1",
}

rainbow_config = {
    "width": 512,
    "loss_function": losses.CategoricalCrossentropy(),
    "activation": "relu",
    "kernel_initializer": "orthogonal",
    "adam_epsilon": 0.0003125,
    "transfer_interval": 100,
    "dense_layers": 2,
    "per_epsilon": 0.001,
    "per_alpha": 0.5,
    "per_beta": 0.4,
    "clipnorm": None,
}


actor_config = {
    "actor_buffer_size": 128,  # sets minibatch size and replay buffer size
    "poll_params_interval": 128,
}

conf = {**rainbow_config, **distributed_config, **actor_config}


def make_cartpole_env():
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    return env


def main():
    parser = argparse.ArgumentParser(description="Run a distributed Ape-X actor")
    parser.add_argument("--config_file", type=str)
    parser.add_argument("--name", type=str)
    parser.add_argument("--spectator", default=False, action="store_true")
    # parser.add_argument("--epsilon", type=float)

    args = parser.parse_args()
    config = ApeXActorConfig.load(args.config_file)

    actor = ApeXActor(
        env=make_cartpole_env(),
        config=config,
        name=args.name,
        spectator=args.spectator,
    )
    actor.run()


if __name__ == "__main__":
    main()
