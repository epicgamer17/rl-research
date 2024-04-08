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
    "actor_buffer_size": 128,
    "poll_params_interval": 128,
}

conf = {**rainbow_config, **distributed_config, **actor_config}


def make_cartpole_env():
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    return env


def main():
    parser = argparse.ArgumentParser(description="Run a distributed Ape-X actor")
    parser.add_argument("id", type=str)
    parser.add_argument("epsilon", type=float)

    parser.add_argument("replay_addr", type=str)
    parser.add_argument("replay_port", type=str)
    parser.add_argument("storage_hostname", type=str)
    parser.add_argument("storage_port", type=int)
    parser.add_argument("storage_username", type=str)
    parser.add_argument("storage_password", type=str)

    parser.add_argument("--spectator", default=False, action="store_true")

    args = parser.parse_args()

    conf["replay_addr"] = args.replay_addr
    conf["replay_port"] = args.replay_port
    conf["storage_hostname"] = args.storage_hostname
    conf["storage_port"] = args.storage_port
    conf["storage_username"] = args.storage_username
    conf["storage_password"] = args.storage_password
    # do something with epsilon

    config = ApeXActorConfig(conf, CartPoleConfig())

    actor = ApeXActor(
        env=make_cartpole_env(),
        config=config,
        name=id,
        spectator=args.spectator,
    )
    actor.run()


if __name__ == "__main__":
    main()
