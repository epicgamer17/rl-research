import tensorflow as tf
from tensorflow import keras
from keras import losses
from agent_configs import ApeXConfig, LearnerApeXMixin, DistributedConfig
from game_configs import CartPoleConfig
from learner import DistributedLearner
import gymnasium as gym
import argparse

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fh = logging.FileHandler("main_learner.log", mode="w")
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(message)s"))

logger.addHandler(fh)
logger.addHandler(ch)

logging.basicConfig(
    level=logging.INFO,
    handlers=[fh, ch],
    format="%(asctime)s %(name)s %(threadName)s %(levelname)s: %(message)s",
)

distributed_config = {
    "learner_addr": "127.0.0.1",
    "learner_port": 5556,
    "replay_port": 5554,
    "replay_addr": "127.0.0.1",
}

rainbow_config = {
    "width": 1024,
    "atom_size": 51,
    "loss_function": losses.CategoricalCrossentropy(),
    "activation": "relu",
    "kernel_initializer": "orthogonal",
    "adam_epsilon": 0.0003125,
    "ema_beta": 0.95,
    "transfer_interval": 100,
    "minibatch_size": 128,
    "dense_layers": 2,
    "dense_layers_noisy": True,
    "dueling": True,
    "per_epsilon": 0.001,
    "per_alpha": 0.05 * 10,
    "per_beta": 0.05 * 7,
    "clipnorm": 0.5,
    "replay_interval": 1,
}


learner_config = {
    "training_steps": 1000,
    "remove_old_experiences_interval": 1000,
    "samples_queue_size": 16,
    "updates_queue_size": 16,
    "push_params_interval": 1,
}

conf = {**rainbow_config, **distributed_config, **learner_config}


def make_cartpole_env():
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    return env


class LearnerConfig(ApeXConfig, LearnerApeXMixin, DistributedConfig):
    def __init__(self, learner_config, game_config):
        super().__init__(learner_config, game_config)


def main():
    parser = argparse.ArgumentParser(description="Run a distributed Ape-X learner")
    parser.add_argument(
        "port", type=str, help="Port that the learner zmq socket will bind to"
    )
    parser.add_argument("replay_addr", type=str, help="Address of the replay server")
    parser.add_argument(
        "replay_port",
        type=str,
        help="Port of the replay server that the learner will connect to. It is different than the port for actors.",
    )
    args = parser.parse_args()

    conf["learner_port"] = args.port
    conf["replay_addr"] = args.replay_addr
    conf["replay_port"] = args.replay_port

    config = LearnerConfig(conf, CartPoleConfig())

    learner = DistributedLearner(
        env=make_cartpole_env(),
        config=config,
    )
    learner.run()


if __name__ == "__main__":
    main()
