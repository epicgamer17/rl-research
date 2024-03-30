import tensorflow as tf
from tensorflow import keras
from keras import losses
from actor import DistributedActor
import argparse
import gymnasium as gym

import logging

from agent_configs import ApeXConfig, ActorApeXMixin, DistributedConfig
from game_configs import CartPoleConfig

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

distributed_config = {
    "learner_addr": "127.0.0.1",
    "learner_port": 5556,
    "replay_port": 5554,
    "replay_addr": "127.0.0.1",
}

rainbow_config = {
    "loss_function": losses.Huber(),
    "activation": "relu",
    "kernel_initializer": "he_normal",
}


actor_config = {}

conf = {**rainbow_config, **distributed_config, **actor_config}


def make_cartpole_env():
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    return env


class ActorConfig(ApeXConfig, ActorApeXMixin, DistributedConfig):
    def __init__(self, actor_config, game_config):
        super(ApeXConfig, self).__init__(actor_config, game_config)


def main():
    parser = argparse.ArgumentParser(description="Run a distributed Ape-X actor")
    parser.add_argument("id", type=str)
    parser.add_argument("learner_addr", type=str)
    parser.add_argument("learner_port", type=str)
    parser.add_argument("replay_addr", type=str)
    parser.add_argument("replay_port", type=str)
    args = parser.parse_args()

    conf["learner_addr"] = args.learner_addr
    conf["learner_port"] = args.learner_port
    conf["replay_addr"] = args.replay_addr
    conf["replay_port"] = args.replay_port

    config = ActorConfig(conf, CartPoleConfig())

    actor = DistributedActor(
        id=args.id,
        env=make_cartpole_env(),
        config=config,
    )
    actor.run()


if __name__ == "__main__":
    main()
