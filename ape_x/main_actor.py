import tensorflow as tf
import copy
from actor import DistributedActor
import argparse
import gymnasium as gym

import logging

from agent_configs import ApeXConfig
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

actor_config = {
    "poll_params_interval": 100,
    "buffer_size": 100,
    "num_training_steps": 50000,
    "learner_addr": None,
    "learner_port": None,
    "replay_addr": None,
    "replay_port": None,
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

    actor_config["learner_addr"] = args.learner_addr
    actor_config["learner_port"] = args.learner_port
    actor_config["replay_addr"] = args.replay_addr
    actor_config["replay_port"] = args.replay_port

    config = ApeXConfig(actor_config, CartPoleConfig())

    actor = DistributedActor(
        id=args.id,
        env=make_cartpole_env(),
        config=config,
    )
    actor.run()


if __name__ == "__main__":
    main()
