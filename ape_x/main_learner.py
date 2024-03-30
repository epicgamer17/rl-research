import tensorflow as tf
import copy
from configs.agent_configs.ape_x_config import ApeXConfig
from configs.game_configs.cartpole_config import CartPoleConfig
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

learner_config = {
    "num_training_steps": 1000,
    "remove_old_experiences_interval": 1000,
    "push_weights_interval": 20,
    "samples_queue_size": 16,
    "updates_queue_size": 16,
    "port": None,
    "replay_addr": None,
    "replay_port": None,
}


def make_cartpole_env():
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    return env


def main():
    parser = argparse.ArgumentParser(description="Run a distributed Ape-X learner")
    parser.add_argument("port", type=str)
    parser.add_argument("replay_addr", type=str)
    parser.add_argument("replay_port", type=str)
    args = parser.parse_args()

<<<<<<< Updated upstream
    learner_config = copy.deepcopy(base_config)
    learner_config["num_training_steps"] = 1000
    learner_config["remove_old_experiences_interval"] = 1000
    learner_config["push_weights_interval"] = 20

    learner_config["samples_queue_size"] = 16
    learner_config["updates_queue_size"] = 16

    learner_config["port"] = args.port
    learner_config["replay_addr"] = args.replay_addr
    learner_config["replay_port"] = args.replay_port
=======
    learner_config["port"] = args.port
    learner_config["replay_addr"] = args.replay_addr
    learner_config["replay_port"] = args.replay_port

    config = ApeXConfig(learner_config, CartPoleConfig())

>>>>>>> Stashed changes
    learner = DistributedLearner(
        env=make_cartpole_env(),
        config=learner_config,
    )
    learner.run()


if __name__ == "__main__":
    main()
