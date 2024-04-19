import tensorflow as tf
from tensorflow import keras
from keras import losses
from agent_configs import ApeXLearnerConfig
from learner import ApeXLearner
import gymnasium as gym
import argparse

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler("main_learner.log", mode="w")
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(message)s"))

logger.addHandler(fh)
logger.addHandler(ch)

logging.basicConfig(
    level=logging.DEBUG,
    handlers=[fh, ch],
    format="%(asctime)s %(name)s %(threadName)s %(levelname)s: %(message)s",
)

# distributed_config = {
#     "learner_addr": "127.0.0.1",
#     "learner_port": 5556,
#     "replay_port": 5554,
#     "replay_addr": "127.0.0.1",
# }

# rainbow_config = {
#     "atom_size": 51,
#     "activation": "relu",
#     "kernel_initializer": "orthogonal",
#     "ema_beta": 0.95,
#     "transfer_interval": 100,
#     "minibatch_size": 128,
#     "per_epsilon": 0.001,
#     "per_alpha": 0.5,
#     "per_beta": 0.4,
#     "clipnorm": None,
#     # "discount_factor": 0.99,
#     # "n_step": 3,
#     "transfer_interval": 100,
#     "dense_layers": 2,
#     "width": 512,
#     "learning_rate": 0.0001,
#     "loss_function": losses.CategoricalCrossentropy(),
#     "adam_epsilon": 0.0003125,
# }


# learner_config = {
#     "training_steps": 1000,
#     "remove_old_experiences_interval": 1000,
#     "samples_queue_size": 2,
#     "updates_queue_size": 16,
#     "push_params_interval": 1,
# }

# conf = {**rainbow_config, **distributed_config, **learner_config}


def make_cartpole_env():
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    return env


def main():
    parser = argparse.ArgumentParser(description="Run a distributed Ape-X learner")
    parser.add_argument(
        "--config_file", type=str, default="learner_config_example.yaml"
    )

    args = parser.parse_args()

    config = ApeXLearnerConfig.load(args.config_file)

    learner = ApeXLearner(env=make_cartpole_env(), config=config, name="learner")
    learner.run()


if __name__ == "__main__":
    main()
