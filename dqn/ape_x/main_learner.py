from agent_configs import ApeXLearnerConfig
from learner import ApeXLearner
import gymnasium as gym
import argparse

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler("main_learner.log", mode="w")
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

logger.addHandler(fh)
logger.addHandler(ch)

logging.basicConfig(
    level=logging.DEBUG,
    handlers=[fh, ch],
    format="%(asctime)s %(name)s %(threadName)s %(levelname)s: %(message)s",
)


def make_cartpole_env():
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    return env


def recv_stop_msg(msg):
    global stop_chan
    stop_chan.put(msg)


def main():
    parser = argparse.ArgumentParser(description="Run a distributed Ape-X learner")
    parser.add_argument(
        "--config_file", type=str, default="configs/learner_config_example.yaml"
    )

    args = parser.parse_args()

    config = ApeXLearnerConfig.load(args.config_file)

    learner = ApeXLearner(
        env=make_cartpole_env(), config=config, name="learner", stop_fn=recv_stop_msg
    )
    learner.run()


if __name__ == "__main__":
    main()
