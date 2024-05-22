import argparse
import gymnasium as gym

import logging

from agent_configs import ApeXActorConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler("main_actor.log", mode="a")
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(message)s"))

logger.addHandler(fh)
logger.addHandler(ch)

logging.basicConfig(
    level=logging.DEBUG,
    handlers=[fh, ch],
    format="%(asctime)s %(name)s %(threadName)s %(levelname)s: %(message)s",
)

import sys

sys.path.append("..")
from actor import ApeXActor


def make_cartpole_env():
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    return env


def main():
    parser = argparse.ArgumentParser(description="Run a distributed Ape-X actor")
    parser.add_argument(
        "--config_file", type=str, default="configs/actor_config_example.yaml"
    )
    parser.add_argument("--noisy_sigma", type=float)
    parser.add_argument("--name", type=str, default="actor_0")
    parser.add_argument("--spectator", default=False, action="store_true")

    args = parser.parse_args()
    config = ApeXActorConfig.load(args.config_file)

    if args.noisy_sigma is not None:
        config.noisy_sigma = args.noisy_sigma  # noisy_sigma override
    else:
        # for spectators
        config.conv_layers_noisy = False
        config.dense_layers_noisy = False

    actor = ApeXActor(
        env=make_cartpole_env(),
        config=config,
        name=args.name,
        spectator=args.spectator,
    )
    actor.run()


if __name__ == "__main__":
    main()
