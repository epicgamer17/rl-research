import tensorflow as tf
import pathlib
import argparse
import gymnasium as gym

from agent_configs import (
    ApeXActorConfig,
    ApeXLearnerConfig,
    ReplayBufferConfig,
    RainbowConfig,
)
from game_configs import CartPoleConfig

env = gym.make("CartPole-v1", render_mode="rgb_array")

storage_dict = dict(
    replay_addr="",
    storage_hostname="",
    storage_port="",
    storage_username="",
    storage_password="",
)

rainbow_dict = dict(
    activation="relu",
    kernel_initializer="glorot_uniform",
    loss_function=tf.keras.losses.CategoricalCrossentropy(),
)

actor_dict = dict(actor_replay_port="")

learner_dict = dict(learner_replay_port="")

replay_dict = dict(
    observation_dimensions=env.observation_space.shape,
    batch_size=128,
)


def main():
    parser = argparse.ArgumentParser(description="generate configs")
    parser.add_argument("--replay_addr", type=str, default="127.0.0.1")
    parser.add_argument("--replay_learner_port", type=int, default=5554)
    parser.add_argument("--replay_actors_port", type=int, default=5555)

    parser.add_argument("--storage_hostname", type=str, default="127.0.0.1")
    parser.add_argument("--storage_port", type=int, default=5553)
    parser.add_argument("--storage_username", type=str, default="ezra")

    parser.add_argument(
        "--actor_rainbow_base", type=str, default="configs/rainbow_base_example.yaml"
    )
    parser.add_argument(
        "--learner_rainbow_base", type=str, default="configs/rainbow_base_example.yaml"
    )

    args = parser.parse_args()

    with open(f"{pathlib.Path.home()}/mongodb/mongodb_admin_password", "r") as f:
        password = f.read()

    storage_dict["replay_addr"] = args.replay_addr
    storage_dict["storage_hostname"] = args.storage_hostname
    storage_dict["storage_port"] = args.storage_port
    storage_dict["storage_username"] = args.storage_username
    storage_dict["storage_password"] = password.strip()

    actor_dict["actor_replay_port"] = args.replay_actors_port
    learner_dict["learner_replay_port"] = args.replay_learner_port

    actor_rainbow_conf = RainbowConfig.load(args.actor_rainbow_base)
    learner_rainbow_conf = RainbowConfig.load(args.learner_rainbow_base)

    actor_conf = {**actor_rainbow_conf.config_dict, **storage_dict, **actor_dict}
    actor_config = ApeXActorConfig(actor_conf, CartPoleConfig())
    actor_config.dump("configs/actor_config_example.yaml")

    learner_conf = {**learner_rainbow_conf.config_dict, **storage_dict, **learner_dict}
    learner_config = ApeXLearnerConfig(learner_conf, CartPoleConfig())
    learner_config.dump("configs/learner_config_example.yaml")

    replay_conf = {**replay_dict}
    replay_config = ReplayBufferConfig(replay_conf)
    replay_config.dump("configs/replay_config_example.yaml")


if __name__ == "__main__":
    main()
