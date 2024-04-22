# script to generate configs for actors and learners by injecting base configurations
# with ip addresses and ports of both the replay buffer server and model weights storage server,
# and the password of the model weights storage server

import pathlib
import argparse

from agent_configs import ApeXActorConfig, ApeXLearnerConfig


def main():
    parser = argparse.ArgumentParser(description="generate configs")
    parser.add_argument("--replay_addr", type=str, default="127.0.0.1")
    parser.add_argument("--replay_learner_port", type=int, default=5554)
    parser.add_argument("--replay_actors_port", type=int, default=5555)

    parser.add_argument("--storage_hostname", type=str, default="127.0.0.1")
    parser.add_argument("--storage_port", type=int, default=5553)
    parser.add_argument("--storage_username", type=str, default="ezra")

    parser.add_argument(
        "--actor_base", type=str, default="configs/actor_config_example.yaml"
    )
    parser.add_argument(
        "--learner_base",
        type=str,
        default="configs/learner_config_example.yaml",
    )

    parser.add_argument("-o", "--output", type=str, default="generated")

    args = parser.parse_args()

    with open(f"{pathlib.Path.home()}/mongodb/mongodb_admin_password", "r") as f:
        password = f.read()

    storage_dict = dict(
        replay_addr=args.replay_addr,
        storage_hostname=args.storage_hostname,
        storage_port=args.storage_port,
        storage_username=args.storage_username,
        storage_password=password.strip(),
    )
    actor_dict = dict(actor_replay_port=args.replay_actors_port)
    learner_dict = dict(learner_replay_port=args.replay_learner_port)

    actor_conf = ApeXActorConfig.load(args.actor_base)
    learner_conf = ApeXLearnerConfig.load(args.learner_base)

    injected_actor_conf = dict(
        **actor_conf.config_dict | {**storage_dict, **actor_dict}
    )
    injected_learner_conf = dict(
        **learner_conf.config_dict | {**storage_dict, **learner_dict}
    )

    actor_config = ApeXActorConfig(injected_actor_conf, actor_conf.game)
    learner_config = ApeXLearnerConfig(injected_learner_conf, learner_conf.game)

    pathlib.Path(args.output).mkdir(exist_ok=True)
    actor_config.dump(pathlib.Path(args.output, "actor_config.yaml"))
    learner_config.dump(pathlib.Path(args.output, "learner_config.yaml"))


if __name__ == "__main__":
    main()
