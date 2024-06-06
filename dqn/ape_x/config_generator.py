# script to generate configs for actors and learners by injecting base configurations
# with distributed info

import argparse
from agent_configs import ApeXActorConfig, ApeXLearnerConfig


def main():
    parser = argparse.ArgumentParser(description="generate configs")
    parser.add_argument("--rpc_port", type=int, default=3333)
    parser.add_argument("--world_size", type=int, default=4)

    parser.add_argument("--master_addr", type=str, default="127.0.0.1")
    parser.add_argument("--replay_addr", type=str, default="127.0.0.1")
    parser.add_argument("--storage_addr", type=str, default="127.0.0.1")
    parser.add_argument(
        "--actor_base", type=str, default="configs/actor_config_example.yaml"
    )
    parser.add_argument(
        "--learner_base",
        type=str,
        default="configs/learner_config_example.yaml",
    )

    parser.add_argument("--learner_output", type=str, default="generated")
    parser.add_argument("--actor_output", type=str, default="generated")

    args = parser.parse_args()

    distributed_dict = dict(
        # "rank": int = self.parse_field("rank", required=True)
        # "worker_name": int = self.parse_field("worker_name", required=True)
        world_size=args.world_size,
        rpc_port=args.rpc_port,
        replay_addr=args.replay_addr,
        storage_addr=args.storage_addr,
    )

    actor_conf = ApeXActorConfig.load(args.actor_base)
    learner_conf = ApeXLearnerConfig.load(args.learner_base)

    injected_actor_conf = actor_conf.config_dict | {**distributed_dict | {"rank": 0}}
    injected_learner_conf = learner_conf.config_dict | {**distributed_dict | {"rank": 3}}

    actor_config = ApeXActorConfig(injected_actor_conf, actor_conf.game)
    learner_config = ApeXLearnerConfig(injected_learner_conf, learner_conf.game)

    actor_config.dump(args.actor_output)
    learner_config.dump(args.learner_output)


if __name__ == "__main__":
    main()
