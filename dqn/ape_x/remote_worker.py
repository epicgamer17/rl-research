import os
import argparse
import torch.distributed.rpc as rpc


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--rank", type=int, default=2)  # 2 for params, 1 for replay
    parser.add_argument(
        "--name", type=str, default="parameter_server"
    )  # "parameter_server" or "replay_server"
    parser.add_argument("--world_size", type=int, default=4)
    parser.add_argument("--master_addr", type=str, default="127.0.0.1")
    parser.add_argument("--rpc_port", type=int, default=3333)

    args = parser.parse_args()

    os.environ["MASTER_ADDR"] = args.master_addr  # learner is the master
    os.environ["MASTER_PORT"] = str(args.rpc_port)

    options = rpc.TensorPipeRpcBackendOptions()
    rpc.init_rpc(
        name=args.name,
        rank=args.rank,
        world_size=args.world_size,
        rpc_backend_options=options,
    )


if __name__ == "__main__":
    main()
