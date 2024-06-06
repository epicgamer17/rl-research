import os
import argparse
import torch.distributed.rpc as rpc


def main():
    parser = argparse.ArgumentParser

    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--world_size", type=int, default=4)
    parser.add_argument("--master_addr", type=str, default="127.0.0.1")
    parser.add_argument("--rpc_port", type=int, default=3333)

    os.environ["MASTER_ADDR"] = args.master_addr  # learner is the master
    os.environ["MASTER_PORT"] = args.rpc_port

    args = parser.parse_args()

    options = rpc.TensorPipeRpcBackendOptions()
    rpc.init_rpc("replay_server", options, args.rank, args.world_size)


if __name__ == "__main__":
    main()
