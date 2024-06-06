import os
import argparse
import torch.distributed.rpc as rpc

import logging

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


def main():
    parser = argparse.ArgumentParser(description="Run a distributed Ape-X actor")

    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--world_size", type=int, default=4)
    parser.add_argument("--master_addr", type=str, default="127.0.0.1")
    parser.add_argument("--rpc_port", type=int, default=3333)

    args = parser.parse_args()

    os.environ["MASTER_ADDR"] = args.master_addr  # learner is the master
    os.environ["MASTER_PORT"] = str(args.rpc_port)

    options = rpc.TensorPipeRpcBackendOptions()

    rpc.init_rpc(
        name=f"actor_{args.rank}",
        rank=args.rank,
        world_size=args.world_size,
        rpc_backend_options=options,
    )


if __name__ == "__main__":
    main()
