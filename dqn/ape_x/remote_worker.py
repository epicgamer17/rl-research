import time
import os
import argparse
import torch
### DO NOT DELETE THE UNUSED IMPORTS!!!
import copy
import torch.distributed.rpc as rpc
import torch.distributed.rpc

import sys

sys.path.append("../..")
import dqn
import dqn.rainbow
import dqn.ape_x
import replay_buffers
import replay_buffers.prioritized_n_step_replay_buffer

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler("remote_worker.log", mode="a")
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(message)s"))

logger.addHandler(fh)
logger.addHandler(ch)


def main():
    assert torch.distributed.is_available() and torch.distributed.is_nccl_available()
    parser = argparse.ArgumentParser()

    time.sleep(10)

    parser.add_argument("--rank", type=int, default=2)  # 2 for params, 1 for replay
    parser.add_argument(
        "--name", type=str, default="parameter_server"
    )  # "parameter_server" or "replay_server"
    parser.add_argument("--world_size", type=int, default=6)
    parser.add_argument("--master_addr", type=str, default="127.0.0.1")
    parser.add_argument("--rpc_port", type=int, default=3333)

    args = parser.parse_args()

    os.environ["MASTER_ADDR"] = args.master_addr  # learner is the master
    os.environ["MASTER_PORT"] = str(args.rpc_port)
    os.environ["WORLD_SIZE"] = str(args.world_size)
    os.environ["RANK"] = str(args.rank)

    logger.debug(
        f'{args.rank} environment: MASTER_ADDR: {os.environ["MASTER_ADDR"]} MASTER_PORT: {os.environ["MASTER_PORT"]} WORLD_SIZE: {os.environ["WORLD_SIZE"]} RANK: {os.environ["RANK"]}'
    )

    # logger.info(
    #     f"[{args.name}] Initializing process group on remote worker with rank {args.rank}"
    # )
    # torch.distributed.init_process_group(backend=torch.distributed.Backend.NCCL)

    # assert (
    #     torch.distributed.is_initialized()
    #     and torch.distributed.get_backend() == torch.distributed.Backend.NCCL
    # )

    options = rpc.TensorPipeRpcBackendOptions(devices=[torch.device("cuda:0")])
    for callee in ["parameter_server", "replay_server"]:
        options.set_device_map(callee, {0: 0})

    logger.info(f"[{args.name}] Initializing rpc on remote worker with rank {args.rank}")
    try:
        rpc.init_rpc(
            name=args.name,
            rank=args.rank,
            world_size=args.world_size,
            rpc_backend_options=options,
        )
    except Exception as e:
        logger.exception(f"error initializing rpc: {e}")
    logger.info(f"[{args.name}] rpc initialized.")
    
    while True:
        print("sleeping")
        time.sleep(1)


if __name__ == "__main__":
    main()
