### DO NOT DELETE THE UNUSED IMPORTS!!!
import os
import sys
import time
import torch
import queue
import logging
import argparse
import torch.distributed
import torch.distributed.rpc as rpc


stop_chan = queue.Queue()


def recv_stop_msg(msg):
    global stop_chan
    stop_chan.put(msg)


sys.path.append("../..")
import dqn
import dqn.rainbow
import dqn.ape_x
import replay_buffers
import replay_buffers.prioritized_n_step_replay_buffer


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

    logger.info(
        f"[{args.name}] Initializing process group on remote worker with rank {args.rank}"
    )
    torch.distributed.init_process_group(backend=torch.distributed.Backend.NCCL)

    assert (
        torch.distributed.is_initialized()
        and torch.distributed.get_backend() == torch.distributed.Backend.NCCL
    )

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
        logger.exception(f"[{args.name}] error initializing rpc: {e}")
    logger.info(f"[{args.name}] rpc initialized.")

    logger.info("waiting for stop signal")
    stop_chan.get()
    logger.info("recieved stop msg, waiting for all workers to finish outstanding work")

    workers = ["learner", "parameter_server", "replay_server"]
    workers.extend(f"actor_{i}" for i in range(0, args.world_size - 3))
    rpc.api._barrier(workers)

    logger.info(f"[{args.name}] shutting down rpc")
    try:
        rpc.shutdown()
        logger.info("rpc shutdown on worker")
    except Exception as e:
        logger.exception(f"[{args.name}] error shutting down rpc: {e}")
    
    try:
        logger.info(f"[{args.name}] destroying process group")
        torch.distributed.destroy_process_group()
    except Exception as e:
        logger.exception(f"[{args.name}] error destroying process group: {e}")
    

    


if __name__ == "__main__":
    main()
