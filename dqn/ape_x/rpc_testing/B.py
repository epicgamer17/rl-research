### DO NOT DELETE THE UNUSED IMPORTS!!!
import os
import sys
import classes
import time
import torch
import queue
import pathlib
import logging
import argparse
import torch.distributed
import torch.distributed.rpc as rpc


stop_chan = queue.Queue()


def recv_stop_msg(msg):
    global stop_chan
    stop_chan.put(msg)


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler("B.log", mode="a")
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
    parser = argparse.ArgumentParser()
    # 2 for params, 1 for replay
    parser.add_argument("--rank", type=int, default=2)
    # "parameter" or "replay"
    parser.add_argument("--name", type=str, default="parameter_server")
    parser.add_argument("--world_size", type=int, default=3)

    args = parser.parse_args()

    logger.debug(f"WORLD_SIZE: {args.world_size} RANK: {args.rank}")

    path = pathlib.Path(pathlib.Path.cwd(), "generated", "learner")

    options = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=64, devices=["cuda:0"], init_method=f"file://{path.absolute()}"
    )
    for callee in ["parameter", "replay"]:
        options.set_device_map(callee, {0: 0})

    print(f"[{args.name}] Initializing rpc on remote worker with rank {args.rank}")
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
    logger.info("recieved stop msg")

    time.sleep(1)

    logger.info("collecting garbage")
    import gc
    gc.collect()

    logger.info(f"[{args.name}] shutting down rpc")
    try:
        rpc.shutdown(graceful=True)
        logger.info("rpc shutdown on worker")
    except Exception as e:
        logger.exception(f"[{args.name}] error shutting down rpc: {e}")


if __name__ == "__main__":
    main()
