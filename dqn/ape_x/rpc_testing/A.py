import time
import logging
import pathlib
import argparse
import torch.distributed.rpc as rpc
import classes

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler("A.log", mode="w")
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(message)s"))

logger.addHandler(fh)
logger.addHandler(ch)


def recv_stop_msg(msg):
    global stop_chan
    stop_chan.put(msg)


def main():
    l = classes.LearnerTest(stop_fn=recv_stop_msg)
    l.run()

    logger.info("deleting l")
    del l
    try:
        logger.info("shutting down rpc")
        rpc.shutdown(graceful=True)
        logger.info("rpc shutdown on worker")
    except Exception as e:
        logger.exception(f"[A] error shutting down rpc: {e}")


if __name__ == "__main__":
    main()
