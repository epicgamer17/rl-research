import os
import queue
import torch
import torch.distributed.rpc as rpc

stop_chan = queue.Queue()


def recv_stop_msg(msg):
    global stop_chan
    stop_chan.put(msg)


def main():
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "3333"
    os.environ["WORLD_SIZE"] = "2"
    os.environ["RANK"] = "1"

    rpc.init_rpc("B", rank=1, world_size=2)
    stop_chan.get()
    print("done")
    rpc.shutdown()


if __name__ == "__main__":
    main()
