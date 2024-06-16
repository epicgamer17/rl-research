import os
import torch.distributed.rpc as rpc
import C

def recv_stop_msg(msg):
    global stop_chan
    stop_chan.put(msg)


def main():
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "3333"
    os.environ["WORLD_SIZE"] = "2"
    os.environ["RANK"] = "1"

    rpc.init_rpc("A", rank=0, world_size=2)
    info = rpc.get_worker_info("B")
    x = C.Class(recv_stop_msg)
    A = x.do_stop(info)
    print("done")
    print(A)
    rpc.shutdown()


if __name__ == "__main__":
    main()
