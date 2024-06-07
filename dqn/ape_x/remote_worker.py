import os
import argparse
import torch
import torch.distributed.rpc as rpc
import torch.distributed


def main():
    assert torch.distributed.is_available()
    assert torch.distributed.is_nccl_available()
    parser = argparse.ArgumentParser()

    parser.add_argument("--rank", type=int, default=2)  # 2 for params, 1 for replay
    parser.add_argument(
        "--name", type=str, default="parameter_server"
    )  # "parameter_server" or "replay_server"
    parser.add_argument("--world_size", type=int, default=4)
    parser.add_argument("--master_addr", type=str, default="127.0.0.1")
    parser.add_argument("--rpc_port", type=int, default=3333)
    parser.add_argument("--pg_port", type=int, default=3334)

    args = parser.parse_args()

    torch.distributed.init_process_group(
        backend=torch.distributed.Backend.NCCL,
        init_method=f"tcp://{args.master_addr}:{args.pg_port}",
        world_size=args.world_size,
        rank=args.rank,
    )

    assert torch.distributed.is_initialized()
    assert torch.distributed.get_backend() == torch.distributed.Backend.NCCL

    os.environ["MASTER_ADDR"] = args.master_addr  # learner is the master
    os.environ["MASTER_PORT"] = str(args.rpc_port)

    options = rpc.TensorPipeRpcBackendOptions(devices=[torch.device("cuda:0")])
    for callee in ["parameter_server", "replay_server"]:
        options.set_device_map(callee, {0: 0})

    rpc.init_rpc(
        name=args.name,
        rank=args.rank,
        world_size=args.world_size,
        rpc_backend_options=options,
    )


if __name__ == "__main__":
    main()
