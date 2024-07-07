import torch.distributed.rpc as rpc
import torch.nn
import numpy as np
import time
import pathlib
import logging

logger = logging.getLogger()


import os


class LearnerTest:
    def __init__(self, stop_fn):
        self.stop_fn = stop_fn

        path = pathlib.Path(pathlib.Path.cwd(), "generated", "learner")

        if os.path.exists(path):
            os.remove(path)

        options = rpc.TensorPipeRpcBackendOptions(
            num_worker_threads=64,
            devices=["cuda:0"],
            init_method=f"file://{path.absolute()}",
        )
        for callee in ["parameter", "replay"]:
            options.set_device_map(callee, {0: 0})

        try:
            print(f"[learner] Initializing rpc on remote worker with rank 0")
            rpc.init_rpc(
                name="learner",
                rank=0,
                world_size=6,
                rpc_backend_options=options,
            )
        except Exception as e:
            logger.exception(f"[learner] error initializing rpc: {e}")

        print("creating replay")
        self.replay_rref = rpc.remote("replay", ReplayTest)
        print("creating target")
        self.target_rref = rpc.remote("parameter", torch.nn.Identity, (16,))
        print("creating online")
        self.online_rref = rpc.remote("parameter", torch.nn.Identity, (16,))

        self.actor_rrefs: list[rpc.RRef[ActorTest]] = []

        for i in range(3):
            print("creating actor", i)
            self.actor_rrefs.append(rpc.remote(f"actor_{i}", ActorTest, (self.replay_rref, self.target_rref, self.online_rref)))

        a = [self.replay_rref, self.target_rref, self.online_rref]
        a.extend(self.actor_rrefs)

        print("waiting for confirmations")
        self._wait_for_confirmations(a)

        for actor in self.actor_rrefs:
            print("starting actor on", actor.owner_name())
            actor.remote().run()

    def _wait_for_confirmations(self, rrefs):
        logger.info("waiting for confirmations")
        confirmed = 0
        to_confirm = len(rrefs)
        while confirmed < to_confirm:
            print(f"{confirmed} / {to_confirm} rrefs confirmed")
            for rref in rrefs:
                if rref.confirmed_by_owner():
                    print(rref.owner_name())
                    confirmed += 1
                    rrefs.remove(rref)
            time.sleep(1)

        logger.info(f"{confirmed} / {to_confirm} rrefs confirmed")
        return confirmed == to_confirm

    def run(self):
        for i in range(5):
            print(i)
            time.sleep(1)

        self.cleanup()

    def cleanup(self):
        print("stopping actors:")
        for actor in self.actor_rrefs:
            actor.rpc_async().stop()

        for info in ["replay", "parameter", "actor_0", "actor_1", "actor_2"]:
            self.do_stop(info)

        print("deleting refs")

    def do_stop(self, info):
        print("stopping", info)
        return rpc.remote(info, self.stop_fn, (True,))


class ReplayTest:
    def __init__(self) -> None:
        self.replay_buffer = np.zeros((100, 16))
        pass


class ActorTest:
    def __init__(self, replay_rref, target_rref, online_rref) -> None:
        self.stop_flag = False

        self.replay_rref = replay_rref
        self.target_rref = target_rref
        self.online_rref = online_rref

        self._wait_for_confirmations(
            [self.replay_rref, self.target_rref, self.online_rref]
        )

    def _wait_for_confirmations(self, rrefs):
        logger.info("waiting for confirmations")
        confirmed = 0
        to_confirm = len(rrefs)
        while confirmed < to_confirm:
            logger.debug(f"{confirmed} / {to_confirm} rrefs confirmed")
            for rref in rrefs:
                if rref.confirmed_by_owner():
                    print(rref.owner_name())
                    confirmed += 1
                    rrefs.remove(rref)
            time.sleep(1)

        logger.info(f"{confirmed} / {to_confirm} rrefs confirmed")
        return confirmed == to_confirm

    def run(self):
        while not self.stop_flag:
            print("loop")
            time.sleep(1)

        self.cleanup()

    def cleanup(self):
        pass

    def stop(self):
        self.stop_flag = True
