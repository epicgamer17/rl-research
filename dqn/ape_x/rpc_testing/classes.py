import torch.distributed.rpc as rpc
import torch.nn
import numpy as np
import time
import pathlib
import logging
import gymnasium as gym
import threading
import queue
from typing import NamedTuple


import sys

sys.path.append("../../..")
from replay_buffers.prioritized_n_step_replay_buffer import PrioritizedNStepReplayBuffer

logger = logging.getLogger()


import os


class LearnerTest:
    def __init__(self, stop_fn):
        self.stop_fn = stop_fn

        self.samples_queue = queue.Queue(maxsize=5)

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

        env: gym.Env = gym.make("CartPole-v1", render_mode="rgb_array")
        self.observation_dimensions = env.observation_space.shape
        replay_buffer_args = dict(
            observation_dimensions=self.observation_dimensions,
            observation_dtype=env.observation_space.dtype,
            max_size=128,
            batch_size=16,
            n_step=5,
        )

        print("creating replay")
        self.replay_rref = rpc.remote(
            "replay", PrioritizedNStepReplayBuffer, args=None, kwargs=replay_buffer_args
        )

        self.online_network = torch.nn.Linear(16, 1, False)
        self.target_network = torch.nn.Linear(16, 1, False)

        print("creating online")
        self.online_rref = rpc.remote("parameter", torch.nn.Linear, (16, 1, False))
        print("creating target")
        self.target_rref = rpc.remote("parameter", torch.nn.Linear, (16, 1, False))

        self.store_weights()

        self.actor_rrefs: list[rpc.RRef[ActorTest]] = []

        for i in range(3):
            print("creating actor", i)
            self.actor_rrefs.append(
                rpc.remote(
                    f"actor_{i}",
                    ActorTest,
                    (self.replay_rref, self.target_rref, self.online_rref),
                )
            )

        a = [self.replay_rref, self.target_rref, self.online_rref]
        a.extend(self.actor_rrefs)

        print("waiting for confirmations")
        self._wait_for_confirmations(a)

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
        self.flag = threading.Event()

        self.replay_thread = threading.Thread(
            target=self._fetch_batches, args=(self.flag,)
        )
        self.replay_thread.daemon = True
        self.replay_thread.start()

        for actor in self.actor_rrefs:
            print("starting actor on", actor.owner_name())
            actor.remote().run()

        for i in range(100):
            sample = self.samples_queue.get()

            del sample

            self.store_weights()

        self.cleanup()

    def _fetch_batches(self, flag: threading.Event):
        samples_queue_size = 5
        dummy_q = queue.Queue(samples_queue_size)

        for i in range(samples_queue_size):
            dummy_q.put(0)

        def on_sample_recieved(samples):
            sample = samples.wait()
            if sample == None:
                # no sample recieved, request another sample after 1 second
                logger.info("no sample recieved, waiting 1 second")
                time.sleep(1)
                dummy_q.put(0)
            else:
                batch = samples.wait()
                # print("got batch", batch)
                self.samples_queue.put(batch)
                dummy_q.put(0)

        while not flag.is_set():
            try:
                dummy_q.get(timeout=5)
            except queue.Empty:
                continue

            print("requesting batch")
            try:
                self.replay_rref.rpc_async(10).sample(False).then(on_sample_recieved)
            except Exception as e:
                logger.exception(f"error getting batch: {e}")

        logger.info("replay thread exited")

    def cleanup(self):
        self.flag.set()
        print("stopping actors:")
        for actor in self.actor_rrefs:
            actor.rpc_async().stop()

        for info in ["replay", "parameter", "actor_0", "actor_1", "actor_2"]:
            self.do_stop(info)

        print("deleting refs")
        del self.samples_queue

    def do_stop(self, info):
        print("stopping", info)
        return rpc.remote(info, self.stop_fn, (True,))

    def store_weights(self):
        # TODO - async
        logger.info("storing weights")
        attempts = 5
        t = time.time()
        for attempt in range(attempts):
            try:
                with torch.no_grad():
                    self.target_network.weight = torch.nn.Parameter(
                        torch.rand_like(self.target_network.weight)
                    )
                    self.online_network.weight = torch.nn.Parameter(
                        torch.rand_like(self.target_network.weight)
                    )

                self.target_rref.rpc_async(10).load_state_dict(
                    self.target_network.state_dict()
                ).then(
                    lambda x: logger.info(
                        f"target model succesfully updated in {time.time() - t} s"
                    )
                )
                self.online_rref.rpc_async(10).load_state_dict(
                    self.online_network.state_dict()
                ).then(
                    lambda x: logger.info(
                        f"online model succesfully updated in {time.time() - t} s"
                    )
                )
                return
            except Exception as e:
                logger.exception(
                    f"try {attempt+1} of {attempts}: error setting weights: {e}"
                )
                time.sleep(1)
        logger.info(f"failed to store weights after {attempts} tries")


from replay_buffers.n_step_replay_buffer import NStepReplayBuffer


class Batch(NamedTuple):
    observations: np.ndarray
    infos: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    next_infos: np.ndarray
    dones: np.ndarray
    ids: np.ndarray
    priorities: np.ndarray


class ActorTest:
    def __init__(self, replay_rref, target_rref, online_rref) -> None:
        self.stop_flag = False

        self.env: gym.Env = gym.make("CartPole-v1", render_mode="rgb_array")
        self.replay_rref = replay_rref
        self.target_rref = target_rref
        self.online_rref = online_rref
        self.rb = NStepReplayBuffer(
            self.env.observation_space.shape, self.env.observation_space.dtype, 16, 16, 1
        )

        self.online_network = torch.nn.Linear(16, 1, False)
        self.target_network = torch.nn.Linear(16, 1, False)

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
        self.setup()

        state, info = self.env.reset()
        while not self.stop_flag:
            print("loop")
            time.sleep(1)

            for i in range(100):

                action = 1
                next_state, reward, terminated, truncated, next_info = self.env.step(
                    action
                )
                done = terminated or truncated
                self.rb.store(state, info, action, reward, next_state, next_info, done, 0)
                state = next_state
                info = next_info
                if done:
                    state, info = self.env.reset()

                if self.rb.size == 16:
                    try:
                        batch = Batch(
                            observations=self.rb.observation_buffer,
                            infos=self.rb.info_buffer,
                            actions=self.rb.action_buffer,
                            rewards=self.rb.reward_buffer,
                            next_observations=self.rb.next_observation_buffer,
                            next_infos=self.rb.next_info_buffer,
                            dones=self.rb.done_buffer,
                            ids=np.zeros_like(self.rb.action_buffer, dtype=str),
                            priorities=np.ones_like(self.rb.reward_buffer),
                        )

                        self.replay_rref.rpc_sync().store_batch(batch)
                        print("stored batch")
                        self.rb.clear()
                    except Exception as e:
                        print(f"failed to store batch: {e}")
                    
                    self.update_params()

        self.cleanup()

    def cleanup(self):
        del self.replay_rref
        pass

    def stop(self):
        self.stop_flag = True

    def setup(self):
        # if self.spectator:
        #     self.t_i = time.time()
        # wait for initial network parameters
        logger.info("fetching initial network params from learner...")
        has_weights = self.update_params()
        while not has_weights:
            print("no weights, trying again")
            has_weights = self.update_params()
            time.sleep(2)


    def update_params(self):
        ti = time.time()
        logger.info("fetching weights from storage...")
        try:
            remote_model_params = self.target_rref.rpc_sync(10).state_dict()
            remote_target_params = self.online_rref.rpc_sync(10).state_dict()

            self.online_network.load_state_dict(remote_model_params)
            self.target_network.load_state_dict(remote_target_params)
            logger.info(f"fetching weights took {time.time() - ti} s")
        except Exception as e:
            logger.info(f"failed to fetch weights: {e}")
            return False
        return True