import torch
import sys
import logging
import time
import numpy as np
import queue
import threading
from typing import NamedTuple
import copy
from agent_configs import ApeXLearnerConfig
from actor import ApeXActor
from utils import update_per_beta
import torch.distributed
from torch.nn.utils import clip_grad_norm_

sys.path.append("../")
from dqn.rainbow.rainbow_agent import RainbowAgent

import matplotlib

matplotlib.use("Agg")


logger = logging.getLogger(__name__)


class Update(NamedTuple):
    indices: np.ndarray
    priorities: np.ndarray
    ids: np.ndarray


class ApeXLearnerBase(RainbowAgent):
    def __init__(self, env, config: ApeXLearnerConfig, name):
        super().__init__(
            env,
            config,
            name,
        )
        self.config = config
        self.samples_queue: queue.Queue[dict] = queue.Queue(
            maxsize=self.config.samples_queue_size
        )
        self.updates_queue: queue.Queue[Update] = queue.Queue(
            maxsize=self.config.updates_queue_size
        )

        self.stats = {
            "loss": [],
            "test_score": [],
        }
        self.targets = {
            "test_score": self.env.spec.reward_threshold,
        }

        self.per_sample_beta = self.config.per_beta

    # apex learner methods
    def store_weights(self, weights):
        pass

    def update_replay_priorities(self, samples, priorities):
        # return super().update_replay_priorities(samples, priorities)
        pass

    def on_training_step_end(self):
        pass

    def on_run(self):
        pass

    def on_done(self):
        pass

    def learn(self):
        ti = time.time()
        samples = self.samples_queue.get()
        loss = super().learn_from_sample(samples)
        logger.info(f"experience replay took {time.time()-ti}. loss: {loss} s")
        return loss

    def run(self):
        try:
            start_time = time.time()
            target_model_updated = False
            self.on_run()
            logger.info("learner running")

            self.training_steps += self.start_training_step
            for training_step in range(self.start_training_step, self.training_steps):
                # stop training if going over 1.5 hours
                logger.info(
                    f"learner training step: {training_step}/{self.training_steps}"
                )

                if time.time() - start_time > 3600 * 1.5:
                    break

                if training_step % self.config.push_params_interval == 0:
                    self.store_weights()

                loss = self.learn()
                target_model_updated = False
                if training_step % self.config.transfer_interval == 0:
                    self.update_target_model()
                    target_model_updated = True

                self.stats["loss"].append(
                    {"loss": loss, "target_model_updated": target_model_updated}
                )

                if training_step % self.checkpoint_interval == 0:
                    self.save_checkpoint(
                        training_step, training_step, time.time() - start_time
                    )

                    if training_step // self.training_steps > 0.125:
                        past_scores_dicts = self.stats["test_score"][-5:]
                        scores = [score_dict["score"] for score_dict in past_scores_dicts]
                        avg = np.sum(scores) / 5
                        if avg < 10:
                            return  # could do stopping param as the slope of line of best fit

                self.on_training_step_end()

            logger.info("loop done")

            self.save_checkpoint(training_step, training_step, time.time() - start_time)
        except Exception as e:
            logger.exception(f"run method ended by error: {e}")
        finally:
            self.env.close()
            self.on_done()


import torch.distributed.rpc as rpc
import os
import socket


from replay_buffers.prioritized_n_step_replay_buffer import PrioritizedNStepReplayBuffer
from dqn.rainbow.rainbow_network import RainbowNetwork


class ApeXLearner(ApeXLearnerBase):
    def __init__(self, env, config: ApeXLearnerConfig, name: str):
        super().__init__(env, config, name)
        assert torch.distributed.is_available()
        assert torch.distributed.is_nccl_available()
        self.updates_queue = queue.Queue()
        self.config = config

        # torch rpc initialization
        os.environ["MASTER_ADDR"] = socket.getfqdn()  # learner is the master
        os.environ["MASTER_PORT"] = str(self.config.rpc_port)

        # for cuda to cuda rpc
        options = rpc.TensorPipeRpcBackendOptions(
            num_worker_threads=32, devices=["cuda:0"]
        )

        for callee in ["parameter_server", "replay_server"]:
            options.set_device_map(callee, {0: 0})

        print("initializing rpc...")
        torch.distributed.init_process_group(
            backend=torch.distributed.Backend.NCCL,
            world_size=self.config.world_size,
            rank=0,
            store=torch.distributed.TCPStore(
                host_name=self.config.master_addr,
                port=self.config.pg_port,
                world_size=self.config.world_size,
                is_master=True,
                wait_for_workers=True,
            ),
            pg_options=None,
            device_id=None,
        )
        assert torch.distributed.is_initialized()
        assert torch.distributed.get_backend() == torch.distributed.Backend.NCCL
        rpc.init_rpc(
            name="learner",
            rank=0,
            world_size=self.config.world_size,
            rpc_backend_options=options,
        )

        # use WorkerInfo instead of expensive strings, "Use this WorkerInfo to avoid passing an expensive string on every invocation."
        self.replay_worker_info = rpc.get_worker_info("replay_server")
        self.parameter_worker_info = rpc.get_worker_info("parameter_server")

        # create remote references (rref)s to the online_network, target_network, (parameter server) and replay buffer (replay memory server)
        rainbow_network_args = (
            config,
            self.num_actions,
            (self.config.minibatch_size,) + self.observation_dimensions,
        )
        replay_buffer_args = dict(
            observation_dimensions=self.observation_dimensions,
            observation_dtype=self.env.observation_space.dtype,
            max_size=self.config.replay_buffer_size,
            batch_size=self.config.minibatch_size,
            max_priority=1.0,
            alpha=self.config.per_alpha,
            beta=self.config.per_beta,
            # epsilon=config["per_epsilon"],
            n_step=self.config.n_step,
            gamma=self.config.discount_factor,
        )

        self.online_network_rref: rpc.RRef[RainbowNetwork] = rpc.remote(
            self.parameter_worker_info, RainbowNetwork, rainbow_network_args
        )
        self.target_network_rref: rpc.RRef[RainbowNetwork] = rpc.remote(
            self.parameter_worker_info, RainbowNetwork, rainbow_network_args
        )
        self.replay_rref: rpc.RRef[PrioritizedNStepReplayBuffer] = rpc.remote(
            self.replay_worker_info,
            PrioritizedNStepReplayBuffer,
            args=None,
            kwargs=replay_buffer_args,
        )

        print("running tests...")
        # print("target network owner", self.target_network_rref.owner_name())
        # print("online network owner", self.online_network_rref.owner_name())
        # print("replay owner", self.replay_rref.owner_name())
        print("target network confirmed: ", self.target_network_rref.confirmed_by_owner())
        print("online network confirmed: ", self.online_network_rref.confirmed_by_owner())
        print("replay confirmed: ", self.replay_rref.confirmed_by_owner())

        while (
            (not self.target_network_rref.confirmed_by_owner)
            or (not self.online_network_rref.confirmed_by_owner())
            or (not self.replay_rref.confirmed_by_owner())
        ):
            print("waiting for confirmation")
            time.sleep(1)
        print("confirmed.")

        # if the start training step is not zero, attempt to load states from a checkpoint onto the remote server
        if self.start_training_step != 0:
            raise "starting from checkpoint not implemented yet"
            # self.load_from_checkpoint() ...

        # update the remote references with the current weights
        self.store_weights()

    def on_save(self):
        self.replay_buffer = self.replay_rref.to_here(0)

    def on_load(self):
        self.store_weights()
        # TODO - send a copy of the replay buffer state to the remote replay server
        # self.replay_rref.rpc_sync()...

    def store_weights(self):
        # TODO - async
        logger.info("storing weights")
        failed = True
        while failed:
            try:
                self.target_network_rref.rpc_async().load_state_dict(
                    self.target_model.state_dict()
                )
                self.online_network_rref.rpc_async().load_state_dict(
                    self.model.state_dict()
                )
                failed = False
            except Exception as e:
                logger.exception(f"error setting weights: {e}")
                time.sleep(1)

    def update_replay_priorities(self, samples, priorities):
        self.updates_queue.put(
            Update(
                ids=np.array(samples["ids"]),
                indices=np.array(samples["indices"]),
                priorities=np.array(priorities),
            )
        )

    def on_run(self):
        self.flag = threading.Event()

        self.replay_thread = threading.Thread(
            target=self._handle_replay_socket, args=(self.flag,)
        )
        self.replay_thread.daemon = True
        self.replay_thread.start()

        # actors
        for i in range(1, self.config.num_actors):
            self._start_actor(i, False)

        # spectator
        self._start_actor(0, True)

    def _start_actor(self, actor_num: int, spectator: bool) -> torch.Future:
        env_copy = copy.deepcopy(self.env)
        config_copy = copy.deepcopy(self.config.actor_config)
        config_copy.config_dict["rank"] = actor_num + 3
        config_copy.rank = actor_num
        worker_info = rpc.get_worker_info(f"actor_{actor_num}")
        args = (
            env_copy,
            config_copy,
            f"actor_{actor_num}" if not spectator else "spectator",
            self.replay_rref,
            self.online_network_rref,
            self.target_network_rref,
            spectator,
        )
        remote_actor_rref: rpc.RRef[ApeXActor] = rpc.remote(worker_info, ApeXActor, args)

        # no timeout
        res: torch.Future[None] = remote_actor_rref.rpc_async(timeout=0).learn()
        return res

    def on_done(self):
        self.flag.set()
        self.replay_thread.join()

        rpc.shutdown()

    def on_training_step_end(self):
        super().on_training_step_end()

        # This beta gets send over to the remote replay buffer
        try:
            self.replay_rref.remote(30).set_beta(
                update_per_beta(
                    self.per_sample_beta,
                    self.config.per_beta_final,
                    self.training_steps,
                )
            )
        except Exception as e:
            logger.exception(f"error updating remote PER beta: {e}")

    def _handle_replay_socket(self, flag: threading.Event):
        active = False
        while not flag.is_set():
            active = False
            try:
                if self.samples_queue.qsize() < self.config.samples_queue_size:
                    logger.info("requesting batch")
                    samples_rref: rpc.RRef[dict | None] = (
                        self.replay_rref.remote().sample(False)
                    )
                    samples = samples_rref.to_here()
                    if samples == None:  # replay buffer size < min_size
                        logger.info("no batch recieved, continuing and waiting")
                    else:
                        logger.info("recieved batch")

                        self.samples_queue.put(samples)
                        active = True
                else:
                    logger.debug("queue full")
            except Exception as e:
                logger.exception(f"error getting batch: {e}")

            try:
                t = self.updates_queue.get(block=False)  # (indices, priorities, ids)
                active = True
                self.replay_rref.rpc_sync().update_priorities(*t)
            except queue.Empty:
                logger.debug("no updates to send, continuing")
            except Exception as e:
                logger.exception(f"error updating priorities: {e}")

            if not active:
                time.sleep(1)

    def learn_from_sample(self, samples: dict):
        ids, indices, observations, next_observations, rewards, weights, actions = (
            samples["ids"],
            samples["indices"],
            samples["observations"],
            samples["next_observations"],
            samples["rewards"],
            samples["weights"],
            torch.from_numpy(samples["actions"]).to(self.device).long(),
        )
        predicted_distributions = self.predict(
            self.preprocess(next_observations, device=self.device)
        )
        bootstrapped = (self.config.discount_factor**self.config.n_step) * (
            self.predict_target(self.preprocess(observations, device=self.device))
            * self.support
        )[0, self.select_actions(predicted_distributions)].sum(dim=1)
        Gt = rewards + bootstrapped  # already discounted

        predicted_q = (predicted_distributions * self.support)[
            range(self.config.minibatch_size), actions
        ].sum(dim=1)

        weights_cuda = torch.from_numpy(weights).to(torch.float32).to(self.device)
        elementwise_loss = 1 / 2 * (Gt - predicted_q).square()
        loss = elementwise_loss * weights_cuda
        self.optimizer.zero_grad()
        loss.mean().backward()
        if self.config.clipnorm > 0:
            # print("clipnorm", self.config.clipnorm)
            clip_grad_norm_(self.model.parameters(), self.config.clipnorm)

        self.optimizer.step()
        self.update_replay_priorities(
            samples=samples,
            priorities=elementwise_loss.detach().to("cpu").numpy()
            + self.config.per_epsilon,
        )
        self.model.reset_noise()
        self.target_model.reset_noise()
        return loss.detach().to("cpu").mean().item()
