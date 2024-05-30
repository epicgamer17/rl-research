import io
import sys
import logging
import time
import torch
from torch.nn.utils import clip_grad_norm_
import numpy as np
import queue
import threading
from typing import NamedTuple
from agent_configs import ApeXLearnerConfig

from utils import update_per_beta

sys.path.append("../")
from dqn.rainbow.rainbow_agent import RainbowAgent
from storage.storage import Storage, StorageConfig
from storage.compress_utils import decompress
from storage.compress_utils import compress

import matplotlib

matplotlib.use("Agg")


logger = logging.getLogger(__name__)


class Sample(NamedTuple):
    observations: np.ndarray
    actions: np.ndarray
    next_observations: np.ndarray
    rewards: np.ndarray
    ids: np.ndarray
    indices: np.ndarray
    weights: np.ndarray
    dones: np.ndarray


class Update(NamedTuple):
    ids: np.ndarray
    indices: np.ndarray
    losses: np.ndarray


class ApeXLearnerBase(RainbowAgent):
    def __init__(self, env, config: ApeXLearnerConfig, name):
        super().__init__(
            env,
            config,
            name,
        )
        self.config = config
        self.samples_queue: queue.Queue[Sample] = queue.Queue(
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

    def store_weights(self, weights):
        pass

    def on_run(self):
        pass

    def on_done(self):
        pass

    def _learn(self):
        ti = time.time()

        samples = self.samples_queue.get()
        observations, actions, weights, indices, ids = (
            samples.observations,
            torch.from_numpy(samples.actions).to(self.device).long(),
            torch.from_numpy(samples.weights).to(self.device).to(torch.float32),
            samples.indices,
            samples.ids,
        )
        compute_target_distributions_input = {
            "next_observations": samples.next_observations,
            "rewards": samples.rewards,
            "dones": samples.dones,
        }
        online_distributions = self.predict(observations)[
            range(self.config.minibatch_size), actions
        ]
        target_distributions = self.compute_target_distributions(
            compute_target_distributions_input
        )
        elementwise_loss = self.config.loss_function(
            online_distributions, target_distributions
        )
        assert torch.all(elementwise_loss) >= 0, "Elementwise Loss: {}".format(
            elementwise_loss
        )
        assert elementwise_loss.shape == weights.shape, "Loss Shape: {}".format(
            elementwise_loss.shape
        )
        weighted_loss = elementwise_loss * weights
        self.optimizer.zero_grad()
        weighted_loss.mean().backward()
        if self.config.clipnorm:
            clip_grad_norm_(self.model.parameters(), self.config.clipnorm)

        self.optimizer.step()
        loss_for_prior: torch.Tensor = (
            elementwise_loss.detach().to("cpu").numpy() + self.config.per_epsilon
        )

        try:
            self.updates_queue.put(
                Update(ids=ids, indices=indices, losses=loss_for_prior.numpy()),
                block=False,
            )
        except queue.Full:
            logger.warning("updates queue full, dropping update")
            pass

        self.model.reset_noise()
        self.target_model.reset_noise()
        logger.info(f"experience replay took {time.time()-ti} s")
        return weighted_loss.detach().to("cpu").mean().item()

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
                    logger.info("pushed params")

                # TODO - remote update PER beta
                self.replay_buffer.beta = update_per_beta(
                    self.replay_buffer.beta, 1.0, self.training_steps
                )

                loss = self._learn()
                target_model_updated = False
                if training_step % self.config.transfer_interval == 0:
                    # target_model_updated = True
                    self.update_target_model(training_step)
                    target_model_updated = True

                self.stats["loss"].append(
                    {"loss": loss, "target_model_updated": target_model_updated}
                )

                if training_step % self.checkpoint_interval == 0:
                    self.save_checkpoint(
                        5, training_step, training_step, time.time() - start_time
                    )

                    if training_step // self.training_steps > 0.125:
                        past_scores_dicts = self.stats["test_score"][-5:]
                        scores = [score_dict["score"] for score_dict in past_scores_dicts]
                        avg = np.sum(scores) / 5
                        if avg < 10:
                            return  # could do stopping param as the slope of line of best fit

            logger.info("loop done")

            self.save_checkpoint(
                5, training_step, training_step, time.time() - start_time
            )
        except Exception as e:
            print(e)
        finally:
            self.env.close()
            self.on_done()


import zmq
import message_codes

import entities.replayMemory_capnp as replayMemory_capnp


class ApeXLearner(ApeXLearnerBase):
    def __init__(self, env, config: ApeXLearnerConfig, name: str):
        super().__init__(env, config, name)
        self.updates_queue = queue.Queue()
        self.config = config

        storage_config = StorageConfig(
            hostname=self.config.storage_hostname,
            port=self.config.storage_port,
            username=self.config.storage_username,
            password=self.config.storage_password,
        )

        # the learner will reset the storage's model weights on initialization
        self.storage = Storage(storage_config, reset=True)

    def handle_replay_socket(self, flag: threading.Event):
        ctx = zmq.Context()

        replay_socket = ctx.socket(zmq.REQ)
        replay_url = f"tcp://{self.config.replay_addr}:{self.config.replay_port}"

        logger.info(f"learner connecting to replay buffer at {replay_url}")
        replay_socket.connect(replay_url)

        # alternate between getting samples and updating priorities
        while not flag.is_set():
            active = False
            if self.samples_queue.qsize() < self.config.samples_queue_size:
                logger.info("requesting batch")
                replay_socket.send(message_codes.LEARNER_REQUESTS_BATCH, zmq.SNDMORE)
                replay_socket.send_string(str(self.config.per_beta))

                logger.info("wating for batch")
                res = replay_socket.recv()
                if res == b"":  # empty message
                    logger.info("no batch recieved, continuing and waiting")
                    time.sleep(1)
                    continue
                logger.info("recieved batch")

                samples = replayMemory_capnp.TransitionBatch.from_bytes_packed(res)

                self.samples_queue.put(
                    Sample(
                        ids=np.array(samples.ids, dtype=np.object_),
                        indices=np.array(samples.indices),
                        actions=np.array(samples.actions),
                        observations=decompress(samples.observations),
                        next_observations=decompress(samples.nextObservations),
                        weights=np.array(samples.weights),
                        rewards=np.array(samples.rewards),
                        dones=np.array(samples.dones),
                    )
                )
                active = True
            else:
                logger.debug("queue full")

            try:
                t = self.updates_queue.get(block=False)
                active = True
                ids, indices, losses = t
                update = replayMemory_capnp.PriorityUpdate.new_message()
                update.ids = ids.tolist()
                update.indices = indices.astype(int).tolist()

                update.losses = losses.astype(float).tolist()

                replay_socket.send(message_codes.LEARNER_UPDATE_PRIORITIES, zmq.SNDMORE)
                replay_socket.send(update.to_bytes_packed())
                replay_socket.recv()
            except queue.Empty:
                logger.debug("no updates to send, continuing")

            if not active:
                time.sleep(1)

    def on_save(self):
        pass
        # trigger replay buffer save to file with zmq

    def on_load(self):
        self.store_weights()
        # trigger replay buffer load from file with zmq

    def store_weights(self):
        self.storage.store_models(self.model, self.target_model)

    def on_run(self):
        self.flag = threading.Event()

        state, info = self.env.reset()
        self.select_actions(state)
        self.replay_thread = threading.Thread(
            target=self.handle_replay_socket, args=(self.flag,)
        )
        self.replay_thread.daemon = True
        self.replay_thread.start()

    def on_done(self):
        self.flag.set()
        self.replay_thread.join()
