import sys
import logging
import time
import tensorflow as tf
import numpy as np
import queue
import threading
from typing import NamedTuple
from agent_configs import ApeXLearnerConfig

from utils import update_per_beta

sys.path.append("../")
from rainbow.rainbow_agent import RainbowAgent
from storage.storage import Storage, StorageConfig
from storage.compress_utils import decompress
from storage.compress_utils import compress

import matplotlib

matplotlib.use("Agg")


logger = logging.getLogger(__name__)


class Sample(NamedTuple):
    ids: np.ndarray
    indices: np.ndarray
    actions: np.ndarray
    observations: np.ndarray
    weights: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
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

    def store_weights(self, weights):
        pass

    def on_run(self):
        pass

    def on_done(self):
        pass

    def _experience_replay(self):
        ti = time.time()
        with tf.GradientTape() as tape:
            samples = self.samples_queue.get()
            ids = samples.ids
            indices = samples.indices
            weights = samples.weights.reshape(-1, 1)
            # print(weights)
            # if self.use_n_step:
            discount_factor = self.config.discount_factor**self.config.n_step
            actions = samples.actions
            n_step_observations = samples.observations
            observations = n_step_observations
            inputs = self.prepare_states(observations)
            target_ditributions = self.compute_target_distributions_np(
                samples, discount_factor
            )
            initial_distributions = self.model(inputs)
            distributions_to_train = tf.gather_nd(
                initial_distributions,
                list(zip(range(initial_distributions.shape[0]), actions)),
            )
            elementwise_loss = self.config.loss_function.call(
                y_pred=distributions_to_train,
                y_true=tf.convert_to_tensor(target_ditributions),
            )
            # add the losses together to reduce variance (original paper just uses n_step loss)
            assert np.all(elementwise_loss) >= 0, "Elementwise Loss: {}".format(
                elementwise_loss
            )

            loss = tf.reduce_mean(elementwise_loss * weights)

        logger.info("tape done, training with gradient tape")
        # TRAINING WITH GRADIENT TAPE
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.config.optimizer.apply_gradients(
            grads_and_vars=zip(gradients, self.model.trainable_variables)
        )
        # TRAINING WITH tf.train_on_batch
        # loss = self.model.train_on_batch(samples["observations"], target_ditributions, sample_weight=weights)

        prioritized_loss = elementwise_loss + self.config.per_epsilon
        # CLIPPING PRIORITIZED LOSS FOR ROUNDING ERRORS OR NEGATIVE LOSSES (IDK HOW WE ARE GETTING NEGATIVE LSOSES)
        # prioritized_loss = np.clip(
        #     prioritized_loss, 0.01, tf.reduce_max(prioritized_loss)
        # )

        try:
            self.updates_queue.put(
                Update(ids=ids, indices=indices, losses=prioritized_loss.numpy()),
                block=False,
            )
        except queue.Full:
            logger.warning("updates queue full, dropping update")
            pass

        self.model.reset_noise()
        self.target_model.reset_noise()
        loss = loss.numpy()

        logger.info(f"experience replay took {time.time()-ti} s")
        return loss

    def run(self):
        try:
            start_time = time.time()
            self.on_run()

            logger.info("learner running")
            self.is_test = False
            stats = {
                "loss": [],
                "test_score": [],
            }
            targets = {
                "test_score": self.env.spec.reward_threshold,
            }
            # target_model_updated = False
            # self.fill_replay_buffer()
            state, info = self.env.reset()

            for training_step in range(self.training_steps + 1):
                # stop training if going over 1.5 hours
                if time.time() - start_time > 3600 * 1.5:
                    break

                logger.info(
                    f"learner training step: {training_step}/{self.training_steps}"
                )

                if training_step % self.config.push_params_interval == 0:
                    self.store_weights()
                    logger.info("pushed params")

                self.config.per_beta = update_per_beta(
                    self.config.per_beta, 1.0, self.training_steps
                )

                loss = self._experience_replay()
                logger.info(f"finished exp replay")
                stats["loss"].append(loss)
                if training_step % self.config.transfer_interval == 0:
                    # target_model_updated = True
                    self.update_target_model(training_step)

                if training_step % self.checkpoint_interval == 0:
                    self.save_checkpoint(
                        stats,
                        targets,
                        5,
                        training_step,
                        training_step,
                        time.time() - start_time,
                    )

                    if training_step // self.training_steps > 0.125:
                        past_scores = stats["test_score"][-5:][
                            "score"
                        ]  # this may not work, you might need to convert it to a list first like in utils plots :)
                        avg = np.sum(past_scores) / 5
                        if avg < 10:
                            return

            logger.info("loop done")

            self.save_checkpoint(
                stats,
                targets,
                5,
                training_step,
                training_step,
                time.time() - start_time,
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

        # the learner will reset the storage for model weights on initialization
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

    def store_weights(self):
        weights = self.model.get_weights()
        # print(weights)
        compressed = compress(weights)
        logger.info("storing weights")
        self.storage.store_weights(compressed)

    def on_run(self):
        self.flag = threading.Event()

        state, info = self.env.reset()
        self.select_action(state)
        self.replay_thread = threading.Thread(
            target=self.handle_replay_socket, args=(self.flag,)
        )
        self.replay_thread.daemon = True
        self.replay_thread.start()

    def on_done(self):
        self.flag.set()
        self.replay_thread.join()
