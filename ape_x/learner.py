import sys
import logging
import time
import tensorflow as tf
import numpy as np
import queue
import threading
from compress_utils import compress
from abc import ABC, abstractclassmethod
from typing import NamedTuple
from compress_utils import decompress

sys.path.append("../")
from rainbow.rainbow_agent import RainbowAgent

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


class LearnerBase(RainbowAgent):
    def __init__(self, env, config):
        super().__init__(model_name="learner", env=env, config=config)
        self.graph_interval = 200
        self.remove_old_experiences_interval = config["remove_old_experiences_interval"]
        self.push_weights_interval = config["push_weights_interval"]

        self.samples_queue_size = config["samples_queue_size"]
        self.samples_queue: queue.Queue[Sample] = queue.Queue(
            maxsize=self.samples_queue_size
        )
        self.updates_queue_size = config["updates_queue_size"]
        self.updates_queue: queue.Queue[Update] = queue.Queue(
            maxsize=self.updates_queue_size
        )

    def on_run(self):
        pass

    def on_done(self):
        pass

    def _experience_replay(self):
        with tf.GradientTape() as tape:
            elementwise_loss = 0
            samples = self.samples_queue.get()
            ids = samples.ids
            indices = samples.indices
            actions = samples.actions
            observations = samples.observations
            weights = samples.weights.reshape(-1, 1)

            inputs = self.prepare_states(observations)
            discount_factor = self.discount_factor

            target_ditributions = self.compute_target_distributions_np(
                samples, discount_factor
            )
            initial_distributions = self.model(inputs)
            distributions_to_train = tf.gather_nd(
                initial_distributions,
                list(zip(range(initial_distributions.shape[0]), actions)),
            )
            elementwise_loss = self.model.loss.call(
                y_pred=distributions_to_train,
                y_true=tf.convert_to_tensor(target_ditributions),
            )
            assert np.all(elementwise_loss) >= 0, "Elementwise Loss: {}".format(
                elementwise_loss
            )

            # if self.use_n_step:
            discount_factor = self.discount_factor**self.n_step
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
            elementwise_loss_n_step = self.model.loss.call(
                y_pred=distributions_to_train,
                y_true=tf.convert_to_tensor(target_ditributions),
            )
            # add the losses together to reduce variance (original paper just uses n_step loss)
            elementwise_loss += elementwise_loss_n_step
            assert np.all(elementwise_loss) >= 0, "Elementwise Loss: {}".format(
                elementwise_loss
            )

            loss = tf.reduce_mean(elementwise_loss * weights)

        logger.info("tape done, training with gradient tape")
        # TRAINING WITH GRADIENT TAPE
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer(
            learning_rate=self.learning_rate,
            epsilon=self.adam_epsilon,
            clipnorm=self.clipnorm,
        ).apply_gradients(grads_and_vars=zip(gradients, self.model.trainable_variables))
        # TRAINING WITH tf.train_on_batch
        # loss = self.model.train_on_batch(samples["observations"], target_ditributions, sample_weight=weights)

        prioritized_loss = elementwise_loss + self.per_epsilon
        # CLIPPING PRIORITIZED LOSS FOR ROUNDING ERRORS OR NEGATIVE LOSSES (IDK HOW WE ARE GETTING NEGATIVE LSOSES)
        prioritized_loss = np.clip(
            prioritized_loss, 0.01, tf.reduce_max(prioritized_loss)
        )

        try:
            self.updates_queue.put(
                Update(ids=ids, indices=indices, losses=prioritized_loss), block=False
            )
        except queue.Full:
            logger.warning("updates queue full, dropping update")
            pass

        self.model.reset_noise()
        self.target_model.reset_noise()
        loss = loss.numpy()
        return loss

    def run(self, graph_interval=20):
        self.on_run()

        logger.info("learner running")
        self.is_test = False
        stat_score = (
            []
        )  # make these num trials divided by graph interval so i dont need to append (to make it faster?)
        stat_test_score = []
        stat_loss = []
        # self.fill_replay_buffer()
        state, _ = self.env.reset()
        model_update_count = 0
        score = 0
        training_step = 0

        while training_step <= self.num_training_steps:
            logger.info(
                f"learner training step: {training_step}/{self.num_training_steps}"
            )
            self.per_beta = min(1.0, self.per_beta + self.per_beta_increase)

            model_update_count += 1
            loss = self._experience_replay()
            logger.info(f"finished exp replay")
            training_step += 1
            stat_loss.append(loss)
            self.update_target_model(model_update_count)

            if training_step % graph_interval == 0:
                # self.save_checkpoint(training_step)
                stat_test_score.append(self.test(num_trials=5))
                self.plot_graph(stat_score, stat_loss, stat_test_score, training_step)

        logger.info("loop done")

        self.plot_graph(stat_score, stat_loss, stat_test_score, training_step)
        self.save_checkpoint(training_step)
        self.env.close()
        self.on_done()


class SingleMachineLearner(LearnerBase):
    def __init__(self, env, config):
        super().__init__(env=env, config=config)

    def consume_weights(self):
        weights = self.weights_queue.get()

        while weights is not None:
            print("push weights to remote here")
            weights = self.weights_queue.get()

            # push sample

        logger.info("consume weights finished")

    def consume_priority_updates(self):
        t = self.priority_updates_queue.get()

        while t is not None:
            logger.debug(f"updating priorities, {t}")
            ids, indices, prioritized_loss = t

            self.replay_buffer.update_priorities(indices, prioritized_loss, ids)
            t = self.priority_updates_queue.get()

        logger.info("consume priority finished")

    def produce_samples(self):
        target_size = self.samples_queue_size / 2

        while not self.finished_event.isSet():
            logger.debug("polling for flag")
            if self.samples_queue.qsize() < target_size:
                if self.replay_buffer.size < self.min_replay_buffer_size:
                    logger.debug(
                        "replay buffer not large enough to sample from, polling"
                    )
                    time.sleep(1)
                    continue
                logger.debug("sampling")
                samples = self.replay_buffer.sample()
                indicies = samples["indices"]
                n_step_samples = self.n_step_replay_buffer.sample_from_indices(indicies)
                self.samples_queue.put((samples, n_step_samples))
            else:
                time.sleep(1)

        logger.info("produce samples finished")

    def remove_old_experiences_from_remote_replay_buffer(self):
        # done atomatically in the replay buffer (old experiences overwritten)
        pass

    def get_weights(self):
        # helper for single machine
        return self.model.get_weights()


import zmq
import message_codes

import entities.replayMemory_capnp as replayMemory_capnp


class DistributedLearner(LearnerBase):
    def __init__(self, env, config):
        super().__init__(env=env, config=config)
        self.updates_queue = queue.Queue()

    def handle_replay_socket(self, flag: threading.Event):
        ctx = zmq.Context()

        replay_socket = ctx.socket(zmq.REQ)
        replay_addr = self.config["replay_addr"]
        replay_port = self.config["replay_port"]
        replay_url = f"tcp://{replay_addr}:{replay_port}"

        logger.info(f"learner connecting to replay buffer at {replay_url}")
        replay_socket.connect(replay_url)

        # alternate between getting samples and updating priorities
        i = 0
        while not flag.is_set():
            logger.info("poll")
            if i == 0:
                if self.samples_queue.qsize() < self.samples_queue_size:
                    logger.info("requesting batch")
                    replay_socket.send(message_codes.LEARNER_REQUESTS_BATCH)
                    logger.info("wating for batch")
                    res = replay_socket.recv()
                    if res == b"":  # empty message
                        logger.info("no batch recieved, continuing and waiting")
                        time.sleep(1)
                        continue
                    logger.info("recieved batch")

                    samples = replayMemory_capnp.TransitionBatch.from_bytes_packed(res)
                    logger.info(f"recieved samples {samples}")

                    self.samples_queue.put(
                        Sample(
                            ids=np.array(samples.ids),
                            indices=np.array(samples.indices),
                            actions=np.array(samples.actions),
                            observations=decompress(samples.observations),
                            next_observations=decompress(samples.nextObservations),
                            weights=np.array(samples.weights),
                            rewards=np.array(samples.rewards),
                            dones=np.array(samples.dones),
                        )
                    )
                i = 1
            elif i == 1:
                try:
                    t = self.updates_queue.get(timeout=0.1)
                except queue.Empty:
                    logger.info("no updates to send, continuing")
                    continue
                ids, indices, losses = t
                update = replayMemory_capnp.PriorityUpdate.new_message()
                update.ids = ids.tolist()
                update.indices = indices.tolist()
                update.losses = losses.tolist()

                replay_socket.send(message_codes.LEARNER_UPDATE_PRIORITIES, zmq.SNDMORE)
                replay_socket.send(update.to_bytes())
                replay_socket.recv()
                i = 0

    def handle_learner_requests(self, flag: threading.Event):
        ctx = zmq.Context()
        port = self.config["port"]

        learner_socket = ctx.socket(zmq.REP)

        learner_socket.bind(f"tcp://*:{port}")
        logger.info(f"learner started on port {port}")

        while not flag.is_set():
            message = learner_socket.recv()
            if message == message_codes.ACTOR_GET_PARAMS:
                weights = self.model.get_weights()
                learner_socket.send(compress(weights))
            else:
                learner_socket.send(b"")

    def on_run(self):
        self.flag = threading.Event()

        self.replay_thread = threading.Thread(
            target=self.handle_replay_socket, args=(self.flag,)
        )
        self.learner_thread = threading.Thread(
            target=self.handle_learner_requests, args=(self.flag,)
        )
        self.replay_thread.start()
        self.learner_thread.start()

    def on_done(self):
        self.flag.set()
        self.replay_thread.join()
        self.learner_thread.join()
