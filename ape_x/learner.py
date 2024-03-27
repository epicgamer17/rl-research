import sys
import logging
import time
import tensorflow as tf
import numpy as np
import queue
import threading
from compress_utils import compress

sys.path.append("../")
from rainbow.rainbow_agent import RainbowAgent

import matplotlib

matplotlib.use("Agg")


logger = logging.getLogger(__name__)


# signals are sent into queues
# queues are constantly read by the learner rpc thread


class LearnerBase(RainbowAgent):
    def __init__(self, env, config):
        super().__init__(model_name="learner", env=env, config=config)
        self.graph_interval = 200
        self.remove_old_experiences_interval = config["remove_old_experiences_interval"]

        self.push_weights_interval = config["push_weights_interval"]
        self.weights_queue = queue.Queue(1)
        self.priority_updates_queue = queue.Queue(
            16
        )  # contains tuples of (ids, indices, priorities)

        self.samples_queue_size = 64
        self.samples_queue = queue.Queue(
            self.samples_queue_size
        )  # contains tuples of (sample, n_step_sample)
        self.finished_event = threading.Event()

    def consume_weights(self):
        pass

    def produce_samples(self):
        pass

    def consume_priority_updates(self):
        pass

    def remove_old_experiences_from_remote_replay_buffer(self):
        pass

    def _experience_replay(self):
        with tf.GradientTape() as tape:
            elementwise_loss = 0
            logger.info("learner waiting for sample...")
            samples = self.samples_queue.get()
            logger.info("got sample")

            ids = samples["ids"]
            indices = samples["indices"]
            actions = samples["actions"]
            observations = samples["observations"]
            inputs = self.prepare_states(observations)
            weights = np.array(samples["weights"]).reshape(-1, 1)
            discount_factor = self.discount_factor

            target_ditributions = self.compute_target_distributions(
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
            actions = samples["actions"]
            n_step_observations = samples["observations"]
            observations = n_step_observations
            inputs = self.prepare_states(observations)
            target_ditributions = self.compute_target_distributions(
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
            logger.info("pushing update")
            self.priority_updates_queue.put((ids, indices, prioritized_loss))
            logger.info(
                f"new priority_updates_queue size: {self.priority_updates_queue.qsize()}"
            )
        except queue.Full:
            logger.warning(f"priority_updates_queue full, dropping priority update")

        self.model.reset_noise()
        self.target_model.reset_noise()
        loss = loss.numpy()
        return loss

    def run(self, graph_interval=20):
        consume_weights_thread = threading.Thread(target=self.consume_weights)
        produce_samples_thread = threading.Thread(target=self.produce_samples)
        consume_priority_updates_thread = threading.Thread(
            target=self.consume_priority_updates
        )

        consume_weights_thread.start()
        produce_samples_thread.start()
        consume_priority_updates_thread.start()

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

        # push initial weights
        self.weights_queue.put(self.model.get_weights(), False)

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

            if training_step % self.push_weights_interval == 0:
                try:
                    logger.debug("pushing weights to remote")
                    self.weights_queue.put(self.model.get_weights(), False)
                except queue.Full:
                    logger.warning(f"weights_queue full, dropping weight update")

        logger.info("loop done")

        self.priority_updates_queue.put(None)
        self.weights_queue.put(None)

        self.finished_event.set()

        produce_samples_thread.join()
        consume_weights_thread.join()
        consume_priority_updates_thread.join()

        self.plot_graph(stat_score, stat_loss, stat_test_score, training_step)
        self.save_checkpoint(training_step)
        self.env.close()


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


import asyncio
import pickle
from client import ReplayMemoryClient


class LearnerRPCClient:
    def __init__(
        self,
        weights_queue: queue.Queue,
        samples_queue: queue.Queue,
        priority_updates_queue: queue.Queue,
        addr="localhost",
        port=60000,
    ):
        self.client = ReplayMemoryClient(addr=addr, port=port)

        self.weights_queue = weights_queue
        self.samples_queue = samples_queue
        self.priority_updates_queue = priority_updates_queue

        self.client.extra_coroutines.append(self.consume_weights())
        self.client.extra_coroutines.append(self.consume_priority_updates())
        self.client.extra_coroutines.append(self.produce_samples())

    async def consume_weights(self):
        logger.info("consume weights started")
        while self.client.running:
            try:
                logger.info(
                    f"getting weights from queue. queue size: {self.weights_queue.qsize()}"
                )
                weights = self.weights_queue.get(timeout=0.01)
                logger.debug(f"got weights from queue")
                if weights is None:
                    logger.info(
                        "got stop signal, stopping consume_weights and stopping client"
                    )
                    self.client.stop()
                    return True

                logger.debug("pushing weights to remote")
                t_i = time.time()
                compressed = compress(weights)
                request = self.client.get_rpc().setWeights_request()
                request.weights = compressed

                await asyncio.wait_for(request.send().a_wait(), 10)
                logger.info(f"setWeights_request took {time.time() - t_i} seconds")
            except queue.Empty:
                logger.debug("weights_queue empty, polling")
                await asyncio.sleep(0.1)
            except asyncio.TimeoutError:
                logger.warning("timeout pushing weights to remote - dropping update")
            except Exception as e:
                logger.exception(
                    f"error pushing weights to remote {e} - stopping consume_weights"
                )
                return False

        return True

    async def consume_priority_updates(self):
        logger.info("consume priority updates started")
        while self.client.running:
            try:
                logger.info(
                    f"getting batch. queue size: {self.priority_updates_queue.qsize()}"
                )
                t = self.priority_updates_queue.get(block=False)
                logger.debug(f"got priorities from queue, {t}")
                if t is None:
                    logger.info(
                        "got stop signal, stopping consume_priority_updates and stopping client"
                    )
                    self.client.stop()
                    return True

                ids, indices, prioritized_loss = t
                logger.debug("updating priorities to remote")
                t_i = time.time()

                request = self.client.get_rpc().updatePriorities_request()
                request.ids = list(ids)
                request.indices = indices.tolist()
                request.priorities = prioritized_loss.tolist()
                await asyncio.wait_for(request.send().a_wait(), 10)
                logger.info(
                    f"updatePriorities_request took {time.time() - t_i} seconds"
                )
            except queue.Empty:
                logger.info("priority_updates_queue empty, polling")
                await asyncio.sleep(0.1)
            except asyncio.TimeoutError:
                logger.info("timeout updating priorities - dropping update")
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.exception(
                    f"error updating priorities {e} - stopping consume_priority_updates"
                )
                return False

        logger.warning("comsume priority updates finished normally")
        return True

    async def produce_samples(self):
        logger.info("produce samples started")
        while self.client.running:
            try:
                logger.info("getting samples from remote")
                t_i = time.time()

                request = self.client.get_rpc().sample_request()
                request.batchSize = 0
                res = await asyncio.wait_for(request.send().a_wait(), 10)

                logger.debug(f"produce_samples took {time.time() - t_i} seconds")

                sample = {
                    "ids": res.batch.ids,
                }

                if len(sample["ids"]) > 0:
                    sample["ids"] = res.batch.ids
                    sample["indices"] = np.array(res.batch.indices)
                    sample["actions"] = np.array(res.batch.actions)
                    sample["observations"] = pickle.loads(res.batch.observations)
                    sample["rewards"] = np.array(res.batch.rewards)
                    sample["next_observations"] = pickle.loads(
                        res.batch.nextObservations
                    )
                    sample["dones"] = np.array(res.batch.dones)
                    sample["weights"] = np.array(res.batch.weights)
                    logger.debug(
                        f"got samples from remote. new queue size: {self.samples_queue.qsize()} "
                    )

                    self.samples_queue.put(sample)
                else:
                    logger.debug("no samples from remote, polling")
                    await asyncio.sleep(0.1)
            except asyncio.TimeoutError:
                logger.warning("timeout getting samples from remote")
            except Exception as e:
                logger.exception(
                    f"error getting samples {e} - stopping produce_samples"
                )
                return False

        logger.info("produce samples finished successfully")
        return True


class DistributedLearner(LearnerBase):
    def __init__(self, env, config):
        super().__init__(env=env, config=config)
        self.addr = config["capnp_conn"].split(":")[0]
        self.port = config["capnp_conn"].split(":")[1]

    def consume_weights(self):
        c = LearnerRPCClient(
            weights_queue=self.weights_queue,
            samples_queue=self.samples_queue,
            priority_updates_queue=self.priority_updates_queue,
            addr=self.addr,
            port=self.port,
        )
        asyncio.run(c.client.start(), debug=True)
        logger.info("consume weights finished")

    def produce_samples(self):
        # handled in consume_weights
        logger.info("produced samples ignored")

    def consume_priority_updates(self):
        # handled in consume_weights
        logger.info("consume priority updates ignored")

    def remove_old_experiences_from_remote_replay_buffer(self):
        # not needed
        pass
