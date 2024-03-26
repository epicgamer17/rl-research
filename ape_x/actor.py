import sys
import tensorflow as tf
import numpy as np
import threading
import time
import logging
import learner
import gymnasium as gym
import queue

sys.path.append("../")

from rainbow.rainbow_agent import RainbowAgent


logger = logging.getLogger(__name__)


class ActorBase(RainbowAgent):
    def __init__(
        self,
        id,
        env,
        config,
    ):
        # override local replay config
        config["min_replay_buffer_size"] = config["buffer_size"]
        config["replay_buffer_size"] = config["buffer_size"]
        config["n_step"] = config["buffer_size"]
        config["replay_batch_size"] = config["buffer_size"]

        super().__init__(model_name=f"actor_{id}", env=env, config=config)
        self.id = id
        self.poll_params_interval = config["poll_params_interval"]
        self.buffer_size = config["buffer_size"]

        # max about of transition batches to queue before dropping
        self.max_transitions_queue_size = 16
        self.transitions_queue = queue.Queue(maxsize=self.max_transitions_queue_size)
        self.params_queue = queue.Queue(maxsize=1)

    def produce_weight_updates(self):
        pass

    # to be implemented by subclasses
    def consume_transitions_queue(self):
        pass

    def calculate_losses(self, indices):
        print("calculating losses...")
        t = time.time()
        elementwise_loss = 0
        samples = self.replay_buffer.sample_from_indices(indices)
        actions = samples["actions"]
        observations = samples["observations"]
        inputs = self.prepare_states(observations)
        discount_factor = self.discount_factor
        target_distributions = self.compute_target_distributions(
            samples, discount_factor
        )
        initial_distributions = self.model(inputs)
        distributions_to_train = tf.gather_nd(
            initial_distributions,
            list(zip(range(initial_distributions.shape[0]), actions)),
        )
        elementwise_loss = self.model.loss.call(
            y_pred=distributions_to_train,
            y_true=tf.convert_to_tensor(target_distributions),
        )
        assert np.all(elementwise_loss) >= 0, "Elementwise Loss: {}".format(
            elementwise_loss
        )
        # if self.use_n_step:
        discount_factor = self.discount_factor**self.n_step
        n_step_samples = self.n_step_replay_buffer.sample_from_indices(indices)
        actions = n_step_samples["actions"]
        observations = n_step_samples["observations"]
        # observations = n_step_observations
        inputs = self.prepare_states(observations)
        target_distributions = self.compute_target_distributions(
            n_step_samples, discount_factor
        )
        initial_distributions = self.model(inputs)
        distributions_to_train = tf.gather_nd(
            initial_distributions,
            list(zip(range(initial_distributions.shape[0]), actions)),
        )
        elementwise_loss_n_step = self.model.loss.call(
            y_pred=distributions_to_train,
            y_true=tf.convert_to_tensor(target_distributions),
        )
        # add the losses together to reduce variance (original paper just uses n_step loss)
        elementwise_loss += elementwise_loss_n_step
        assert np.all(elementwise_loss) >= 0, "Elementwise Loss: {}".format(
            elementwise_loss
        )

        prioritized_loss = elementwise_loss + self.per_epsilon
        # CLIPPING PRIORITIZED LOSS FOR ROUNDING ERRORS OR NEGATIVE LOSSES (IDK HOW WE ARE GETTING NEGATIVE LSOSES)
        prioritized_loss = np.clip(
            prioritized_loss, 0.01, tf.reduce_max(prioritized_loss)
        )

        delta_t = time.time() - t
        logger.info(f"calculate_losses took: {delta_t} s")
        return prioritized_loss

    def run(self):
        consume_queue = threading.Thread(target=self.consume_transitions_queue)
        produce_queue = threading.Thread(target=self.produce_weight_updates)
        consume_queue.start()
        produce_queue.start()

        self.is_test = False
        self.model.set_weights(self.params_queue.get())
        print("filling replay buffer...")
        self.fill_replay_buffer()

        state, _ = self.env.reset()
        score = 0
        stat_score = []
        num_trials_truncated = 0

        training_step = 0
        while training_step <= self.num_training_steps:
            # logger.info( f"{self.model_name} training step: {training_step}/{self.num_training_steps}")

            state_input = self.prepare_states(state)
            action = self.select_action(state_input)
            next_state, reward, terminated, truncated = self.step(action)
            done = terminated or truncated
            state = next_state
            score += reward

            self.replay_buffer.store(state, action, reward, next_state, done)

            if training_step % self.replay_buffer_size == 0 and training_step > 0:
                logger.info(
                    f"{self.model_name} training step: {training_step}/{self.num_training_steps}"
                )
                indices = list(range(self.replay_batch_size))
                n_step_samples = self.n_step_replay_buffer.sample_from_indices(indices)
                prioritized_loss = self.calculate_losses(indices)
                priorities = self.replay_buffer.update_priorities(
                    indices, prioritized_loss
                )

                # push to transitions queue. If queue is full and does not have space after 5 seconds, drop the batch
                try:
                    self.transitions_queue.put((n_step_samples, priorities), timeout=5)
                except queue.Full:
                    logger.warn(
                        f"{self.model_name} transitions queue full, dropped batch"
                    )

            self.per_beta = min(1.0, self.per_beta + self.per_beta_increase)

            if done:
                state, _ = self.env.reset()
                state = state
                stat_score.append(score)
                score = 0

            if (training_step % self.poll_params_interval) == 0:
                weights = self.model.get_weights()
                if len(weights) == 0:
                    logger.warn("model weights empty")
                else:
                    self.model.set_weights(weights)

            training_step += 1

        self.env.close()
        self.transitions_queue.put(None)

        # wait for consume_transitions_queue to finish
        logger.debug("Waiting for consume_transitions_queue to finish")
        consume_queue.join()

        return num_trials_truncated / self.num_training_steps


class SingleMachineActor(ActorBase):
    def __init__(
        self,
        id,
        env,
        config,
        single_machine_learner: learner.SingleMachineLearner = None,  # TODO: change this to single machine learner
    ):
        super().__init__(id, env, config)
        self.learner = single_machine_learner

    def produce_weight_updates(self):
        pass

    def consume_transitions_queue(self):
        thread = threading.Thread(target=self._consume_transitions_queue)
        thread.start()

    def _consume_transitions_queue(self):
        i = self.transitions_queue.get()

        while i is not None:
            self.push_experiences_to_remote_replay_buffer(i[0], i[1])
            i = self.transitions_queue.get()

    def push_experiences_to_remote_replay_buffer(self, experiences, priorities):
        t = time.time()
        n = len(experiences["observations"])
        logger.info(
            f" {self.model_name} pushing {n} experiences to remote replay buffer"
        )

        for i in range(n):
            self.learner.replay_buffer.store_with_priority_exact(
                experiences["observations"][i],
                experiences["actions"][i],
                experiences["rewards"][i],
                experiences["next_observations"][i],
                experiences["dones"][i],
                priorities[i],
            )

        delta_t = time.time() - t
        print("learner replay buffer size: ", self.learner.replay_buffer.size)
        logger.info(f"push_experiences_to_remote_replay_buffer took: {delta_t} s")


import asyncio
import pickle
import uuid
from client import ReplayMemoryClient


class ActorRPCClient:
    def __init__(self, exp_q: queue.Queue, params_q: queue.Queue, actor_id: str):
        self.exp_q = exp_q
        self.params_q = params_q
        self.id = actor_id

        self.client = ReplayMemoryClient()
        self.client.extra_coroutines.append(self.produce_params())
        self.client.extra_coroutines.append(self.consume_transitions())

    def prepare_request(self, id, batch):
        (experiences, priorities) = batch
        n = len(experiences["dones"])
        logger.info(f"pushing {n} experiences to remote replay buffer")

        ids = list()
        actions = list()
        rewards = list()
        dones = list()
        pri = list()

        for i in range(n):
            ids.append(f"{id}-{uuid.uuid4()}")
            actions.append(experiences["actions"][i].item())
            rewards.append(experiences["rewards"][i].item())
            dones.append(experiences["dones"][i].item() == 1)
            pri.append(priorities[i].item())

        request = self.client.get_rpc().addTransitionBatch_request()
        # request.batch = input
        request.batch.ids = ids
        request.batch.observations = pickle.dumps(
            experiences["observations"], protocol=5
        )
        request.batch.nextObservations = pickle.dumps(
            experiences["next_observations"], protocol=5
        )
        request.batch.actions = actions
        request.batch.rewards = rewards
        request.batch.dones = dones
        request.batch.priorities = pri

        return request

    async def consume_transitions(self):
        logger.debug("push transitions started.")
        while self.client.running:
            try:
                t = self.exp_q.get(block=False)
                logger.debug(f"got batch from queue: {t}")
                if t is None:
                    logger.info(
                        "recieved finished signal, finishing task and triggering stop sequence"
                    )
                    self.client.stop()
                    return True

                logger.debug("sending batch to learner")
                t_i = time.time()
                request = self.prepare_request(self.id, t)

                await asyncio.wait_for(request.send().a_wait(), timeout=5)

                logger.info(f"push_batch took: {time.time() - t_i}s")
            except queue.Empty:
                logger.debug("no batch in queue, retrying...")
                await asyncio.sleep(0.1)
            except asyncio.TimeoutError:
                logger.warn("push_batch timed out - batch dropped!")
                await asyncio.sleep(0.1)
            except Exception as e:
                # unexpected error - immediately stop the client and finish the task

                logger.exception(f"Error pushing batch: {e}")
                self.client.stop()
                return False

        return True

    async def produce_params(self):
        logger.debug("get params started.")
        while self.client.running:
            try:
                logger.info("getting latest network params from learner")
                t_i = time.time()

                request = self.client.get_rpc().getWeights_request()
                res = await asyncio.wait_for(request.send().a_wait(), timeout=5)
                unpickled_params = pickle.loads(res.weights)
                print("got unpickeld params: ", unpickled_params)

                if unpickled_params is not None:
                    self.params_q.put(unpickled_params)
                    logger.debug(f"produce_params took: {time.time() - t_i}s")
                else:
                    print("no params recieved, retrying after 0.1s ...")
                    await asyncio.sleep(0.1)

            except asyncio.TimeoutError:
                logger.warning("getWeights timed out, retrying...")
            except Exception as e:
                logger.exception(f"Error getting params: {e}")
                self.client.stop()
                return False

        logger.info("produce params finished sucessfully")
        return True


class DistributedActor(ActorBase):
    def __init__(
        self,
        id,
        env,
        config,
    ):
        super().__init__(id, env, config)

    def produce_weight_updates(self):
        # included with consume_transitions_queue
        pass

    def consume_transitions_queue(self):
        # pusher must be created in the same thread as the event loop, which is where the capnp rpc objects are created
        pusher = ActorRPCClient(self.transitions_queue, self.params_queue, self.id)
        asyncio.run(pusher.client.start(), debug=True)
        logger.debug("transitionPusher done.")
