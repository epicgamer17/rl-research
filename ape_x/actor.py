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
        self.transitions_queue = queue.Queue(
            3
        )  # max about of transition batches to queue before waiting

    def _fetch_latest_params(self):
        t = time.time()
        self.fetch_latest_params()
        delta_t = time.time() - t
        logging.info(f"fetch_latest_params took: {delta_t} s")

    # to be implemented by subclasses
    def fetch_latest_params(self):
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
        self.consume_transitions_queue()

        self.is_test = False
        self.fetch_latest_params()
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
                self.transitions_queue.put((n_step_samples, priorities))

            self.per_beta = min(1.0, self.per_beta + self.per_beta_increase)

            if done:
                state, _ = self.env.reset()
                state = state
                stat_score.append(score)
                score = 0

            if (training_step % self.poll_params_interval) == 0:
                # launch background thread to fetch latest params from learner
                thread = threading.Thread(target=self._fetch_latest_params)
                thread.start()

            training_step += 1

        self.env.close()
        self.transitions_queue.put(None)
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

    def fetch_latest_params(self):
        logger.info(f" {self.model_name} fetching latest params from learner")
        return self.learner.get_weights()

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
            self.learner.replay_buffer.store_with_priority(
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
import socket
import capnp
import uuid
import pickle

capnp.remove_import_hook()
replay_memory_capnp = capnp.load("./entities/replayMemory.capnp")


class TransitionPusher:
    def __init__(self):
        self.replay_memory = capnp.TwoPartyClient("localhost:60000")
        self.replay_memory = self.replay_memory.bootstrap().cast_as(
            replay_memory_capnp.ReplayMemory
        )

    def push_batch(self, batch):
        t = time.time()
        n = len(batch)
        logger.info(f"pushing {n} experiences to remote replay buffer")

        def on_complete(v):
            delta_t = time.time() - t
            logger.info(f"push_batch took: {delta_t} s")

        promise = self.replay_memory.addTransitionBatch(batch)
        promise.then(on_complete).wait()


class DistributedActor(ActorBase):
    def __init__(
        self,
        id,
        env,
        config,
    ):
        super().__init__(id, env, config)

    def fetch_latest_params(self):
        return self.model.get_weights()

    def consume_transitions_queue(self):
        thread = threading.Thread(target=self._consume_transitions_queue)
        thread.start()

    def _consume_transitions_queue(self):
        pusher = TransitionPusher()
        t = self.transitions_queue.get()

        while t is not None:
            batch = list()
            (experiences, priorities) = t
            n = len(experiences["observations"])
            for t in range(n):
                transition = {
                    "id": f"{self.id}-{uuid.uuid4()}",
                    "observation": pickle.dumps(experiences["observations"][t]),
                    "nextObservation": pickle.dumps(
                        experiences["next_observations"][t]
                    ),
                    "action": experiences["actions"][t].item(),
                    "reward": experiences["rewards"][t].item(),
                    "done": experiences["dones"][t].item() == 1,
                    "priority": priorities[t].item(),
                }
                batch.append(transition)

            pusher.push_batch({"transitions": batch})
            t = self.transitions_queue.get()
