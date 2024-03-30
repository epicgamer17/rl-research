import sys
import tensorflow as tf
import numpy as np
import time
import logging
from configs.agent_configs.ape_x_config import ApeXConfig
import learner
from abc import ABC, abstractclassmethod
from compress_utils import decompress, compress
import entities.replayMemory_capnp as replayMemory_capnp
from typing import NamedTuple

import uuid

sys.path.append("../")

from rainbow.rainbow_agent import RainbowAgent


logger = logging.getLogger(__name__)


class Batch(NamedTuple):
    observations: np.ndarray
    next_observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    ids: np.ndarray
    priorities: np.ndarray


class ActorBase(RainbowAgent, ABC):
    def __init__(
        self,
        id,
        env,
        config: ApeXConfig,
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

    @abstractclassmethod
    def update_params(self):
        # get the latest weights from the learner
        pass

    @abstractclassmethod
    def push_experience_batch(self, batch: Batch):
        # push the experience batch to the replay buffer
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
        self.is_test = False

        logger.info("fetching initial network params from learner...")
        self.update_params()

        logger.info("filling replay buffer...")
        self.fill_replay_buffer()

        state, _ = self.env.reset()
        score = 0
        stat_score = []
        num_trials_truncated = 0

        training_step = 0
        while training_step <= self.num_training_steps:
            logger.debug(
                f"{self.model_name} training step: {training_step}/{self.num_training_steps}"
            )

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
                # dict(
                #     observations=self.observation_buffer[indices],
                #     next_observations=self.next_observation_buffer[indices],
                #     actions=self.action_buffer[indices],
                #     rewards=self.reward_buffer[indices],
                #     dones=self.done_buffer[indices],
                #     ids=self.id_buffer[indices],
                # )

                ids = list()

                for i in range(len(indices)):
                    ids.append(uuid.uuid4().hex)

                prioritized_loss = self.calculate_losses(indices)
                priorities = self.replay_buffer.update_priorities(
                    indices, prioritized_loss
                )

                batch = Batch(
                    observations=n_step_samples["observations"],
                    next_observations=n_step_samples["next_observations"],
                    actions=n_step_samples["actions"],
                    rewards=n_step_samples["rewards"],
                    dones=n_step_samples["dones"],
                    ids=np.array(ids, dtype=np.object_),
                    priorities=priorities,
                )

                self.push_experience_batch(batch)

            self.per_beta = min(1.0, self.per_beta + self.per_beta_increase)

            if done:
                state, _ = self.env.reset()
                state = state
                stat_score.append(score)
                score = 0

            if (training_step % self.poll_params_interval) == 0:
                self.update_params()

            training_step += 1

        self.env.close()

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

    def update_params(self):
        # get the latest weights from the learner
        pass

    def push_experience_batch(self, batch):
        # push the experience batch to the replay buffer
        pass

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


import zmq
import entities.replayMemory_capnp as replayMemory_capnp
import message_codes


class DistributedActor(ActorBase):
    def __init__(
        self,
        id,
        env,
        config,
    ):
        super().__init__(id, env, config)
        self.socket_ctx = zmq.Context()

<<<<<<< Updated upstream
        learner_address = config["learner_addr"]
        learner_port = config["learner_port"]
        learner_url = f"tcp://{learner_address}:{learner_port}"

        replay_address = config["replay_addr"]
        replay_port = config["replay_port"]
=======
        learner_address = self.config.learner_addr
        learner_port = self.config.learner_port
        learner_url = f"tcp://{learner_address}:{learner_port}"

        replay_address = self.config.replay_addr
        replay_port = self.configreplay_port
>>>>>>> Stashed changes
        replay_url = f"tcp://{replay_address}:{replay_port}"

        self.learner_socket = self.socket_ctx.socket(zmq.REQ)
        self.replay_socket = self.socket_ctx.socket(zmq.PUSH)

        self.learner_socket.connect(learner_url)
        logger.info(f"connected to learner at {learner_url}")
        self.replay_socket.connect(replay_url)
        logger.info(f"connected to replay buffer at {replay_url}")

    def update_params(self):
        self.learner_socket.send(message_codes.ACTOR_GET_PARAMS)
        ti = time.time()
        logger.info("fetching weights from learner...")
        res = self.learner_socket.recv()
        decompressed = decompress(res)

        # todo: fix issues with starting up the weights aren't full
        if len(decompressed) < 24:
            logger.info("not enough weights received from learner")
        else:
            self.model.set_weights(decompressed)
            logger.info(f"fetching weights took {time.time() - ti} s")

    def push_experience_batch(self, batch):
        builder = replayMemory_capnp.TransitionBatch.new_message()
        builder.observations = compress(batch.observations)
        builder.nextObservations = compress(batch.next_observations)
        builder.rewards = batch.rewards.astype(np.float32).tolist()
        builder.actions = batch.actions.astype(np.int32).tolist()
        builder.dones = batch.dones.astype(bool).tolist()
        builder.ids = batch.ids.astype(str).tolist()
        builder.priorities = batch.priorities.astype(np.float32).tolist()

        self.replay_socket.send(message_codes.ACTOR_SEND_BATCH, zmq.SNDMORE)

        res = builder.to_bytes_packed()
        self.replay_socket.send(res)
