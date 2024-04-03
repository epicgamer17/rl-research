import time
import tensorflow as tf
import numpy as np
from typing import NamedTuple
from uuid import uuid4
import zmq
from gymnasium import Env
from base_agent.distributed_agents import ActorAgent, PollingActor
from agent_configs import (
    Config,
    ActorApeXMixin,
    DistributedConfig,
    RainbowConfig,
)
import entities.replayMemory_capnp as replayMemory_capnp
import message_codes
from compress_utils import compress, decompress

import logging

logger = logging.getLogger(__name__)

import sys

sys.path.append("..")
from rainbow.rainbow_agent import RainbowAgent


class TransitionBuffer(NamedTuple):
    observations: np.ndarray
    next_observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray


class Batch(NamedTuple):
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray
    ids: np.ndarray
    priorities: np.ndarray


class ApexActor(ActorAgent, PollingActor):
    """
    Apex Actor base class
    """

    def __init__(
        self,
        env: Env,
        config: Config | ActorApeXMixin,
        name,
    ):
        super().__init__(env, config, name)
        self.config = config
        self.score = 0
        self.stats = dict(score=list())

        self.env_state = None

        observation_buffer_shape = []
        observation_buffer_shape += [self.config.replay_buffer_size]
        observation_buffer_shape += list(self.env.observation_space.shape)
        observation_buffer_shape = list(observation_buffer_shape)
        self.transitions_buffer: TransitionBuffer = TransitionBuffer(
            observations=np.zeros(observation_buffer_shape, dtype=np.float32),
            next_observations=np.zeros(observation_buffer_shape, dtype=np.float32),
            actions=np.zeros(self.config.replay_buffer_size, dtype=np.int32),
            rewards=np.zeros(self.config.replay_buffer_size, dtype=np.float32),
            dones=np.zeros(self.config.replay_buffer_size, dtype=np.bool_),
        )
        self.transitions_buffer_index = 0

    def reset_transitions_buffer(self):
        self.transitions_buffer = TransitionBuffer(
            observations=np.zeros_like(self.transitions_buffer.observations),
            next_observations=np.zeros_like(self.transitions_buffer.next_observations),
            actions=np.zeros_like(self.transitions_buffer.actions),
            rewards=np.zeros_like(self.transitions_buffer.rewards),
            dones=np.zeros_like(self.transitions_buffer.dones),
        )
        self.transitions_buffer_index = 0

    def on_run_start(self):
        super().on_run_start()
        self.env_state, _ = self.env.reset()

    def collect_experience(self):
        state_input = self.prepare_states(self.env_state)
        action = self.select_action(state_input)

        next_state, reward, terminated, truncated, info = super(ActorAgent, self).step(
            action
        )
        done = terminated or truncated
        self.score += reward

        self.transitions_buffer.observations[self.transitions_buffer_index] = (
            self.env_state
        )
        self.transitions_buffer.next_observations[self.transitions_buffer_index] = (
            next_state
        )
        self.transitions_buffer.actions[self.transitions_buffer_index] = action
        self.transitions_buffer.rewards[self.transitions_buffer_index] = reward
        self.transitions_buffer.dones[self.transitions_buffer_index] = done
        self.transitions_buffer_index += 1

        self.env_state = next_state

        if done:
            self.env_state, _ = self.env.reset()
            self.stats["score"].append(self.score)
            self.score = 0

    def should_send_experience_batch(self, training_step: int):
        return (
            training_step % self.config.replay_buffer_size == 0 and training_step != 0
        )

    def should_update_params(self, training_step: int):
        return (
            training_step % self.config.poll_params_interval == 0 and training_step != 0
        )


class DistributedApex(ApexActor, RainbowAgent):
    def __init__(
        self,
        env: Env,
        config: Config | ActorApeXMixin | DistributedConfig | RainbowConfig,
        name,
    ):
        super().__init__(env, config, name)
        self.config = config

        self.socket_ctx = zmq.Context()
        learner_address = self.config.learner_addr
        learner_port = self.config.learner_port
        learner_url = f"tcp://{learner_address}:{learner_port}"

        replay_address = self.config.replay_addr
        replay_port = self.config.replay_port
        replay_url = f"tcp://{replay_address}:{replay_port}"

        self.learner_socket = self.socket_ctx.socket(zmq.REQ)
        self.replay_socket = self.socket_ctx.socket(zmq.PUSH)

        self.learner_socket.connect(learner_url)
        logger.info(f"connected to learner at {learner_url}")
        self.replay_socket.connect(replay_url)
        logger.info(f"connected to replay buffer at {replay_url}")

    def send_experience_batch(self):
        ids = np.zeros(self.config.replay_buffer_size, dtype=np.object_)

        for i in range(self.config.replay_buffer_size):
            ids[i] = uuid4().hex

        batch = Batch(
            observations=self.transitions_buffer.observations,
            actions=self.transitions_buffer.actions,
            rewards=self.transitions_buffer.rewards,
            next_observations=self.transitions_buffer.next_observations,
            dones=self.transitions_buffer.dones,
            ids=ids,
            priorities=self.calculate_loss(self.transitions_buffer)
            ** self.replay_buffer.alpha,
        )

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

        self.reset_transitions_buffer()

    def update_params(self):
        self.learner_socket.send(message_codes.ACTOR_GET_PARAMS)
        ti = time.time()
        logger.info("fetching weights from learner...")
        res = self.learner_socket.recv()
        decompressed = decompress(res)

        # todo: fix issues with starting up the weights aren't full
        # if len(decompressed) < 24:
        #     logger.info("not enough weights received from learner")
        # else:
        try:
            self.model.set_weights(decompressed)
        except Exception as e:
            print(e)
            logger.info("not enough weights received from learner")
        logger.info(f"fetching weights took {time.time() - ti} s")

    def on_run_start(self):
        logger.info("fetching initial network params from learner...")
        self.update_params()
        self.env_state, _ = self.env.reset()

    def on_training_step_end(self):
        per_beta_increase = (1 - self.config.per_beta) / self.training_steps
        self.config.per_beta = min(1.0, self.config.per_beta + per_beta_increase)

    def calculate_loss(self, batch: TransitionBuffer):
        print("calculating losses...")
        t = time.time()
        discount_factor = self.config.discount_factor**self.config.n_step
        inputs = self.prepare_states(batch.observations)
        initial_distributions = self.model(inputs)
        target_distributions = self.compute_target_distributions_np(
            batch, discount_factor
        )
        distributions_to_train = tf.gather_nd(
            initial_distributions,
            list(zip(range(initial_distributions.shape[0]), batch.actions)),
        )
        elementwise_loss = self.config.loss_function.call(
            y_pred=distributions_to_train,
            y_true=tf.convert_to_tensor(target_distributions),
        )

        prioritized_loss = elementwise_loss + self.config.per_epsilon
        # CLIPPING PRIORITIZED LOSS FOR ROUNDING ERRORS OR NEGATIVE LOSSES (IDK HOW WE ARE GETTING NEGATIVE LSOSES)
        prioritized_loss = np.clip(
            prioritized_loss, 0.01, tf.reduce_max(prioritized_loss)
        )

        delta_t = time.time() - t
        logger.info(
            f"calculate_losses took: {delta_t} s. Elementwise loss: {elementwise_loss} Losses: {prioritized_loss}"
        )
        return prioritized_loss
