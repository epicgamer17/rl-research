import time
import tensorflow as tf
import numpy as np
from typing import NamedTuple
from uuid import uuid4
import zmq
from gymnasium import Env
from agent_configs import ApeXActorConfig
import entities.replayMemory_capnp as replayMemory_capnp
import message_codes

import matplotlib

matplotlib.use("Agg")
import logging

logger = logging.getLogger(__name__)

import sys

sys.path.append("..")
from rainbow.rainbow_agent import RainbowAgent
from base_agent.distributed_agents import ActorAgent, PollingActor
from storage.compress_utils import compress, decompress
from storage.storage import Storage, StorageConfig

# from refactored_replay_buffers.prioritized_nstep import ReplayBuffer
from replay_buffers.n_step_replay_buffer import ReplayBuffer


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


class ApeXActorBase(ActorAgent, PollingActor):
    """
    Apex Actor base class
    """

    def __init__(
        self,
        env: Env,
        config: ApeXActorConfig,
        name,
    ):
        super().__init__(env, config, name)
        self.config = config
        self.score = 0
        self.stats = dict(score=list())
        self.rb = ReplayBuffer(
            observation_dimensions=env.observation_space.shape,
            batch_size=self.config.replay_buffer_size,
            max_size=self.config.replay_buffer_size,
            n_step=self.config.n_step,
            gamma=0.99,  # self.config.gamma,
        )

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

        self.rb.store(self.env_state, action, reward, next_state, done)

        if done:
            self.env_state, _ = self.env.reset()
            self.stats["score"].append(self.score)
            self.score = 0
        else:
            self.env_state = next_state

    def should_send_experience_batch(self, training_step: int):
        return self.rb.size == self.rb.max_size

    def should_update_params(self, training_step: int):
        return (
            training_step % self.config.poll_params_interval == 0 and training_step != 0
        )


class ApeXActor(ApeXActorBase, RainbowAgent):
    def __init__(self, env: Env, config: ApeXActorConfig, name, spectator=False):
        super().__init__(env, config, name)
        self.config = config

        self.socket_ctx = zmq.Context()

        storage_config = StorageConfig(
            hostname=self.config.storage_hostname,
            port=self.config.storage_port,
            username=self.config.storage_username,
            password=self.config.storage_password,
        )

        self.storage = Storage(storage_config)

        replay_address = self.config.replay_addr
        replay_port = self.config.replay_port
        replay_url = f"tcp://{replay_address}:{replay_port}"

        self.replay_socket = self.socket_ctx.socket(zmq.PUSH)

        self.replay_socket.connect(replay_url)
        logger.info(f"connected to replay buffer at {replay_url}")

        self.spectator = spectator
        if spectator:
            self.model_name = "spectator"
            self.t_i = None
            self.stats = {"score": []}

    def send_experience_batch(self):
        if self.spectator:
            return
        ids = np.zeros(self.config.replay_buffer_size, dtype=np.object_)

        for i in range(self.config.replay_buffer_size):
            ids[i] = uuid4().hex

        prioritized_losses = self.calculate_loss(
            TransitionBuffer(
                observations=self.rb.observation_buffer,
                actions=self.rb.action_buffer,
                rewards=self.rb.reward_buffer,
                next_observations=self.rb.next_observation_buffer,
                dones=self.rb.done_buffer,
            )
        )

        batch = Batch(
            observations=self.rb.observation_buffer,
            actions=self.rb.action_buffer,
            rewards=self.rb.reward_buffer,
            next_observations=self.rb.next_observation_buffer,
            dones=self.rb.done_buffer,
            ids=ids,
            priorities=prioritized_losses,
        )
        # print(batch.actions)

        # print(batch.observations)

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

        self.rb.clear()

    def update_params(self):
        ti = time.time()
        logger.info("fetching weights from storage...")
        res = self.storage.get_weights()

        if res == None:
            logger.info("no weights recieved from learner")
            return

        decompressed = decompress(res)
        try:
            self.model.set_weights(decompressed)
        except Exception as e:
            print(e)
            logger.info("not enough weights received from learner")
        logger.info(f"fetching weights took {time.time() - ti} s")

    def on_run_start(self):
        logger.info("fetching initial network params from learner...")
        state, info = self.env.reset()
        self.select_action(state)
        self.update_params()
        self.env_state, info = self.env.reset()

        if self.spectator:
            self.t_i = time.time()

    def on_training_step_end(self, training_step):
        per_beta_increase = (1 - self.config.per_beta) / self.training_steps
        self.config.per_beta = min(1.0, self.config.per_beta + per_beta_increase)

        if self.spectator:
            targets = {
                "score": self.env.spec.reward_threshold,
            }
            self.plot_graph(
                self.stats,
                targets,
                training_step,
                training_step,
                time_taken=time.time() - self.t_i,
            )

    def calculate_loss(self, batch: TransitionBuffer):
        t = time.time()
        discount_factor = self.config.discount_factor**self.config.n_step
        inputs, actions = self.prepare_states(batch.observations), batch.actions

        initial_distributions = self.model(inputs)
        distributions_to_train = tf.gather_nd(
            initial_distributions,
            list(zip(range(initial_distributions.shape[0]), actions)),
        )

        # print(distributions_to_train)
        # print(tf.convert_to_tensor(target_distributions))
        # print()

        target_distributions = self.compute_target_distributions_np(
            batch, discount_factor
        )
        elementwise_loss = self.config.loss_function.call(
            y_pred=distributions_to_train,
            y_true=tf.convert_to_tensor(target_distributions),
        )
        assert np.all(elementwise_loss) >= 0, "Elementwise Loss: {}".format(
            elementwise_loss
        )

        prioritized_loss = elementwise_loss + self.config.per_epsilon

        delta_t = time.time() - t
        logger.info(f"calculate_losses took: {delta_t} s.")
        return prioritized_loss.numpy()
