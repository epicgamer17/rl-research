import io
import torch
import time
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

sys.path.append("../../")
from rainbow.rainbow_agent import RainbowAgent
from base_agent.distributed_agents import ActorAgent, PollingActor
from storage.compress_utils import compress, decompress
from storage.storage import Storage, StorageConfig

from replay_buffers.n_step_replay_buffer import NStepReplayBuffer


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
        if self.spectator:
            self.targets = {
                "score": self.env.spec.reward_threshold,
            }
        self.stats = {
            "score": [],
        }
        self.rb = NStepReplayBuffer(
            observation_dimensions=env.observation_space.shape,
            batch_size=self.config.replay_buffer_size,
            max_size=self.config.replay_buffer_size,
            n_step=self.config.n_step,
            gamma=0.99,  # self.config.gamma,
        )
        self.bootstrapped_q_buffer = np.zeros(self.config.replay_buffer_size)

    def on_run_start(self):
        super().on_run_start()
        self.env_state, _ = self.env.reset()

    def predict_target_q(self, state, action) -> float:
        pass

    def collect_experience(self):
        with torch.no_grad():
            state_input = self.preprocess(self.env_state)
            action = self.select_actions(state_input)

            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = truncated or terminated
            self.score += reward

            p = self.rb.pointer
            t = self.rb.store(self.env_state, action, reward, next_state, done)
            if t != None:
                predicted_q = self.predict(next_state)
                action = self.select_actions(predicted_q).item()
                self.bootstrapped_q_buffer[p] = (
                    self.config.n_step ** self.predict_target_q(next_state, action)
                )

            if done:
                self.env_state, _ = self.env.reset()
                score_dict = {
                    "score": self.score,
                    # "target_model_updated": target_model_updated[0],
                }
                self.stats["score"].append(score_dict)
                self.score = 0
            else:
                self.env_state = next_state

    def should_send_experience_batch(self):
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

        self.bootstrapped_q_buffer = np.zeros(self.config.replay_buffer_size)
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

    def predict_target_q(self, state, action) -> float:
        target_q_values = self.predict_target(state) * self.support
        q = target_q_values.sum(2, keepdim=False)[1, action].item()
        return q

    def calculate_losses(self):
        with torch.no_grad():
            # (B)
            bootstrapped_qs = torch.from_numpy(self.bootstrapped_q_buffer).to(self.device)
            # (B)
            rewards = torch.from_numpy(self.replay_buffer.reward_buffer).to(self.device)

            # (B)
            Gt = rewards + bootstrapped_qs  # already discounted

            # (B)
            actions = torch.from_numpy(self.replay_buffer.action_buffer).to(self.device)
            # (B, output_size, atoms)
            predicted_distributions = self.predict(self.rb.observation_buffer)

            # (B, output_size, atoms) -> (B, atoms) -> (B)
            predicted = (predicted_distributions * self.support)[
                range(self.config.minibatch_size), actions
            ].sum(2, keepdim=False)

            # (B)
            predicted = predicted

            # (B)
            batched_loss = 1 / 2 * (Gt - predicted_distributions).square()
            return batched_loss.detach().cpu().numpy()

    def send_experience_batch(self):
        if self.spectator:
            return
        ids = np.zeros(self.config.replay_buffer_size, dtype=np.object_)

        for i in range(self.config.replay_buffer_size):
            ids[i] = uuid4().hex

        prioritized_losses = self.calculate_losses()

        batch = Batch(
            observations=self.rb.observation_buffer,
            actions=self.rb.action_buffer,
            rewards=self.rb.reward_buffer,
            next_observations=self.rb.next_observation_buffer,
            dones=self.rb.done_buffer,
            ids=ids,
            priorities=prioritized_losses,
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

        self.rb.clear()

    def update_params(self):
        ti = time.time()
        logger.info("fetching weights from storage...")
        online_bytes, target_bytes = self.storage.get_weights()

        if online_bytes == None or target_bytes == None:
            logger.info("no weights recieved from learner")
            return False

        try:
            decompressed_online = decompress(online_bytes)
            with io.BytesIO(decompressed_online) as buf:
                state_dict = torch.load(buf, self.device)
                self.model.load_state_dict(state_dict)

            decompressed_target = decompress(target_bytes)
            with io.BytesIO(decompressed_target) as buf:
                state_dict = torch.load(buf, self.device)
                self.target_model.load_state_dict(state_dict)
                return True

        except Exception as e:
            logger.warning("error loading weights from storage:", e)
            return False

        finally:
            logger.info(f"fetching weights took {time.time() - ti} s")

    def on_run_start(self):
        logger.info("fetching initial network params from learner...")
        state, info = self.env.reset()
        self.select_actions(state)

        # wait for initial network parameters
        has_weights = self.update_params()
        while not has_weights:
            time.sleep(2)
            has_weights = self.update_params()

        self.env_state, info = self.env.reset()

        if self.spectator:
            self.t_i = time.time()

    def on_training_step_end(self, training_step):
        # self.replay_buffer.beta = update_per_beta(
        #     training_step, self.replay_buffer.beta, self.config.training_steps
        # ) # i dont think actor should update per beta as it should happen after learning steps

        if self.spectator:
            self.targets = {
                "score": self.env.spec.reward_threshold,
            }
            self.plot_graph(
                training_step,
                training_step,
                time_taken=time.time() - self.t_i,
            )
