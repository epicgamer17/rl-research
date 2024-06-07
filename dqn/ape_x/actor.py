from typing import Any
import torch
import time
import numpy as np
from typing import NamedTuple
from uuid import uuid4
import torch.distributed.rpc as rpc
from gymnasium import Env
from agent_configs import ApeXActorConfig
from utils import plot_graphs

import matplotlib
import logging

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

import sys

sys.path.append("../../")
from dqn.rainbow.rainbow_agent import RainbowAgent
from dqn.rainbow.rainbow_network import RainbowNetwork
from base_agent.distributed_agents import ActorAgent, DistreteTransition
from replay_buffers.prioritized_n_step_replay_buffer import PrioritizedNStepReplayBuffer
from replay_buffers.n_step_replay_buffer import NStepReplayBuffer


class Batch(NamedTuple):
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray
    ids: np.ndarray
    priorities: np.ndarray


class ApeXActorBase(ActorAgent):
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
        self.rb = NStepReplayBuffer(
            observation_dimensions=env.observation_space.shape,
            batch_size=self.config.replay_buffer_size,
            max_size=self.config.replay_buffer_size,
            n_step=self.config.n_step,
            gamma=0.99,  # self.config.gamma,
        )
        self.precalculated_q = np.zeros(self.config.replay_buffer_size)

    def predict_target_q(self, state, action) -> float:
        pass

    def collect_experience(self, state, info) -> tuple[DistreteTransition, Any]:
        t, info = super().collect_experience(state, info)

        p = self.rb.pointer
        n_step_t = self.rb.store(
            t.state, t.action, t.reward, t.next_state, t.done, None, t.legal_moves
        )

        if n_step_t != None:
            predicted_q = self.predict(t.next_state)
            action = self.select_actions(predicted_q).item()
            self.precalculated_q[p] = self.config.n_step ** self.predict_target_q(
                t.next_state, action
            )

        return t, info

    def should_send_experience_batch(self):
        return self.rb.size == self.rb.max_size

    def should_update_params(self, training_step: int):
        return (
            training_step % self.config.poll_params_interval == 0 and training_step != 0
        )


class ApeXActor(ApeXActorBase, RainbowAgent):
    def __init__(
        self,
        env: Env,
        config: ApeXActorConfig,
        name,
        remote_replay: rpc.RRef[PrioritizedNStepReplayBuffer],
        remote_online_params: rpc.RRef[RainbowNetwork],
        remote_target_params: rpc.RRef[RainbowNetwork],
        spectator=False,
    ):
        super().__init__(env, config, name)
        self.agent_rref = rpc.RRef(self)
        self.config = config

        self.remote_online_params = remote_online_params
        self.remote_target_params = remote_target_params
        self.remote_replay = remote_replay

        self.spectator = spectator

        if spectator:
            self.score = 0
            self.model_name = "spectator"
            self.t_i = None
            self.stats = {"score": []}
            self.targets = {
                "score": self.env.spec.reward_threshold,
            }

    def predict_target_q(self, state, action) -> float:
        input = self.preprocess(state, device=self.device)
        target_distributions = self.target_model(input)
        q_value = (target_distributions * self.support)[0, action].sum().item()
        return q_value

    def collect_experience(self, state, info) -> tuple[DistreteTransition, Any]:
        t, info = super().collect_experience(state, info)

        if self.spectator:
            self.score += t.reward
            if t.done:
                score_dict = {"score": self.score}
                self.stats["score"].append(score_dict)
                self.score = 0

        return t, info

    def predict_target_q(self, state, action) -> float:
        target_q_values = self.predict_target(state) * self.support
        q = target_q_values.sum(2, keepdim=False)[1, action].item()
        return q

    def calculate_losses(self):
        with torch.no_grad():
            # (B)
            bootstrapped_qs = torch.from_numpy(self.precalculated_q).to(self.device)
            # (B)
            rewards = torch.from_numpy(self.replay_buffer.reward_buffer).to(self.device)

            # (B)
            Gt = rewards + bootstrapped_qs  # already discounted

            # (B)
            actions = torch.from_numpy(self.replay_buffer.action_buffer).to(self.device)
            # (B, output_size, atoms)
            predicted_distributions = self.predict(self.rb.observation_buffer)

            # (B, output_size, atoms) -> (B, atoms) -> (B)
            predicted_q = (predicted_distributions * self.support)[
                range(self.config.minibatch_size), actions
            ].sum(1)

            # (B)
            batched_loss = 1 / 2 * (Gt - predicted_q).square()
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

        self.remote_replay.remote().store_batch(batch).add_done_callback(
            lambda: print("sent batch")
        )

    def update_params(self):
        ti = time.time()
        logger.info("fetching weights from storage...")
        self.model.load_state_dict(self.remote_online_params.rpc_sync().state_dict())
        self.target_model.load_state_dict(
            self.remote_target_params.rpc_sync().state_dict()
        )
        logger.info(f"fetching weights took {time.time() - ti} s")

    def setup(self):
        if self.spectator:
            self.t_i = time.time()
        # wait for initial network parameters
        logger.info("fetching initial network params from learner...")
        has_weights = self.update_params()
        while not has_weights:
            time.sleep(2)
            has_weights = self.update_params()

        self.env_state, info = self.env.reset()

    def on_training_step_end(self, training_step):
        if self.should_send_experience_batch():
            self.send_experience_batch()

        if self.should_update_params(training_step):
            self.update_params()

        if self.spectator:
            self.targets = {
                "score": self.env.spec.reward_threshold,
            }
            plot_graphs(
                self.stats,
                self.targets,
                training_step,
                training_step,
                time.time() - self.t_i,
                "spectator",
            )
