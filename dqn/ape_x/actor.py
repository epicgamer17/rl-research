import sys
import time
import torch
import numpy as np
from typing import Any
from uuid import uuid4
from gymnasium import Env
from typing import NamedTuple
import torch.distributed.rpc as rpc
from agent_configs import ApeXActorConfig
from utils import plot_graphs, epsilon_greedy_policy

import matplotlib
import logging

matplotlib.use("Agg")

logger = logging.Logger(f"actor", logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

import sys

sys.path.append("../../")
from dqn.rainbow.rainbow_agent import RainbowAgent
from dqn.rainbow.rainbow_network import RainbowNetwork
from base_agent.distributed_agents import ActorAgent, DistreteTransition
from replay_buffers.prioritized_n_step_replay_buffer import PrioritizedNStepReplayBuffer
from replay_buffers.n_step_replay_buffer import NStepReplayBuffer


class Batch(NamedTuple):
    observations: np.ndarray
    infos: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    next_infos: np.ndarray
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
            observation_dtype=env.observation_space.dtype,
            batch_size=self.config.replay_buffer_size,
            max_size=self.config.replay_buffer_size,
            n_step=self.config.n_step,
            gamma=0.99,  # self.config.gamma,
        )
        self.precalculated_q = np.zeros(self.config.replay_buffer_size)

    def predict_target_q(self, state, action) -> float:
        pass

    def collect_experience(self, state, info) -> tuple[DistreteTransition, Any]:
        t, next_info = super().collect_experience(state, info)

        p = self.rb.pointer
        n_step_t = self.rb.store(
            t.state,
            t.info,
            t.action,
            t.reward,
            t.next_state,
            t.next_info,
            t.done,
            None,
        )

        if n_step_t != None:
            predicted_q = self.predict(t.next_state)
            action = self.select_actions(predicted_q, next_info).item()

            self.precalculated_q[p] = (
                self.config.discount_factor**self.config.n_step
            ) * self.predict_target_q(t.next_state, action)

        return t, next_info

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
        input = self.preprocess(state)
        target_distributions = self.target_model(input)
        q_value = (target_distributions * self.support)[0, action].sum().item()
        return q_value

    # override
    def collect_experience(self, state, info) -> tuple[DistreteTransition, Any]:
        legal_moves = None  # get_legal_moves(info)
        # (1, output_length, num_atoms)
        values = self.predict(state)
        action = epsilon_greedy_policy(
            values,
            self.config.eg_epsilon,
            wrapper=lambda values: self.select_actions(values, info).item(),
            range=self.num_actions,
        )
        next_state, reward, terminated, truncated, next_info = self.env.step(action)
        done = truncated or terminated

        t = DistreteTransition(
            state, info, action, reward, next_state, next_info, done, legal_moves
        )
        p = self.rb.pointer

        n_step_t = self.rb.store(
            t.state, t.info, t.action, t.reward, t.next_state, t.next_info, t.done, None
        )

        if n_step_t != None and not self.spectator:
            action = self.select_actions(self.predict(t.next_state), next_info).item()

            self.precalculated_q[p] = (
                self.config.discount_factor**self.config.n_step
            ) * self.predict_target_q(t.next_state, action)

        if self.spectator:
            self.score += reward
            if done:
                score_dict = {"score": self.score}
                self.stats["score"].append(score_dict)
                self.score = 0

        return t, next_info

    def calculate_losses(self):
        with torch.no_grad():
            # (B)
            bootstrapped_qs = torch.from_numpy(self.precalculated_q).to(self.device)
            # print("BS",bootstrapped_qs)
            # (B)
            rewards = torch.from_numpy(self.replay_buffer.reward_buffer).to(self.device)
            # print("rewards",rewards)

            # (B)
            Gt = rewards + bootstrapped_qs  # already discounted
            # print("Gt",Gt)

            # (B)
            actions = (
                torch.from_numpy(self.replay_buffer.action_buffer)
                .to(self.device)
                .long()
            )
            # (B, output_size, atoms)
            predicted_distributions = self.predict(self.rb.observation_buffer)
            # print("pred-dist",predicted_distributions)

            # (B, output_size, atoms) -> (B, atoms) -> (B)
            predicted_q = (predicted_distributions * self.support)[
                range(self.config.minibatch_size), actions
            ].sum(1)
            # print("pred-q",predicted_q)

            # (B)
            batched_loss = 1 / 2 * (Gt - predicted_q).square()
            # print("BL",batched_loss)
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
            infos=self.rb.info_buffer,
            actions=self.rb.action_buffer,
            rewards=self.rb.reward_buffer,
            next_observations=self.rb.next_observation_buffer,
            next_infos=self.rb.next_info_buffer,
            dones=self.rb.done_buffer,
            ids=ids,
            priorities=prioritized_losses,
        )

        try:
            self.remote_replay.rpc_sync().store_batch(batch)
            print("stored batch")
        except Exception as e:
            logger.info(f"failed to store batch: {e}")

        self.rb.clear()

    def update_params(self):
        ti = time.time()
        logger.info("fetching weights from storage...")
        try:
            remote_model_params = (
                self.remote_online_params.remote().state_dict().to_here()
            )
            remote_target_params = (
                self.remote_target_params.remote().state_dict().to_here()
            )

            self.model.load_state_dict(remote_model_params)
            self.target_model.load_state_dict(remote_target_params)
        except Exception as e:
            logger.info(f"failed to fetch weights: {e}")
            return False
        logger.info(f"fetching weights took {time.time() - ti} s")
        return True

    def setup(self):
        if self.spectator:
            self.t_i = time.time()
        # wait for initial network parameters
        logger.info("fetching initial network params from learner...")
        has_weights = self.update_params()
        while not has_weights:
            print("no weights, trying again")
            has_weights = self.update_params()
            time.sleep(2)

        self.env_state, info = self.env.reset()

    def on_training_step_end(self, training_step):
        if not self.spectator:
            if self.should_send_experience_batch():
                self.send_experience_batch()

        else:
            logger.debug("spectator plotting graphs")
            plot_graphs(
                self.stats,
                self.targets,
                training_step,
                training_step,
                time.time() - self.t_i,
                "spectator",
            )

        if self.should_update_params(training_step):
            self.update_params()

    def cleanup(self):
        print("cleanup")
        # safely release all remote references
        try:
            del self.remote_online_params
        except:
            pass
        try:
            del self.remote_online_params
        except:
            pass
        try:
            del self.remote_target_params
        except:
            pass
        print("all realeased")
