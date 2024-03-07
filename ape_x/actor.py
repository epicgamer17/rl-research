import sys
import tensorflow as tf
import numpy as np
import threading
import time
import logging
import keras
import learner
import gymnasium as gym

sys.path.append("../")

from rainbow.rainbow_agent import RainbowAgent


class ActorBase(RainbowAgent):
    def __init__(
        self,
        id,
        env,
        config,
    ):
        super().__init__(model_name=f"actor_{id}", env=env, config=config)
        self.id = id
        self.poll_params_interval = config["poll_params_interval"]
        self.buffer_size = config["buffer_size"]

    def _fetch_latest_params(self):
        t = time.time()
        self.fetch_latest_params()
        delta_t = time.time() - t
        logging.info(f"fetch_latest_params took: {delta_t} s")

    # to be implemented by subclasses
    def fetch_latest_params(self):
        pass

    def _push_experiences_to_remote_replay_buffer(self, experiences):
        t = time.time()
        self.push_experiences_to_remote_replay_buffer(experiences)
        delta_t = time.time() - t
        logging.info(f"push_experiences_to_remote_replay_buffer took: {delta_t} s")

    # to be implemented by subclasses
    def push_experiences_to_remote_replay_buffer(self, experiences):
        pass

    def run(self):
        self.is_test = False
        self.fetch_latest_params()
        self.fill_replay_buffer()

        state, _ = self.env.reset()
        score = 0
        stat_score = []
        num_trials_truncated = 0

        training_step = 0
        while training_step < self.num_training_steps:
            logging.info(
                f"{self.model_name} training step: {training_step}/{self.num_training_steps}"
            )

            state_input = self.prepare_states(state)
            action = self.select_action(state_input)
            next_state, reward, terminated, truncated = self.step(action)
            done = terminated or truncated
            state = next_state
            score += reward

            if len(self.replay_buffer) >= self.replay_batch_size:
                # sample n_step replay_buffer
                e = self.n_step_replay_buffer.sample()

                self._push_experiences_to_remote_replay_buffer(e)

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
        logging.info(f" {self.model_name} fetching latest params from learner")
        return self.learner.get_weights()

    def push_experiences_to_remote_replay_buffer(self, experiences):
        logging.info(f" {self.model_name} pushing experiences to remote replay buffer")
        # TODO: push experiences to single machine learner


# TODO make it actually distributed
class RemoteActor(ActorBase):
    def __init__(
        self,
        id,
        env,
        config,
    ):
        super().__init__(id, env, config)

    def fetch_latest_params(self):
        pass

    def push_experiences_to_remote_replay_buffer(self, experiences):
        pass
