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

from rainbow.rainbow_dqn import RainbowDQN

default_actor_config = {
    "poll_params_interval": 400,  # number of steps between when an actor copies the latest network params from the learner
    "buffer_size": 100,  # size of local replay buffer size
    "batch_size": 50,  # number of experiences to push to remote replay buffer in one batch
}

default_rainbow_config = {
    "activation": "relu",
    "kernel_initializer": "he_uniform",
    "optimizer_function": tf.keras.optimizers.legacy.Adam,  # NO SGD OR RMSPROP FOR NOW SINCE IT IS FOR RAINBOW DQN
    "learning_rate": 0.001,  #
    "adam_epsilon": 0.00003125,
    # NORMALIZATION?
    "soft_update": False,  # seems to always be false, we can try it with tru
    "ema_beta": 0.95,
    "transfer_frequency": 100,
    "replay_period": 1,
    "replay_batch_size": 128,
    "memory_size": 10000,  #############
    "min_memory_size": 500,
    "n_step": 3,
    "discount_factor": 0.99,
    "atom_size": 51,  #
    "conv_layers": [[32, 8, (4, 4)], [64, 4, (2, 2)], [64, 3, (1, 1)]],
    "conv_layers_noisy": False,
    "width": 512,
    "dense_layers": 2,
    "dense_layers_noisy": True,  # i think this is always true for rainbow
    # REWARD CLIPPING
    "noisy_sigma": 0.5,  #
    "loss_function": tf.keras.losses.KLDivergence(),
    "dueling": True,
    "advantage_hidden_layers": 1,  #
    "value_hidden_layers": 1,  #
    "num_training_steps": 50000,  # 25000,
    "per_epsilon": 0.001,
    "per_alpha": 0.5,
    "per_beta": 0.5,
    # 'per_beta_increase': hp.uniform('per_beta_increase', 0, 0.015),
    "v_min": -500.0,  # MIN GAME SCORE
    "v_max": 500.0,  # MAX GAME SCORE
    # 'search_max_depth': 5,
    # 'search_max_time': 10,
    "num_units_per_dense_layer": 512,
}


class ActorBase(RainbowDQN):
    def __init__(
        self,
        id,
        env,
        config=default_rainbow_config,
        actor_config=default_actor_config,
    ):
        super().__init__(model_name=f"actor_{id}", env=env, config=config)
        self.id = id
        self.poll_params_interval = actor_config["poll_params_interval"]
        self.buffer_size = actor_config["buffer_size"]

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
        self.fetch_latest_params()
        self.fill_memory()

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

            if len(self.memory) >= self.replay_batch_size:
                # sample n_step memory
                e = self.memory_n.sample()

                self._push_experiences_to_remote_replay_buffer(e)

            if truncated:
                num_trials_truncated += 1

            self.per_beta = min(1.0, self.per_beta + self.per_beta_increase)

            if done:
                state, _ = self.env.reset()
                state = state
                stat_score.append(score)
                if score >= self.env.spec.reward_threshold:
                    print("your dqn agent has achieved the env's reward threshold.")
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
        actor_config=default_actor_config,
        single_machine_learner: learner.SingleMachineLearner = None,  # TODO: change this to single machine learner
    ):
        super().__init__(id, env, config, actor_config)
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
        actor_config=default_actor_config,
    ):
        super().__init__(id, env, config, actor_config)

    def fetch_latest_params(self):
        pass

    def push_experiences_to_remote_replay_buffer(self, experiences):
        pass
