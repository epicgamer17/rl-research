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

logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")


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

    def _fetch_latest_params(self):
        t = time.time()
        self.fetch_latest_params()
        delta_t = time.time() - t
        logging.info(f"fetch_latest_params took: {delta_t} s")

    # to be implemented by subclasses
    def fetch_latest_params(self):
        pass

    def _push_experiences_to_remote_replay_buffer(self):
        thread = threading.Thread(target=self.push_experiences_to_remote_replay_buffer)
        thread.run()

    # to be implemented by subclasses
    def push_experiences_to_remote_replay_buffer(self):
        pass

    def run(self):
        self.is_test = False
        self.fetch_latest_params()
        print("filling replay buffer...")
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

            self.replay_buffer.store(state, action, reward, next_state, done)

            if training_step % self.replay_buffer_size == 0 and training_step > 0:
                self._push_experiences_to_remote_replay_buffer()

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

    def push_experiences_to_remote_replay_buffer(self):
        t = time.time()
        logging.info(
            f" {self.model_name} pushing {self.replay_batch_size} experiences to remote replay buffer"
        )
        indices = list(range(self.replay_batch_size))
        elementwise_loss = 0
        samples = self.replay_buffer.sample_from_indices(indices)
        actions = samples["actions"]
        observations = samples["observations"]
        inputs = self.prepare_states(observations)
        discount_factor = self.discount_factor
        target_ditributions = self.compute_target_distributions(
            samples, discount_factor
        )
        initial_distributions = self.model(inputs)
        distributions_to_train = tf.gather_nd(
            initial_distributions,
            list(zip(range(initial_distributions.shape[0]), actions)),
        )
        elementwise_loss = self.model.loss.call(
            y_pred=distributions_to_train,
            y_true=tf.convert_to_tensor(target_ditributions),
        )
        assert np.all(elementwise_loss) >= 0, "Elementwise Loss: {}".format(
            elementwise_loss
        )
        # if self.use_n_step:
        discount_factor = self.discount_factor**self.n_step
        n_step_samples = self.n_step_replay_buffer.sample_from_indices(indices)
        actions = n_step_samples["actions"]
        n_step_observations = n_step_samples["observations"]
        # observations = n_step_observations
        inputs = self.prepare_states(n_step_observations)
        target_ditributions = self.compute_target_distributions(
            n_step_samples, discount_factor
        )
        initial_distributions = self.model(inputs)
        distributions_to_train = tf.gather_nd(
            initial_distributions,
            list(zip(range(initial_distributions.shape[0]), actions)),
        )
        elementwise_loss_n_step = self.model.loss.call(
            y_pred=distributions_to_train,
            y_true=tf.convert_to_tensor(target_ditributions),
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

        self.replay_buffer.update_priorities(indices, prioritized_loss)
        print(n_step_samples["observations"].shape)

        for i in range(self.replay_batch_size):
            self.learner.replay_buffer.store_with_priority(
                n_step_samples["observations"][i],
                n_step_samples["actions"][i],
                n_step_samples["rewards"][i],
                n_step_samples["next_observations"][i],
                n_step_samples["dones"][i],
                prioritized_loss[i],
            )

        delta_t = time.time() - t
        logging.info(f"push_experiences_to_remote_replay_buffer took: {delta_t} s")


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
