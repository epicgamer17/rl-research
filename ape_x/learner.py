import sys
import logging
import time
import tensorflow as tf
import numpy as np

sys.path.append("../")
from rainbow.rainbow_agent import RainbowAgent


class LearnerBase(RainbowAgent):
    def __init__(self, env, config):
        super().__init__(model_name="learner", env=env, config=config)
        self.graph_interval = 200
        self.remove_old_experiences_interval = config["remove_old_experiences_interval"]
        self.running = False

    def sample_experiences_from_remote_replay_buffer(self):
        pass

    def sample_n_step_experiences_from_remote_replay_buffer(self, indices):
        pass

    def update_remote_replay_buffer_priorities(self, indices, priorities):
        pass

    def remove_old_experiences_from_remote_replay_buffer(self):
        pass

    def _experience_replay(self):
        with tf.GradientTape() as tape:
            elementwise_loss = 0
            samples = self.sample_experiences_from_remote_replay_buffer()
            actions = samples["actions"]
            observations = samples["observations"]
            inputs = self.prepare_states(observations)
            weights = samples["weights"].reshape(-1, 1)
            indices = samples["indices"]
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
            n_step_samples = self.sample_n_step_experiences_from_remote_replay_buffer(
                indices
            )
            actions = n_step_samples["actions"]
            n_step_observations = n_step_samples["observations"]
            observations = n_step_observations
            inputs = self.prepare_states(observations)
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

            loss = tf.reduce_mean(elementwise_loss * weights)

        # TRAINING WITH GRADIENT TAPE
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer(
            learning_rate=self.learning_rate,
            epsilon=self.adam_epsilon,
            clipnorm=self.clipnorm,
        ).apply_gradients(grads_and_vars=zip(gradients, self.model.trainable_variables))
        # TRAINING WITH tf.train_on_batch
        # loss = self.model.train_on_batch(samples["observations"], target_ditributions, sample_weight=weights)

        prioritized_loss = elementwise_loss + self.per_epsilon
        # CLIPPING PRIORITIZED LOSS FOR ROUNDING ERRORS OR NEGATIVE LOSSES (IDK HOW WE ARE GETTING NEGATIVE LSOSES)
        prioritized_loss = np.clip(
            prioritized_loss, 0.01, tf.reduce_max(prioritized_loss)
        )
        self.update_remote_replay_buffer_priorities(indices, prioritized_loss)
        self.model.reset_noise()
        self.target_model.reset_noise()
        loss = loss.numpy()
        return loss

    def run(self, graph_interval=200):
        logging.info("learner running")
        self.is_test = False
        stat_score = (
            []
        )  # make these num trials divided by graph interval so i dont need to append (to make it faster?)
        stat_test_score = []
        stat_loss = []
        # self.fill_replay_buffer()
        num_trials_truncated = 0
        state, _ = self.env.reset()
        model_update_count = 0
        score = 0
        training_step = 0
        while training_step < self.num_training_steps:
            logging.info(
                f"learner training step: {training_step}/{self.num_training_steps}"
            )
            self.per_beta = min(1.0, self.per_beta + self.per_beta_increase)

            if self.replay_buffer.size >= self.replay_batch_size:
                model_update_count += 1
                loss = self._experience_replay()
                training_step += 1
                stat_loss.append(loss)
                self.update_target_model(model_update_count)
            else:
                time.sleep(0.1)

            if training_step % graph_interval == 0 and training_step > 0:
                self.export()
                # stat_test_score.append(self.test())
                self.plot_graph(stat_score, stat_loss, stat_test_score, training_step)

        self.plot_graph(stat_score, stat_loss, stat_test_score, training_step)
        self.export()
        self.env.close()
        return num_trials_truncated / self.num_training_steps


class SingleMachineLearner(LearnerBase):
    def __init__(self, env, config):
        super().__init__(env=env, config=config)

    def sample_experiences_from_remote_replay_buffer(self):
        print("Sampling experiences from remote replay buffer")
        return self.replay_buffer.sample()

    def sample_n_step_experiences_from_remote_replay_buffer(self, indices):
        print("Sampling n-step experiences from remote replay buffer")
        return self.n_step_replay_buffer.sample_from_indices(indices)

    def update_remote_replay_buffer_priorities(self, indices, priorities):
        print("Updating remote replay buffer priorities")
        return self.replay_buffer.update_priorities(indices, priorities)

    def remove_old_experiences_from_remote_replay_buffer(self):
        # done atomatically in the replay buffer (old experiences overwritten)
        pass

    def get_weights(self):
        return self.model.get_weights()
