import tensorflow as tf
from NFSP.supervised_network import SupervisedNetwork
from base_agent.agent import BaseAgent
from replay_buffers.reservoir_buffer import ReservoirBuffer
import tensorflow_probability as tfp


class AverageStrategyAgent(BaseAgent):
    def __init__(self, env, config, name):
        super().__init__(env, config, name)
        self.replay_buffer = ReservoirBuffer(10000)
        self.model = SupervisedNetwork(
            config, self.num_actions, self.observation_dimensions
        )

    def experience_replay(self):
        for training_iteration in range(self.config.training_iterations):
            with tf.GradientTape() as tape:
                sample = self.replay_buffer.sample()
                observations = sample["observations"]
                actions = sample["actions"]

                state_input = self.prepare_states(observations)
                probabilities = self.model(inputs=state_input)
                best_actions_mask = tf.one_hot(actions, self.num_actions)
                action_probabilities = tf.reduce_sum(
                    best_actions_mask * probabilities, axis=1
                )
                loss = -tf.reduce_mean(tf.math.log(action_probabilities))

                gradients = tape.gradient(loss, self.model.trainable_variables)
                self.config.optimizer.apply_gradients(
                    grads_and_vars=zip(gradients, self.model.trainable_variables)
                )

                # RESET NOISE IF IM DOING THAT
        return loss

    def select_action(self, state, legal_moves=None):
        probabilities = self.model.predict_single(state)
        distribution = tfp.distributions.Categorical(probs=probabilities)
        selected_action = distribution.sample().numpy()[0]

        return selected_action

    def predict_single(self, state):
        state_input = self.prepare_states(state)
        probabilities = self.model(inputs=state_input).numpy()
        return probabilities
