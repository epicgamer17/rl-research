import tensorflow as tf
from NFSP.supervised_network import SupervisedNetwork
from base_agent.agent import BaseAgent
from replay_buffers.reservoir_buffer import ReservoirBuffer
import tensorflow_probability as tfp


class AverageStrategyAgent(BaseAgent):
    def __init__(self, env, config, name):
        super().__init__(env, config, name)
        self.replay_buffer = ReservoirBuffer(
            observation_dimensions=self.observation_dimensions,
            max_size=self.config.replay_buffer_size,
            batch_size=self.config.minibatch_size,
        )
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
                policy = self.model(inputs=state_input)
                # LEGAL MOVE MASKING?
                best_actions_mask = tf.one_hot(actions, self.num_actions)
                action_policy = tf.reduce_sum(best_actions_mask * policy, axis=1)
                loss = -tf.reduce_mean(tf.math.log(action_policy))

                gradients = tape.gradient(loss, self.model.trainable_variables)
                self.config.optimizer.apply_gradients(
                    grads_and_vars=zip(gradients, self.model.trainable_variables)
                )

                # RESET NOISE IF IM DOING THAT
        return loss

    def select_action(self, state, legal_moves=None):
        policy = self.predict_single(state, legal_moves)
        print("policy", policy)
        distribution = tfp.distributions.Categorical(probs=policy)
        selected_action = distribution.sample().numpy()
        print("selected_action", selected_action)
        return selected_action

    def predict_single(self, state, legal_moves=None):
        state_input = self.prepare_states(state)
        policy = self.model(inputs=state_input).numpy()[0]
        policy = self.action_mask(policy, legal_moves)
        policy /= tf.reduce_sum(policy)
        return policy
