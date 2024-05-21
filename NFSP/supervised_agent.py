import tensorflow as tf
from NFSP.supervised_network import SupervisedNetwork
from base_agent.agent import BaseAgent
from imitation_learning.imitation_agent import BaseImitationAgent
from replay_buffers.nfsp_reservoir_buffer import NFSPReservoirBuffer
import tensorflow_probability as tfp


class AverageStrategyAgent(BaseImitationAgent):
    def __init__(self, env, config, name):
        replay_buffer = NFSPReservoirBuffer(
            observation_dimensions=self.observation_dimensions,
            max_size=self.config.replay_buffer_size,
            batch_size=self.config.minibatch_size,
        )
        model = SupervisedNetwork(config, self.num_actions, self.observation_dimensions)
        super().__init__(
            env,
            config,
            name,
            replay_buffer,
            model,
            tf.keras.losses.CategoricalCrossentropy(),
        )

    def select_action(self, state, legal_moves=None):
        policy = self.predict_single(state, legal_moves)
        distribution = tfp.distributions.Categorical(probs=policy)
        selected_action = distribution.sample().numpy()
        return selected_action

    def predict_single(self, state, legal_moves=None):
        state_input = self.prepare_states(state)
        policy = self.model(inputs=state_input).numpy()[0]
        policy = self.action_mask(policy, legal_moves)
        policy /= tf.reduce_sum(policy)
        return policy
