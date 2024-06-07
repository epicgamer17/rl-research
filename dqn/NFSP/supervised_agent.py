from NFSP.supervised_network import SupervisedNetwork
from base_agent.agent import BaseAgent
from imitation_learning.imitation_agent import BaseImitationAgent
from replay_buffers.nfsp_reservoir_buffer import NFSPReservoirBuffer

from utils import normalize_policy, action_mask, CategoricalCrossentropyLoss
import torch


class AverageStrategyAgent(BaseImitationAgent):
    def __init__(self, env, config, name):
        replay_buffer = NFSPReservoirBuffer(
            observation_dimensions=self.observation_dimensions,
            observation_dtype=self.env.observation_space.dtype,
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
            CategoricalCrossentropyLoss(),
        )

    def select_actions(self, state, legal_moves=None):
        distribution = self.predict(state, legal_moves)
        selected_action = distribution.sample().numpy()
        return selected_action

    def predict(self, state, legal_moves=None):
        state_input = self.preprocess(state)
        policy = self.model(inputs=state_input)[0]
        policy = action_mask(policy, legal_moves, self.num_actions, mask_value=0)
        policy = normalize_policy(policy)
        distribution = torch.distributions.Categorical(probs=policy)

        return distribution
