from IPython import embed
import torch
import torch.nn.functional as F
from dreamer.basic_rssm_network import (
    ContinuePredictor,
    Decoder,
    DynamicsPredictor,
    Encoder,
    RewardPredictor,
)


class RSSM(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, embedding_dim):
        super(RSSM, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        # Define the encoder, decoder, dynamics predictor, reward predictor, and continue predictor
        self.encoder = Encoder(state_dim, action_dim, hidden_dim, embedding_dim)
        self.decoder = Decoder(state_dim, action_dim, hidden_dim, embedding_dim)
        self.dynamics_predictor = DynamicsPredictor(state_dim, action_dim, hidden_dim)
        self.reward_predictor = RewardPredictor(state_dim, action_dim, hidden_dim)

        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.dynamics_predictor.parameters())
            + list(self.reward_predictor.parameters()),
            lr=1e-3,
        )

    def generate_rollout(
        self, actions, hiddens=None, states=None, observations=None, dones=None
    ):
        if hiddens is None:
            hiddens = torch.zeros((actions.size(0), self.hidden_dim)).to(actions.device)
        if states is None:
            states = torch.zeros((actions.size(0), self.state_dim)).to(actions.device)

        return self.dynamics_predictor(hiddens, states, actions, observations, dones)

    def train(
        self,
        observations,
        actions,
        rewards,
        dones,
    ):
        actions = F.one_hot(actions, num_classes=self.action_dim)
        actions = actions.float()

        encoded_observations = self.encoder(observations)

        (
            hiddens,
            prior_states,
            posterior_states,
            prior_means,
            prior_log_vars,
            posterior_means,
            posterior_log_vars,
        ) = self.generate_rollout(
            actions, observations=encoded_observations, dones=dones
        )

        decoded_observations = self.decoder(hiddens, posterior_states)

        reward_output = self.reward_predictor(hiddens, posterior_states)

        means, log_vars = torch.chunk(reward_output, 2, dim=-1)
        log_vars = F.softplus(log_vars)
        reward_distribution = torch.distributions.Normal(means, torch.exp(log_vars))
        predicted_rewards = reward_distribution.rsample()

        reconstruction_loss = self._reconstruction_loss(
            decoded_observations, observations
        )
        kl_loss = self._kl_loss(
            prior_means,
            F.softplus(prior_log_vars),
            posterior_means,
            F.softplus(posterior_log_vars),
        )
        reward_loss = self._reward_loss(predicted_rewards, rewards)

        loss = reconstruction_loss + kl_loss + reward_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
