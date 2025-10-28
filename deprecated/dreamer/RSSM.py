import torch
import torch.nn.functional as F
from torch import nn
from dreamer.basic_rssm_network import (
    ContinuePredictor,
    Decoder,
    DynamicsPredictor,
    Encoder,
    RewardPredictor,
)
from torch.distributions.kl import kl_divergence
from torch.distributions.normal import Normal
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.optim.sgd import SGD
from torch.optim.adam import Adam

# from packages.utils.utils.utils import KLDivergenceLoss
from replay_buffers.deprecated.rssm_replay_buffer import RSSMReplayBuffer
from torch.nn.utils import clip_grad_norm_
import gc


class RSSM(nn.Module):
    def __init__(
        self, env, config, device="cude" if torch.cuda.is_available() else "cpu"
    ):
        super(RSSM, self).__init__()
        self.env = env
        self.device = device
        self.config = config
        self.state_dim = self.config.state_dim
        # self.action_dim = self.config.action_dim
        self.action_dim = env.action_space.n
        self.hidden_dim = self.config.hidden_dim
        self.embedding_dim = self.config.embedding_dim

        # Define the encoder, decoder, dynamics predictor, reward predictor, and continue predictor
        self.encoder = Encoder(
            input_shape=env.observation_space.shape,
            is_image=self.config.is_image,
            embedding_dim=self.embedding_dim,
            norm=self.config.norm,
            activation=self.config.activation,
        )
        # output_shape is reverse of C x H x W from W x H x C
        output_shape = tuple(reversed(env.observation_space.shape))
        self.decoder = Decoder(
            output_shape=output_shape,
            is_image=self.config.is_image,
            hidden_dim=self.hidden_dim,
            state_dim=self.state_dim,
            embedding_dim=self.embedding_dim,
            norm=self.config.norm,
            activation=self.config.activation,
        )
        self.dynamics_predictor = DynamicsPredictor(
            hidden_dim=self.hidden_dim,
            state_dim=self.state_dim,
            embedding_dim=self.embedding_dim,
            action_dim=self.action_dim,
            activation=self.config.activation,
        )
        self.reward_predictor = RewardPredictor(
            state_dim=self.state_dim,
            hidden_dim=self.hidden_dim,
            activation=self.config.activation,
        )

        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.dynamics_predictor.to(self.device)

        if self.config.optimizer == Adam:
            self.optimizer: torch.optim.Optimizer = self.config.optimizer(
                params=list(self.encoder.parameters())
                + list(self.decoder.parameters())
                + list(self.dynamics_predictor.parameters())
                + list(self.reward_predictor.parameters()),
                lr=self.config.learning_rate,
                eps=self.config.adam_epsilon,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == SGD:
            print("Warning: SGD does not use adam_epsilon param")
            self.optimizer: torch.optim.Optimizer = self.config.optimizer(
                params=list(self.encoder.parameters())
                + list(self.decoder.parameters())
                + list(self.dynamics_predictor.parameters())
                + list(self.reward_predictor.parameters()),
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
            )

        self.replay_buffer = RSSMReplayBuffer(
            observation_dimensions=env.observation_space.shape,
            observation_dtype=torch.float32,
            max_size=self.config.replay_buffer_size,
            batch_size=self.config.batch_size,
            batch_length=self.config.batch_length,
        )

    def generate_rollout(
        self, actions, hiddens=None, states=None, observations=None, dones=None
    ):
        if hiddens is None:
            hiddens = torch.zeros((actions.size(0), self.hidden_dim)).to(self.device)
        if states is None:
            states = torch.zeros((actions.size(0), self.state_dim)).to(self.device)

        return self.dynamics_predictor(hiddens, states, actions, observations, dones)

    def preprocess(self, observation):
        # if self.config.game.is_image:
        observation = (
            observation.float() / 255.0 if self.config.is_image else observation
        )
        # else:
        #     observation = observation.astype(np.float32)
        # return observation

    def train(
        self,
        save_images=True,
    ):
        observation, info = self.env.reset()
        for step in range(self.config.training_steps):
            gc.collect()
            torch.cuda.empty_cache()
            print(f"Training step: {step}/{self.config.training_steps}")
            for i in range(self.config.batch_size):
                for j in range(self.config.batch_length):
                    action = self.env.action_space.sample()
                    observation, reward, terminated, truncated, info = self.env.step(
                        action
                    )
                    done = terminated or truncated
                    self.replay_buffer.store(
                        observation=observation,
                        info=info,
                        action=action,
                        reward=reward,
                        done=done,
                    )
                    if done:
                        observation, info = self.env.reset()

            sample = self.replay_buffer.sample()
            observations = (
                torch.from_numpy(sample["observations"]).float().to(self.device)
            )
            actions = torch.from_numpy(sample["actions"]).long().to(self.device)
            rewards = torch.from_numpy(sample["rewards"]).float().to(self.device)
            dones = torch.from_numpy(sample["dones"]).to(self.device)

            actions = torch.tensor(actions).long()
            actions = F.one_hot(actions, num_classes=self.action_dim).float()

            encoded_observations = self.encoder(
                # self.preprocess(
                observations.reshape(-1, *observations.shape[2:]).permute(0, 3, 1, 2)
                / 255.0
                if self.config.is_image
                else observations.reshape(-1, *observations.shape[2:])
                # )
            )
            encoded_observations = encoded_observations.reshape(
                self.config.batch_size, self.config.batch_length, -1
            )

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
            print("Hiddens shape:", hiddens.shape)
            print("Observations shape:", observations.shape)

            hiddens = hiddens.reshape(
                self.config.batch_size * self.config.batch_length, -1
            ).to(self.device)
            posterior_states = posterior_states.reshape(
                self.config.batch_size * self.config.batch_length, self.state_dim
            ).to(self.device)

            decoded_observations = self.decoder(hiddens, posterior_states)
            decoded_observations = decoded_observations.reshape(
                self.config.batch_size,
                self.config.batch_length,
                *self.env.observation_space.shape,
            )

            reward_output = self.reward_predictor(hiddens, posterior_states)

            means, log_vars = torch.chunk(reward_output, 2, dim=-1)
            log_vars = F.softplus(log_vars)
            reward_distribution = torch.distributions.Normal(means, torch.exp(log_vars))
            predicted_rewards = reward_distribution.rsample()
            predicted_rewards = predicted_rewards.reshape(
                self.config.batch_size, self.config.batch_length, 1
            )

            if save_images:
                batch_idx = np.random.randint(0, self.config.batch_size)
                seq_idx = np.random.randint(0, self.config.batch_length - 3)
                fig = self._visualize(
                    observations / 255.0 if self.config.is_image else observations,
                    decoded_observations,
                    rewards,
                    predicted_rewards,
                    batch_idx,
                    seq_idx,
                    step,
                    grayscale=False,
                )
                if not os.path.exists("reconstructions"):
                    os.makedirs("reconstructions")
                fig.savefig(f"reconstructions/iteration_{step}.png")
                plt.close(fig)

            prediction_loss = self._prediction_loss(
                decoded_observations, observations, predicted_rewards, rewards
            )

            # should be a dynamics loss and representation loss for each obs, so (batch_size, batch_length)
            dynamics_loss = self._dynamics_loss(
                prior_means,
                F.softplus(prior_log_vars),
                posterior_means,
                F.softplus(posterior_log_vars),
            )
            representation_loss = self._representation_loss(
                prior_means,
                F.softplus(prior_log_vars),
                posterior_means,
                F.softplus(posterior_log_vars),
            )
            print(
                f"Prediction Loss: {prediction_loss.mean()}, Dynamics Loss: {dynamics_loss.mean()}, Representation Loss: {representation_loss.mean()}"
            )
            # loss = reconstruction_loss + dynamics_loss + reward_loss
            loss = (
                self.config.prediction_loss_coeff * prediction_loss
                + self.config.dynamics_loss_coeff * dynamics_loss
                + self.config.representation_loss_coeff * representation_loss
            )
            self.optimizer.zero_grad()
            loss.mean().backward()
            if self.config.clipnorm > 0:
                # print("clipnorm", self.config.clipnorm)
                clip_grad_norm_(
                    list(self.encoder.parameters())
                    + list(self.decoder.parameters())
                    + list(self.dynamics_predictor.parameters())
                    + list(self.reward_predictor.parameters()),
                    self.config.clipnorm,
                )
            self.optimizer.step()

    # not currently same as dreamer paper
    # def _prediction_loss(self, decoded_obs, obs):
    def _prediction_loss(
        self,
        decoded_obs,
        obs,
        predicted_reward,
        reward,  # continue_pred, done
    ):
        return (
            F.mse_loss(
                decoded_obs,
                obs / 255.0 if self.config.is_image else obs,
            )
            # -decoded_obs.log_prob(obs).mean()
            + F.mse_loss(predicted_reward, reward)
            # + -predicted_reward.log_prob(
            #     reward
            # ).mean()  # true reward should be twohot? i think
            # + -continue_pred.log_prob(done).mean()  # a distribution of true false
        )

    # not currently same as dreamer paper
    def _dynamics_loss(
        self,
        prior_means,
        prior_logvars,
        posterior_means,
        posterior_logvars,
        clip=1.0,
        uniform=0.01,
    ):
        prior_dist = Normal(prior_means, torch.exp(prior_logvars))
        posterior_dist = Normal(
            posterior_means.detach(), torch.exp(posterior_logvars.detach())
        )
        # print(prior_dist.batch_shape)
        # print(prior_dist.event_shape)
        # prior_dist = prior_dist * (1 - uniform) + uniform / prior_dist.event_shape
        # posterior_dist = (
        #     posterior_dist * (1 - uniform) + uniform / posterior_dist.event_shape
        # )

        # return torch.max(
        #     torch.ones(posterior_dist.batch_shape[:2]) * clip,
        #     kl_divergence(posterior_dist, prior_dist).sum(dim=-1),
        # )
        return torch.clip(
            kl_divergence(posterior_dist, prior_dist),
            min=clip,
        ).mean()

    # not currently same as dreamer paper
    # def _representation_loss(self, rewards, predicted_rewards):
    def _representation_loss(
        self,
        prior_means,
        prior_logvars,
        posterior_means,
        posterior_logvars,
        clip=1.0,
        uniform=0.01,
    ):
        prior_dist = Normal(prior_means.detach(), torch.exp(prior_logvars.detach()))
        posterior_dist = Normal(posterior_means, torch.exp(posterior_logvars))
        # print(prior_dist.batch_shape)
        # print(prior_dist.event_shape)
        # prior_dist = prior_dist * (1 - uniform) + uniform / prior_dist.event_shape
        # posterior_dist = (
        #     posterior_dist * (1 - uniform) + uniform / posterior_dist.event_shape
        # )
        # return torch.max(
        #     torch.ones(posterior_dist.batch_shape[:2]) * clip,
        #     kl_divergence(posterior_dist, prior_dist).sum(dim=-1),
        # )
        return torch.clip(
            kl_divergence(posterior_dist, prior_dist),
            min=clip,
        ).mean()

    def _visualize(
        self,
        obs,
        decoded_obs,
        rewards,
        predicted_rewards,
        batch_idx,
        seq_idx,
        iterations: int,
        grayscale: bool = True,
    ):
        obs = obs[batch_idx, seq_idx : seq_idx + 3]
        decoded_obs = decoded_obs[batch_idx, seq_idx : seq_idx + 3]
        rewards = rewards[batch_idx, seq_idx : seq_idx + 3]
        predicted_rewards = predicted_rewards[batch_idx][seq_idx : seq_idx + 3]

        obs = obs.cpu().detach().numpy()
        decoded_obs = decoded_obs.cpu().detach().numpy()

        print(obs[0].shape)
        print(decoded_obs[0].shape)
        print("shapes should be 96 x 96 x 3")

        fig, axs = plt.subplots(3, 2)
        axs[0][0].imshow(obs[0], cmap="gray" if grayscale else None)
        axs[0][0].set_title(f"Iteration: {iterations} -- Reward: {rewards[0, 0]:.4f}")
        axs[0][0].axis("off")
        axs[0][1].imshow(decoded_obs[0], cmap="gray" if grayscale else None)
        axs[0][1].set_title(f"Pred. Reward: {predicted_rewards[0, 0]:.4f}")
        axs[0][1].axis("off")

        axs[1][0].imshow(obs[1], cmap="gray" if grayscale else None)
        axs[1][0].axis("off")
        axs[1][0].set_title(f"Reward: {rewards[1, 0]:.4f} ")
        axs[1][1].imshow(decoded_obs[1], cmap="gray" if grayscale else None)
        axs[1][1].set_title(f"Pred. Reward: {predicted_rewards[1, 0]:.4f}")
        axs[1][1].axis("off")

        axs[2][0].imshow(obs[2], cmap="gray" if grayscale else None)
        axs[2][0].axis("off")
        axs[2][0].set_title(f"Reward: {rewards[2, 0]:.4f}")
        axs[2][1].imshow(decoded_obs[2], cmap="gray" if grayscale else None)
        axs[2][1].set_title(f"Pred. Reward: {predicted_rewards[2, 0]:.4f}")
        axs[2][1].axis("off")

        return fig
