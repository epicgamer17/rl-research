import dis
from cv2 import log
import torch
import torch.nn as nn
from utils.utils import (
    action_mask,
    clip_low_prob_actions,
    get_legal_moves,
    normalize_policies,
)
from a2c_network import NeuralNetwork
import numpy as np
import gymnasium as gym

from base_agent.agent import BaseAgent


class A2CAgent(BaseAgent):
    def __init__(
        self,
        env,
        config: A2CConfig,
        name=f"rainbow_{current_timestamp():.1f}",
        device: torch.device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            # MPS is sometimes useful for M2 instances, but only for large models/matrix multiplications otherwise CPU is faster
            else (
                torch.device("mps")
                if torch.backends.mps.is_available() and torch.backends.mps.is_built()
                else torch.device("cpu")
            )
        ),
    ):
        super(A2CAgent, self).__init__(env, config, name, device=device)
        self.model = A2CNetwork(
            config=config,
            output_size=self.num_actions,
            input_shape=(self.config.minibatch_size,) + self.observation_dimensions,
        )

        if not self.config.kernel_initializer == None:
            self.model.initialize(self.config.kernel_initializer)

        self.model.to(device)

        if self.config.actor.optimizer == Adam:
            self.actor_optimizer: torch.optim.Optimizer = self.config.actor.optimizer(
                params=self.model.parameters(),
                lr=self.config.learning_rate,
                eps=self.config.adam_epsilon,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.actor.optimizer == SGD:
            print("Warning: SGD does not use adam_epsilon param")
            self.actor_optimizer: torch.optim.Optimizer = self.config.actor.optimizer(
                params=self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )

        if self.config.critic.optimizer == Adam:
            self.critic_optimizer: torch.optim.Optimizer = self.config.critic.optimizer(
                params=self.model.parameters(),
                lr=self.config.learning_rate,
                eps=self.config.adam_epsilon,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.critic.optimizer == SGD:
            print("Warning: SGD does not use adam_epsilon param")
            self.critic_optimizer: torch.optim.Optimizer = self.config.critic.optimizer(
                params=self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )

        # self.replay_buffer = PrioritizedNStepReplayBuffer(
        #     observation_dimensions=self.observation_dimensions,
        #     observation_dtype=self.env.observation_space.dtype,
        #     max_size=self.config.replay_buffer_size,
        #     batch_size=self.config.minibatch_size,
        #     max_priority=1.0,
        #     alpha=self.config.per_alpha,
        #     beta=self.config.per_beta,
        #     # epsilon=config["per_epsilon"],
        #     n_step=self.config.n_step,
        #     gamma=self.config.discount_factor,
        #     compressed_observations=(
        #         self.env.lz4_compress if hasattr(self.env, "lz4_compress") else False
        #     ),
        #     num_players=num_players,
        # )

        self.stats = {
            "score": [],
            "loss": [],
            "test_score": [],
        }
        self.targets = {
            "score": self.env.spec.reward_threshold,
            "test_score": self.env.spec.reward_threshold,
        }

    def select_action(self, predictions):
        distribution = torch.distributions.Categorical(predictions[0])
        action = distribution.sample()
        return action.item()

    def predict(self, state, info: dict = None, mask_actions: bool = True):
        assert info is not None if mask_actions else True, "Need info to mask actions"
        state_input = self.preprocess(state)
        # NEED TO FIX NETWORK
        value = self.model.critic(inputs=state_input)
        if self.discrete_action_space:
            # NEED TO FIX NETWORK
            policy = self.model.actor(inputs=state_input)[0]
            if mask_actions:
                legal_moves = get_legal_moves(info)
                policy = action_mask(
                    policy, legal_moves, mask_value=0, device=self.device
                )
                policy = clip_low_prob_actions(policy, self.config.clip_low_prob)
                policy = normalize_policies(policy)
            distribution = torch.distributions.Categorical(probs=policy)
        else:
            mean, std = self.model.actor(inputs=state_input)
            distribution = torch.distributions.Normal(mean, std)
        return distribution, value

    def learn(self):
        # add training iterations?
        samples = self.replay_buffer.sample()
        log_probabilities = torch.from_numpy(samples["log_probabilities"])
        values = torch.from_numpy(samples["values"])
        advantages = torch.from_numpy(samples["advantages"])
        returns = torch.from_numpy(samples["returns"])

        self.actor_learn(torch.tensor(advantages), log_probabilities)
        self.critic_learn(torch.tensor(returns), values)

    def actor_learn(self, advantages, logprobs):
        # minimizing the loss
        actor_loss = torch.mean(logprobs * advantages)
        entropy_loss = distribution.entropy().mean()
        actor_loss = actor_loss - (self.config.entropy_coefficient * entropy_loss)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()

        if self.config.actor.clipnorm > 0:
            clip_grad_norm_(self.model.actor.parameters(), self.config.actor.clipnorm)

        self.actor_optimizer.step()
        return actor_loss

    def critic_learn(self, returns, values):
        critic_loss = (self.config.critic_coefficient * (returns - values) ** 2).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.config.critic.clipnorm > 0:
            clip_grad_norm_(self.model.critic.parameters(), self.config.critic.clipnorm)

        self.critic_optimizer.step()
        return critic_loss.detach()

    def train(self, env):
        # Training loop
        for i in range(self.config.training_steps):
            print(i)
            # get new init state
            state, info = env.reset()
            done = False
            # reset time step counter, done flag, epsiode list, state
            while not done:
                # since monte carlo sort of alg need to finish the episode before learning
                distribution, value = self.predict(state, info)
                # represents predictions from actor and critic
                action = self.select_action(distribution)
                log_probability = distribution.log_prob(action)

                # take action
                next_state, reward, terminated, truncated, next_info = self.env.step(
                    action
                )

                # NEED TO ADD A REPLAY BUFFER
                self.replay_buffer.store(value, log_probability, reward)

                done = terminated or truncated
                state = next_state
                info = next_info

            self.learn()

            # NEED TO ADD CHECKPOINTING AND STAT TRACKING AND GRAPHING
