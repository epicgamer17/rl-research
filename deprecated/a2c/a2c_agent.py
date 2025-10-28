import dis
from cv2 import log
import torch
import torch.nn as nn
from utils.utils import (
    action_mask,
    clip_low_prob_actions,
    get_legal_moves,
    normalize_policies,
    current_timestamp,
)
import sys
from time import time

sys.path.append("../")

from replay_buffers.deprecated.a2c_replay_buffer import A2CReplayBuffer
from a2c_network import A2CNetwork
import numpy as np
import gymnasium as gym
from agent_configs import A2CConfig
from base_agent.agent import BaseAgent
from torch.optim import Adam, SGD
from torch.nn.utils import clip_grad_norm_


class A2CAgent(BaseAgent):
    def __init__(
        self,
        env,
        config: A2CConfig,
        name=f"A2C{current_timestamp():.1f}",
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
        from_checkpoint=False,
    ):
        super(A2CAgent, self).__init__(
            env, config, name, device=device, from_checkpoint=from_checkpoint
        )
        self.model = A2CNetwork(
            config=config,
            output_size=self.num_actions,
            input_shape=(self.config.minibatch_size,) + self.observation_dimensions,
            discrete=self.config.game.is_discrete,
        )
        if not self.config.kernel_initializer == None:
            self.model.initialize(self.config.kernel_initializer)

        self.model.to(device)

        if self.config.actor.optimizer == Adam:
            self.actor_optimizer: torch.optim.Optimizer = self.config.actor.optimizer(
                params=self.model.actor.parameters(),
                lr=self.config.actor.learning_rate,
                eps=self.config.actor.adam_epsilon,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.actor.optimizer == SGD:
            print("Warning: SGD does not use adam_epsilon param")
            self.actor_optimizer: torch.optim.Optimizer = self.config.actor.optimizer(
                params=self.model.actor.parameters(),
                lr=self.config.actor.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
            )

        if self.config.critic.optimizer == Adam:
            self.critic_optimizer: torch.optim.Optimizer = self.config.critic.optimizer(
                params=self.model.critic.parameters(),
                lr=self.config.critic.learning_rate,
                eps=self.config.critic.adam_epsilon,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.critic.optimizer == SGD:
            print("Warning: SGD does not use adam_epsilon param")
            self.critic_optimizer: torch.optim.Optimizer = self.config.critic.optimizer(
                params=self.model.critic.parameters(),
                lr=self.config.critic.learning_rate,
                momentum=self.config.momentum,
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
        self.replay_buffer = A2CReplayBuffer(
            observation_dimensions=self.observation_dimensions,
            observation_dtype=self.env.observation_space.dtype,
            max_size=self.config.replay_buffer_size,
            gamma=self.config.discount_factor,
        )

        self.stats = {
            "score": [],
            "critic_loss": [],
            "actor_loss": [],
            "test_score": [],
        }
        self.targets = {
            "score": self.env.spec.reward_threshold,
            "test_score": self.env.spec.reward_threshold,
        }

    def checkpoint_optimizer_state(self, checkpoint):
        checkpoint["actor_optimizer"] = self.actor_optimizer.state_dict()
        checkpoint["critic_optimizer"] = self.critic_optimizer.state_dict()
        return checkpoint

    def load_optimizer_state(self, checkpoint):
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])

    def select_actions(self, prediction, info=None, actionmasking={}):
        policy, distribution, value = prediction
        action = distribution.sample()

        return action

    def predict(self, state, info: dict = None, mask_actions: bool = False):
        assert info is not None if mask_actions else True, "Need info to mask actions"
        state_input = self.preprocess(state).to(self.device)
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
        return policy, distribution, value

    def print_graph(self, grad_fn, level=0):
        if grad_fn is not None:
            print(f"{' ' * level * 2} {grad_fn}")
            for next_fn, _ in grad_fn.next_functions:
                self.print_graph(next_fn, level + 1)

    def learn(self):
        # add training iterations?
        samples = self.replay_buffer.sample()
        log_probabilities = samples["log_probabilities"]
        values = samples["values"]
        advantages = torch.from_numpy(samples["advantages"])
        returns = torch.from_numpy(samples["returns"])
        distributions = samples["distributions"]

        advantages = advantages.to(self.device)

        self.actor_learn(advantages, log_probabilities, distributions)
        self.critic_learn(returns, values)

    def actor_learn(self, advantages, log_probabilities, distributions):
        # minimizing the loss

        actor_loss = 0.0
        for i in range(len(log_probabilities)):
            actor_loss += log_probabilities[i] * advantages[i]
        actor_loss = actor_loss / len(log_probabilities)
        entropy_loss = 0
        for i in distributions:
            entropy_loss += i.entropy()
        entropy_loss = entropy_loss / len(distributions)
        actor_loss = -(actor_loss - (self.config.entropy_coefficient * entropy_loss))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()

        if self.config.actor.clipnorm > 0:
            clip_grad_norm_(self.model.actor.parameters(), self.config.actor.clipnorm)
        self.stats["actor_loss"].append(actor_loss.detach().item())
        self.actor_optimizer.step()
        return actor_loss

    def critic_learn(self, returns, values):

        critic_loss = 0.0
        for i in range(len(values)):
            critic_loss += (
                self.config.critic_coefficient * (returns[i] - values[i]) ** 2
            )
        critic_loss = critic_loss / len(values)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.config.critic.clipnorm > 0:
            clip_grad_norm_(self.model.critic.parameters(), self.config.critic.clipnorm)
        self.stats["critic_loss"].append(critic_loss.detach().item())
        self.critic_optimizer.step()
        return critic_loss.detach()

    def train(self):
        super().train()
        start_time = time() - self.training_time
        if self.training_step == 0:
            self.print_resume_training()
        # Training loop
        while self.training_time < self.config.max_training_time:
            if self.training_step % self.config.print_interval == 0:
                self.print_training_progress()
            score = 0
            # get new init state
            state, info = self.env.reset()
            done = False
            # reset time step counter, done flag, epsiode list, state
            while not done:
                # since monte carlo sort of alg need to finish the episode before learning
                prediction = self.predict(state, info)
                # represents predictions from actor and critic
                action = self.select_actions(prediction, info)
                action = action.item()
                policy, distribution, value = prediction
                log_probability = torch.log(policy[action])
                # take action
                next_state, reward, terminated, truncated, next_info = self.env.step(
                    action
                )
                # NEED TO ADD A REPLAY BUFFER
                self.replay_buffer.store(
                    value=value,
                    log_probability=log_probability,
                    reward=reward,
                    distribution=distribution,
                )
                done = terminated or truncated
                state = next_state
                info = next_info
                score = score + reward

            self.replay_buffer.compute_advantage_and_returns()
            score = {"score": score}
            self.stats["score"].append(score)
            self.learn()
            self.replay_buffer.clear()
            if self.training_step % self.checkpoint_interval == 0:
                self.training_time = time() - start_time
                self.total_environment_steps += self.config.steps_per_epoch
            self.training_step += 1

        self.training_time = time() - start_time
        self.total_environment_steps = self.training_step * self.config.steps_per_epoch
        self.save_checkpoint()

        # NEED TO ADD CHECKPOINTING AND STAT TRACKING AND GRAPHING
