import sys
from time import time

import torch
from torch.nn.utils import clip_grad_norm_

from agent_configs import PPOConfig

from utils import (
    normalize_policy,
    action_mask,
    get_legal_moves,
)

sys.path.append("../")

import datetime
from ppo.ppo_network import Network
from replay_buffers.base_replay_buffer import BasePPOReplayBuffer
from base_agent.agent import BaseAgent


class PPOAgent(BaseAgent):
    def __init__(
        self,
        env,
        config: PPOConfig,
        name=datetime.datetime.now().timestamp(),
        device: torch.device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            # MPS is sometimes useful for M2 instances, but only for large models/matrix multiplications otherwise CPU is faster
            # else (
            #     torch.device("mps")
            #     if torch.backends.mps.is_available() and torch.backends.mps.is_built()
            else torch.device("cpu")
            # )
        ),
    ):
        super(PPOAgent, self).__init__(env, config, name, device=device)
        self.model = Network(
            config=config,
            output_size=self.num_actions,
            input_shape=(self.config.minibatch_size,) + self.observation_dimensions,
            discrete=self.discrete_action_space,  # COULD USE GAME CONFIG?
        )

        self.actor_optimizer: torch.optim.Optimizer = self.config.actor.optimizer(
            params=self.model.actor.parameters(),
            lr=self.config.learning_rate,
            eps=self.config.adam_epsilon,
        )

        self.critic_optimizer: torch.optim.Optimizer = self.config.critic.optimizer(
            params=self.model.critic.parameters(),
            lr=self.config.learning_rate,
            eps=self.config.adam_epsilon,
        )

        # self.actor = ActorNetwork(
        #     input_shape=self.observation_dimensions,
        #     output_shape=self.num_actions,
        #     discrete=self.discrete_action_space,
        #     config=config,
        # )

        # self.critic = CriticNetwork(
        #     input_shape=self.observation_dimensions,
        #     config=config,
        # )

        self.replay_buffer = BasePPOReplayBuffer(
            observation_dimensions=self.observation_dimensions,
            max_size=self.config.replay_buffer_size,
            gamma=self.config.discount_factor,
            gae_lambda=self.config.gae_lambda,
        )

        self.transition = list()

        self.stats = {
            "score": [],
            "actor_loss": [],
            "critic_loss": [],
            "test_score": [],
        }
        self.targets = {
            "score": self.env.spec.reward_threshold,
            "test_score": self.env.spec.reward_threshold,
            "actor_loss": self.config.target_kl,
        }

    def predict(self, state, legal_moves=None):
        state_input = self.preprocess(state)
        value = self.model.critic(inputs=state_input)
        if self.discrete_action_space:
            policy = self.model.actor(inputs=state_input)[0]
            # print(policy)
            # policy = action_mask(policy, legal_moves, self.num_actions, mask_value=0)
            # print(policy)
            # policy = normalize_policy(policy)
            # print(policy)
            return policy, value
        else:
            mean, std = self.model.actor(inputs=state_input)
            return mean, std, value

    def select_actions(self, state, legal_moves=None):
        if self.discrete_action_space:
            policy, value = self.predict(state)
            distribution = torch.distributions.Categorical(probs=policy)
        else:
            mean, std, value = self.predict(state)
            distribution = torch.distributions.Normal(mean, std)

        # if self.is_test:
        #     selected_action = distribution.mode
        # else:
        #     selected_action = distribution.sample().numpy()
        selected_action = distribution.sample()
        # if len(selected_action) == 1:
        #     selected_action = selected_action[0]
        log_probability = distribution.log_prob(selected_action)
        print(value)
        value = value[0][0]

        if not self.is_test:
            self.transition = [state, selected_action, value, log_probability]
        return selected_action.item()

    def actor_learn(self, inputs, actions, log_probabilities, advantages):
        # print("Training Actor")
        if self.discrete_action_space:
            distribution = torch.distributions.Categorical(self.model.actor(inputs))
        else:
            mean, std = self.model.actor(inputs)
            distribution = torch.distributions.Normal(mean, std)
        tensor_actions = (
            torch.clone(actions).to(torch.float16).detach().requires_grad_(True)
        )
        log_ratios = distribution.log_prob(tensor_actions) - log_probabilities

        probability_ratios = torch.exp(log_ratios)
        # min_advantages = tf.where(
        #     advantages > 0,
        #     (1 + self.clip_param) * advantages,
        #     (1 - self.clip_param) * advantages,
        # )

        clipped_probability_ratios = torch.clamp(
            probability_ratios,
            1 - self.config.clip_param,
            1 + self.config.clip_param,
        )

        actor_loss = torch.minimum(
            probability_ratios * advantages, clipped_probability_ratios * advantages
        )

        entropy_loss = distribution.entropy().mean()
        actor_loss = -((actor_loss) - (self.config.entropy_coefficient * entropy_loss))

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.config.actor.clipnorm > 0:
            clip_grad_norm_(self.model.actor.parameters(), self.config.clipnorm)

        self.actor_optimizer.step()
        actor_loss = actor_loss  # .detach()
        with torch.no_grad():
            if self.discrete_action_space:
                kl_divergence = torch.mean(
                    log_probabilities - distribution.log_prob(actions)
                )
            else:
                mean, std = self.model.actor(inputs)
                kl_divergence = torch.mean(
                    log_probabilities - distribution.log_prob(actions)
                )
            kl_divergence = torch.sum(kl_divergence)
            print("Method 2", kl_divergence)
            approx_kl = ((probability_ratios - 1) - log_ratios).mean()
            print(
                "Method 1",
                approx_kl,
            )

        return kl_divergence

    def critic_learn(self, inputs, returns):
        critic_loss = (returns - self.model.critic(inputs)) ** 2

        self.critic_optimizer.zero_grad()
        critic_loss.mean().backward()
        if self.config.critic.clipnorm > 0:
            clip_grad_norm_(self.model.critic.parameters(), self.config.clipnorm)

        self.critic_optimizer.step()
        critic_loss = critic_loss  # .detach()
        return critic_loss.mean()

    def train(self):
        training_time = time()
        self.is_test = False

        state, info = self.env.reset()
        self.training_steps += self.start_training_step
        for training_step in range(self.start_training_step, self.training_steps):
            with torch.no_grad():
                print("Training Step: ", training_step)
                num_episodes = 0
                total_score = 0
                score = 0
                for timestep in range(self.config.steps_per_epoch):
                    action = self.select_actions(
                        state,
                        get_legal_moves(info),
                    )

                    next_state, reward, terminated, truncated, info = self.env.step(
                        action
                    )
                    if not self.is_test:
                        self.transition += [reward]
                    self.replay_buffer.store(*self.transition)

                    done = terminated or truncated
                    state = next_state
                    score += reward

                    if done or timestep == self.config.steps_per_epoch - 1:
                        last_value = (
                            0
                            if done
                            else self.model.critic(self.preprocess(next_state))
                        )
                        self.replay_buffer.finish_trajectory(last_value)
                        num_episodes += 1
                        state, info = self.env.reset()
                        # if score >= self.env.spec.reward_threshold:
                        #     print("Your agent has achieved the env's reward threshold.")
                        total_score += score
                        score = 0

                samples = self.replay_buffer.sample()
                observations = samples["observations"]
                actions = torch.from_numpy(samples["actions"]).to(self.device)
                log_probabilities = torch.from_numpy(samples["log_probabilities"]).to(
                    self.device
                )
                advantages = torch.from_numpy(samples["advantages"]).to(self.device)
                returns = torch.from_numpy(samples["returns"]).to(self.device)
                inputs = self.preprocess(observations)

                indices = torch.randperm(len(observations))
                minibatch_size = len(observations) // self.config.num_minibatches

                actor_scheduler = torch.optim.lr_scheduler.LinearLR(
                    self.actor_optimizer,
                    self.config.actor.learning_rate,
                    0,
                    total_iters=self.config.train_policy_iterations,
                )
                for iteration in range(self.config.train_policy_iterations):
                    actor_scheduler.step()
                    for start in range(0, len(observations), minibatch_size):
                        end = start + minibatch_size
                        batch_indices = indices[start:end]
                        batch_observations = inputs[batch_indices]
                        batch_actions = actions[batch_indices]
                        batch_log_probabilities = log_probabilities[batch_indices]
                        batch_advantages = advantages[batch_indices]
                        kl_divergence = self.actor_learn(
                            batch_observations,
                            batch_actions,
                            batch_log_probabilities,
                            batch_advantages,
                        )
                        self.stats["actor_loss"].append(kl_divergence)
                    if kl_divergence > 1.5 * self.config.target_kl:
                        print("Early stopping at iteration {}".format(_))
                        break
                    # kl_divergence = self.train_actor(
                    #     inputs, actions, log_probabilities, advantages, learning_rate
                    # )
                    # stat_actor_loss.append(kl_divergence)
                    # if kl_divergence > 1.5 * self.target_kl:
                    #     print("Early stopping at iteration {}".format(_))
                    #     break
                critic_scheduler = torch.optim.lr_scheduler.LinearLR(
                    self.critic_optimizer,
                    self.config.critic.learning_rate,
                    0,
                    total_iters=self.config.train_value_iterations,
                )
                for iteration in range(self.config.train_value_iterations):
                    critic_scheduler.step()
                    for start in range(0, len(observations), minibatch_size):
                        end = start + minibatch_size
                        batch_indices = indices[start:end]
                        batch_observations = inputs[batch_indices]
                        batch_returns = returns[batch_indices]
                        critic_loss = self.critic_learn(
                            batch_observations,
                            batch_returns,
                        )
                        self.stats["critic_loss"].append(critic_loss)
                    # critic_loss = self.train_critic(inputs, returns, learning_rate)
                    # stat_critic_loss.append(critic_loss)
                    # stat_loss.append(critic_loss)

                # self.old_actor.set_weights(self.actor.get_weights())
                self.stats["score"].append(total_score / num_episodes)
                if training_step % self.checkpoint_interval == 0 and training_step > 0:
                    self.save_checkpoint(
                        5,
                        training_step,
                        training_step * self.config.steps_per_epoch,
                        time() - training_time,
                    )

        self.save_checkpoint(
            5,
            training_step,
            training_step * self.config.steps_per_epoch,
            time() - training_time,
        )
        self.env.close()
