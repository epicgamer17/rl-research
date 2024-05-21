import sys
from time import time

from agent_configs import PPOConfig

from utils import (
    normalize_policy,
    action_mask,
    get_legal_moves,
    update_linear_lr_schedule,
)

sys.path.append("../")

import os

os.environ["OMP_NUM_THREADS"] = f"{8}"
os.environ["MKL_NUM_THREADS"] = f"{8}"
os.environ["TF_NUM_INTEROP_THREADS"] = f"{8}"
os.environ["TF_NUM_INTRAOP_THREADS"] = f"{8}"

import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(0)
tf.config.threading.set_inter_op_parallelism_threads(0)

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # memory growth must be set before GPUs have been initialized
        print(e)

import datetime
import numpy as np
from ppo.ppo_network import Network
from replay_buffers.ppo_replay_buffer import ReplayBuffer
import tensorflow_probability as tfp
from base_agent.agent import BaseAgent


class PPOAgent(BaseAgent):
    def __init__(
        self,
        env,
        config: PPOConfig,
        name=datetime.datetime.now().timestamp(),
    ):
        super(PPOAgent, self).__init__(env, config, name)

        self.model = Network(
            config,
            self.observation_dimensions,
            self.num_actions,
            self.discrete_action_space,  # COULD USE GAME CONFIG?
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

        self.replay_buffer = ReplayBuffer(
            observation_dimensions=self.observation_dimensions,
            max_size=self.config.replay_buffer_size,
            gamma=self.config.discount_factor,
            gae_lambda=self.config.gae_lambda,
        )

        self.transition = list()

    def predict_single(self, state, legal_moves=None):
        state_input = self.prepare_states(state)
        value = self.model.critic(inputs=state_input).numpy()
        if self.discrete_action_space:
            policy = self.model.actor(inputs=state_input).numpy()[0]
            policy = action_mask(policy, legal_moves, self.num_actions, mask_value=0)
            policy = normalize_policy(policy)
            return policy, value
        else:
            mean, std = self.model.actor(inputs=state_input)
            return mean, std, value

    def select_action(self, state, legal_moves=None):
        if self.discrete_action_space:
            policy, value = self.predict_single(state)
            distribution = tfp.distributions.Categorical(probs=policy)
        else:
            mean, std, value = self.predict_single(state)
            distribution = tfp.distributions.Normal(mean, std)

        if self.is_test:
            selected_action = distribution.mode().numpy()
        else:
            selected_action = distribution.sample().numpy()

        if len(selected_action) == 1:
            selected_action = selected_action[0]
        log_probability = distribution.log_prob(selected_action)

        value = value[0][0]

        if not self.is_test:
            self.transition = [state, selected_action, value, log_probability]
        return selected_action

    def step(self, action):
        if not self.is_test:
            next_state, reward, terminated, truncated, info = self.env.step(action)
            self.transition += [reward]
            self.replay_buffer.store(*self.transition)
        else:
            next_state, reward, terminated, truncated, info = self.test_env.step(action)

        return next_state, reward, terminated, truncated, info

    def train_actor(
        self, inputs, actions, log_probabilities, advantages, learning_rate
    ):
        # print("Training Actor")
        with tf.GradientTape() as tape:
            if self.discrete_action_space:
                distribution = tfp.distributions.Categorical(self.model.actor(inputs))
            else:
                mean, std = self.model.actor(inputs)
                distribution = tfp.distributions.Normal(mean, std)

            log_ratios = distribution.log_prob(actions) - log_probabilities

            probability_ratios = tf.exp(log_ratios)
            # min_advantages = tf.where(
            #     advantages > 0,
            #     (1 + self.clip_param) * advantages,
            #     (1 - self.clip_param) * advantages,
            # )

            clipped_probability_ratios = tf.clip_by_value(
                probability_ratios,
                1 - self.config.clip_param,
                1 + self.config.clip_param,
            )

            actor_loss = tf.math.minimum(
                probability_ratios * advantages, clipped_probability_ratios * advantages
            )

            entropy_loss = distribution.entropy()
            actor_loss = -tf.reduce_mean(actor_loss) - (
                self.config.entropy_coefficient * entropy_loss
            )

        actor_gradients = tape.gradient(
            actor_loss, self.model.actor.trainable_variables
        )
        self.config.actor.optimizer.apply_gradients(
            grads_and_vars=zip(actor_gradients, self.model.actor.trainable_variables)
        )
        if self.discrete_action_space:
            kl_divergence = tf.reduce_mean(
                log_probabilities
                - tfp.distributions.Categorical(self.model.actor(inputs)).log_prob(
                    actions
                )
            )
        else:
            mean, std = self.model.actor(inputs)
            kl_divergence = tf.reduce_mean(
                log_probabilities
                - tfp.distributions.Normal(mean, std).log_prob(actions)
            )
        kl_divergence = tf.reduce_sum(kl_divergence)
        return kl_divergence

    def train_critic(self, inputs, returns, learning_rate):
        with tf.GradientTape() as tape:
            critic_loss = tf.reduce_mean(
                (returns - self.model.critic(inputs, training=True)) ** 2
            )
        critic_gradients = tape.gradient(
            critic_loss, self.model.critic.trainable_variables
        )
        self.config.critic.optimizer.apply_gradients(
            grads_and_vars=zip(critic_gradients, self.model.critic.trainable_variables)
        )

        return critic_loss

    def train(self):
        training_time = time()
        self.is_test = False
        stats = {
            "score": [],
            "actor_loss": [],
            "critic_loss": [],
            "test_score": [],
        }
        targets = {
            "score": self.env.spec.reward_threshold,
            "test_score": self.env.spec.reward_threshold,
            "actor_loss": self.config.target_kl,
        }

        state, _ = self.env.reset()
        for training_step in range(self.training_steps):
            print("Training Step: ", training_step)
            num_episodes = 0
            total_score = 0
            score = 0
            for timestep in range(self.config.steps_per_epoch):
                action = self.select_action(
                    state,
                    get_legal_moves(info),
                )
                next_state, reward, terminated, truncated, info = self.step(action)
                done = terminated or truncated
                state = next_state
                score += reward

                if done or timestep == self.config.steps_per_epoch - 1:
                    last_value = (
                        0
                        if done
                        else self.model.critic(self.prepare_states(next_state))
                    )
                    self.replay_buffer.finish_trajectory(last_value)
                    num_episodes += 1
                    state, _ = self.env.reset()
                    # if score >= self.env.spec.reward_threshold:
                    #     print("Your agent has achieved the env's reward threshold.")
                    total_score += score
                    score = 0

            samples = self.replay_buffer.get()
            observations = samples["observations"]
            actions = samples["actions"]
            log_probabilities = samples["log_probabilities"]
            advantages = samples["advantages"]
            returns = samples["returns"]
            inputs = self.prepare_states(observations)

            indices = np.arange(len(observations))
            np.random.shuffle(indices)
            minibatch_size = len(observations) // self.config.num_minibatches

            for iteration in range(self.config.train_policy_iterations):
                learning_rate = max(learning_rate, 0)
                learning_rate = update_linear_lr_schedule(
                    learning_rate,
                    0,
                    self.training_steps,
                    self.config.actor.learning_rate,
                    iteration,
                )
                for start in range(0, len(observations), minibatch_size):
                    end = start + minibatch_size
                    batch_indices = indices[start:end]
                    batch_observations = inputs[batch_indices]
                    batch_actions = actions[batch_indices]
                    batch_log_probabilities = log_probabilities[batch_indices]
                    batch_advantages = advantages[batch_indices]
                    kl_divergence = self.train_actor(
                        batch_observations,
                        batch_actions,
                        batch_log_probabilities,
                        batch_advantages,
                        learning_rate,
                    )
                    stats["actor_loss"].append(kl_divergence)
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

            for iteration in range(self.config.train_value_iterations):
                learning_rate = update_linear_lr_schedule(
                    learning_rate,
                    0,
                    self.training_steps,
                    self.config.critic.learning_rate,
                    iteration,
                )
                learning_rate = max(learning_rate, 0)
                for start in range(0, len(observations), minibatch_size):
                    end = start + minibatch_size
                    batch_indices = indices[start:end]
                    batch_observations = inputs[batch_indices]
                    batch_returns = returns[batch_indices]
                    critic_loss = self.train_critic(
                        batch_observations, batch_returns, learning_rate
                    )
                    stats["critic_loss"].append(critic_loss)
                # critic_loss = self.train_critic(inputs, returns, learning_rate)
                # stat_critic_loss.append(critic_loss)
                # stat_loss.append(critic_loss)

            # self.old_actor.set_weights(self.actor.get_weights())
            stats["score"].append(total_score / num_episodes)
            if training_step % self.checkpoint_interval == 0 and training_step > 0:
                self.save_checkpoint(
                    stats,
                    targets,
                    5,
                    training_step,
                    training_step * self.config.steps_per_epoch,
                    time() - training_time,
                )

        self.save_checkpoint(
            stats,
            targets,
            5,
            training_step,
            training_step * self.config.steps_per_epoch,
            time() - training_time,
        )
        self.env.close()
