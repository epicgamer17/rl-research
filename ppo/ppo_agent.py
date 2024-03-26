import sys

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
import copy
import numpy as np
from ppo.ppo_network import ActorNetwork, CriticNetwork
from replay_buffers.ppo_replay_buffer import ReplayBuffer
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import gymnasium as gym


class PPOAgent:
    def __init__(
        self,
        env,
        model_name=datetime.datetime.now().timestamp(),
        config=None,
    ):
        self.config = config
        self.model_name = model_name
        self.env = env
        self.test_env = copy.deepcopy(env)
        self.observation_dimensions = env.observation_space.shape
        if isinstance(env.action_space, gym.spaces.Discrete):
            self.num_actions = env.action_space.n
            self.discrete_action_space = True
        else:
            self.num_actions = env.action_space.shape[0]
            self.discrete_action_space = False

        self.actor = ActorNetwork(
            input_shape=self.observation_dimensions,
            output_shape=self.num_actions,
            discrete=self.discrete_action_space,
            config=config,
        )

        self.critic = CriticNetwork(
            input_shape=self.observation_dimensions,
            config=config,
        )

        self.actor_optimizer = config["actor_optimizer"]
        self.critic_optimizer = config["critic_optimizer"]
        self.actor_learning_rate = config["actor_learning_rate"]
        self.critic_learning_rate = config["critic_learning_rate"]
        self.actor_clipnorm = config["actor_clipnorm"]
        self.critic_clipnorm = config["critic_clipnorm"]
        self.actor_epsilon = config["actor_epsilon"]
        self.critic_epsilon = config["critic_epsilon"]

        self.clip_param = config["clip_param"]

        self.num_epochs = int(config["num_epochs"])
        self.steps_per_epoch = int(config["steps_per_epoch"])
        self.train_policy_iterations = int(config["train_policy_iterations"])
        self.train_value_iterations = int(config["train_value_iterations"])
        self.target_kl = config["target_kl"]

        # self.replay_batch_size = int(config["replay_batch_size"])
        self.replay_buffer_size = self.steps_per_epoch  # times number of agents
        self.num_minibatches = config["num_minibatches"]

        self.discount_factor = config["discount_factor"]
        self.gae_labmda = config["gae_lambda"]
        self.entropy_coefficient = config["entropy_coefficient"]

        self.replay_buffer = ReplayBuffer(
            observation_dimensions=self.observation_dimensions,
            max_size=self.replay_buffer_size,
            gamma=config["discount_factor"],
        )

        self.transition = list()
        self.is_test = True
        # self.search = search.Search(
        #     scoring_function=self.score_state,
        #     max_depth=config["search_max_depth"],
        #     max_time=config["search_max_time"],
        #     transposition_table=search.TranspositionTable(
        #         buckets=config["search_transposition_table_buckets"],
        #         bucket_size=config["search_transposition_table_bucket_size"],
        #         replacement_strategy=search.TranspositionTable.replacement_strategies[
        #             config["search_transposition_table_replacement_strategy"]
        #         ],
        #     ),
        #     debug=False,
        # )

    def save_checkpoint(self, episode=-1, best_model=False):
        if episode != -1:
            actor_path = "./actor_{}_{}_episodes.keras".format(
                self.model_name, episode + self.start_episode
            )
            critic_path = "./critic_{}_{}_episodes.keras".format(
                self.model_name, episode + self.start_episode
            )
        else:
            actor_path = "./actor_{}.keras".format(self.model_name)
            critic_path = "./critic_{}.keras".format(self.model_name)

        if best_model:
            actor_path = "./actor_best_model.keras"
            critic_path = "./critic_best_model.keras"

        self.actor.save(actor_path)
        self.critic.save(critic_path)

    def prepare_states(self, state):
        state = np.array(state)
        if (self.env.observation_space.high == 255).all():
            state = state / 255
        # print(state.shape)
        if state.shape == self.observation_dimensions:
            new_shape = (1,) + state.shape
            state_input = state.reshape(new_shape)
        else:
            state_input = state
        # print(state_input.shape)
        # observation_high = self.env.observation_space.high
        # observation_low = self.env.observation_space.low
        # for s in state_input:
        #     for i in range(len(s)):
        #         s[i] = s[i] - observation_low[i]
        #         s[i] = s[i] / (observation_high[i] - observation_low[i])
        # print(state_input)
        # NORMALIZE VALUES
        return state_input

    def predict_single(self, state):
        state_input = self.prepare_states(state)
        value = self.critic(inputs=state_input).numpy()
        if self.discrete_action_space:
            probabilities = self.actor(inputs=state_input)
            return probabilities, value
        else:
            mean, std = self.actor(inputs=state_input)
            return mean, std, value

    def select_action(self, state):
        if self.discrete_action_space:
            probabilities, value = self.predict_single(state)
            distribution = tfp.distributions.Categorical(probs=probabilities)
        else:
            mean, std, value = self.predict_single(state)
            distribution = tfp.distributions.Normal(mean, std)

        # print(distribution.sample())
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
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            self.transition += [reward]
            self.replay_buffer.store(*self.transition)
        else:
            next_state, reward, terminated, truncated, _ = self.test_env.step(action)

        return next_state, reward, terminated, truncated

    def train_actor(
        self, inputs, actions, log_probabilities, advantages, learning_rate
    ):
        # print("Training Actor")
        with tf.GradientTape() as tape:
            if self.discrete_action_space:
                distribution = tfp.distributions.Categorical(self.actor(inputs))
            else:
                mean, std = self.actor(inputs)
                distribution = tfp.distributions.Normal(mean, std)

            log_ratios = distribution.log_prob(actions) - log_probabilities

            probability_ratios = tf.exp(log_ratios)
            # min_advantages = tf.where(
            #     advantages > 0,
            #     (1 + self.clip_param) * advantages,
            #     (1 - self.clip_param) * advantages,
            # )

            clipped_probability_ratios = tf.clip_by_value(
                probability_ratios, 1 - self.clip_param, 1 + self.clip_param
            )
            # print(min_advantages, clipped_probability_ratios * advantages)

            actor_loss = tf.math.minimum(
                probability_ratios * advantages, clipped_probability_ratios * advantages
            )

            entropy_loss = distribution.entropy()
            actor_loss = -tf.reduce_mean(actor_loss) - (
                self.entropy_coefficient * entropy_loss
            )

        actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer(
            learning_rate=learning_rate,
            epsilon=self.actor_epsilon,
            clipnorm=self.actor_clipnorm,
        ).apply_gradients(
            grads_and_vars=zip(actor_gradients, self.actor.trainable_variables)
        )
        if self.discrete_action_space:
            kl_divergence = tf.reduce_mean(
                log_probabilities
                - tfp.distributions.Categorical(self.actor(inputs)).log_prob(actions)
            )
        else:
            mean, std = self.actor(inputs)
            kl_divergence = tf.reduce_mean(
                log_probabilities
                - tfp.distributions.Normal(mean, std).log_prob(actions)
            )
        kl_divergence = tf.reduce_sum(kl_divergence)
        # print(kl_divergence)
        # print()
        return kl_divergence

    def train_critic(self, inputs, returns, learning_rate):
        with tf.GradientTape() as tape:
            # print(returns)
            # print(self.critic(inputs, training=True))
            critic_loss = tf.reduce_mean(
                (returns - self.critic(inputs, training=True)) ** 2
            )
        # print(critic_loss)
        critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer(
            learning_rate=learning_rate,
            epsilon=self.critic_epsilon,
            clipnorm=self.critic_clipnorm,
        ).apply_gradients(
            grads_and_vars=zip(critic_gradients, self.critic.trainable_variables)
        )

        return critic_loss

    def action_mask(self, q, state, turn):
        q_copy = copy.deepcopy(q)
        for i in range(len(q_copy)):
            if not state.is_valid_move(i):
                if turn % 2 == 0:
                    q_copy[i] = float("-inf")
                else:
                    q_copy[i] = float("inf")
        return q_copy

    def train(self):
        self.is_test = False
        stat_score = (
            []
        )  # make these num trials divided by graph interval so i dont need to append (to make it faster?)
        stat_test_score = []
        stat_actor_loss = []
        stat_critic_loss = []
        num_trials_truncated = 0
        state, _ = self.env.reset()

        for epoch in range(self.num_epochs):
            num_episodes = 0
            total_score = 0
            score = 0

            for timestep in range(self.steps_per_epoch):
                action = self.select_action(state)
                next_state, reward, terminated, truncated = self.step(action)
                done = terminated or truncated
                state = next_state
                score += reward

                if done or timestep == self.steps_per_epoch - 1:
                    last_value = (
                        0 if done else self.critic(self.prepare_states(next_state))
                    )
                    self.replay_buffer.finish_trajectory(last_value)
                    num_episodes += 1
                    state, _ = self.env.reset()
                    if score >= self.env.spec.reward_threshold:
                        print("Your agent has achieved the env's reward threshold.")
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
            minibatch_size = len(observations) // self.num_minibatches

            for _ in range(self.train_policy_iterations):

                learning_rate = self.actor_learning_rate * (
                    1
                    - (
                        (((epoch * self.steps_per_epoch) + timestep) - 1)
                        / (self.num_epochs * self.steps_per_epoch)
                    )
                )
                # print(learning_rate)
                learning_rate = max(learning_rate, 0)
                # COULD BREAK UP INTO MINI BATCHES
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
                    stat_actor_loss.append(kl_divergence)
                    if kl_divergence > 1.5 * self.target_kl:
                        print("Early stopping at iteration {}".format(_))
                        break
                # kl_divergence = self.train_actor(
                #     inputs, actions, log_probabilities, advantages, learning_rate
                # )
                # stat_actor_loss.append(kl_divergence)
                # if kl_divergence > 1.5 * self.target_kl:
                #     print("Early stopping at iteration {}".format(_))
                #     break

            for _ in range(self.train_value_iterations):
                # COULD BREAK UP INTO MINI BATCHES
                learning_rate = self.actor_learning_rate * (
                    1
                    - (
                        (((epoch * self.steps_per_epoch) + timestep) - 1)
                        / (self.num_epochs * self.steps_per_epoch)
                    )
                )
                # print(learning_rate)
                learning_rate = max(learning_rate, 0)
                for start in range(0, len(observations), minibatch_size):
                    end = start + minibatch_size
                    batch_indices = indices[start:end]
                    batch_observations = inputs[batch_indices]
                    batch_returns = returns[batch_indices]
                    critic_loss = self.train_critic(
                        batch_observations, batch_returns, learning_rate
                    )
                    stat_critic_loss.append(critic_loss)
                # critic_loss = self.train_critic(inputs, returns, learning_rate)
                # stat_critic_loss.append(critic_loss)
                # stat_loss.append(critic_loss)

            # print("Done Training")
            # self.old_actor.set_weights(self.actor.get_weights())
            stat_score.append(total_score / num_episodes)
            stat_test_score.append(self.test())
            self.plot_graph(
                stat_score,
                stat_actor_loss,
                stat_critic_loss,
                stat_test_score,
                (epoch + 1) * self.steps_per_epoch,
            )
            self.save_checkpoint()

        self.plot_graph(
            stat_score,
            stat_actor_loss,
            stat_critic_loss,
            stat_test_score,
            self.num_epochs * self.steps_per_epoch,
        )
        self.save_checkpoint()
        self.env.close()
        return num_trials_truncated / self.num_epochs

    def plot_graph(self, score, actor_loss, critic_loss, test_score, step):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 20))
        ax1.plot(score, linestyle="solid")
        ax1.set_title("Frame {}. Score: {}".format(step, np.mean(score[-10:])))
        ax2.axhline(y=self.env.spec.reward_threshold, color="r", linestyle="-")
        ax2.set_title(
            "Frame {}. Test Score: {}".format(step, np.mean(test_score[-10:]))
        )
        ax2.plot(test_score, linestyle="solid")
        ax3.plot(actor_loss, linestyle="solid")
        ax3.axhline(y=self.target_kl, color="r", linestyle="-")
        ax3.set_title(
            "Frame {}. Actor Loss: {}".format(step, np.mean(actor_loss[-10:]))
        )
        ax4.plot(critic_loss, linestyle="solid")
        ax4.set_title(
            "Frame {}. Critic Loss: {}".format(step, np.mean(critic_loss[-10:]))
        )
        plt.savefig("./{}.png".format(self.model_name))
        plt.close(fig)

    def test(self, num_trials=100, video_folder="") -> None:
        """Test the agent."""
        self.is_test = True
        average_score = 0
        for trials in range(num_trials - 1):
            state, _ = self.test_env.reset()
            done = False
            score = 0

            while not done:
                action = self.select_action(state)
                next_state, reward, terminated, truncated = self.step(action)
                done = terminated or truncated
                state = next_state

                score += reward
            average_score += score
            print("score: ", score)

        if video_folder == "":
            video_folder = "./videos/{}".format(self.model_name)
        # for recording a video
        self.test_env = gym.wrappers.RecordVideo(self.test_env, video_folder)
        state, _ = self.test_env.reset()
        done = False
        score = 0

        while not done:
            action = self.select_action(state)
            next_state, reward, terminated, truncated = self.step(action)
            done = terminated or truncated
            state = next_state

            score += reward

        print("score: ", score)
        average_score += score
        self.test_env.close()

        # reset
        self.is_test = False
        average_score /= num_trials
        return average_score
