import gc
import sys
from time import time

from alphazero.alphazero_agent import AlphaZeroAgent
from muzero.muzero_minmax_stats import MinMaxStats

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
        # Memory growth must be set before GPUs have been initialized
        print(e)

import copy
import numpy as np
import tensorflow as tf
from replay_buffers.muzero_replay_buffer import ReplayBuffer, Game
import math
from muzero.muzero_mcts import Node
from muzero.muzero_network import Network
import matplotlib.pyplot as plt
import gymnasium as gym


class MuZeroAgent(AlphaZeroAgent):
    def __init__(self, env, config, name):
        super(MuZeroAgent, self).__init__(env, config, name)

    def step(self, action):
        if not self.is_test:
            next_state, reward, terminated, truncated, info = self.env.step(action)
        else:
            next_state, reward, terminated, truncated, info = self.test_env.step(action)

        return next_state, reward, terminated, truncated, info

    def train(self):
        training_time = time()
        total_environment_steps = 0
        stats = {
            "score": [],
            "policy_loss": [],
            "value_loss": [],
            "l2_loss": [],
            "reward_loss": [],
            "loss": [],
            "test_score": [],
        }
        targets = {
            "score": self.env.spec.reward_threshold,
            "value_loss": 0,
            "reward_loss": 0,
            "policy_loss": 0,
            "l2_loss": 0,
            "loss": 0,
            "test_score": self.env.spec.reward_threshold,
        }

        for training_step in range(self.training_steps):
            print("Training Step ", training_step + 1)
            for training_game in range(self.games_per_generation):
                score, num_steps = self.play_game()
                total_environment_steps += num_steps
                stats["score"].append(score)

            # STAT TRACKING
            for minibatch in range(self.config.num_minibatches):
                value_loss, reward_loss, policy_loss, l2_loss, loss = (
                    self.experience_replay()
                )
                stats["value_loss"].append(value_loss)
                stats["reward_loss"].append(reward_loss)
                stats["policy_loss"].append(policy_loss)
                stats["l2_loss"].append(l2_loss)
                stats["loss"].append(loss)

            # CHECKPOINTING
            if training_step % self.checkpoint_interval == 0 and training_step > 0:
                self.save_checkpoint(
                    stats,
                    targets,
                    5,
                    training_step,
                    total_environment_steps,
                    time() - training_time,
                )
        self.save_checkpoint(
            stats,
            targets,
            5,
            training_step,
            total_environment_steps,
            time() - training_time,
        )
        # save model to shared storage @Ezra

    def monte_carlo_tree_search(self, env, state, legal_moves, action_history):
        root = Node(0)
        value, policy, hidden_state = self.predict_single_initial_inference(state)
        to_play = state[0][0][2]
        root.expand(legal_moves, to_play, value, policy, hidden_state, reward)

        if not self.is_test:
            root.add_noise(
                self.config.root_dirichlet_alpha, self.config.root_exploration_fraction
            )

        min_max_stats = MinMaxStats(self.config.known_bounds)

        for _ in range(self.config.num_simulations):
            history = copy.deepcopy(action_history)
            node = root
            search_path = [node]

            # GO UNTIL A LEAF NODE IS REACHED
            while node.expanded():
                action, node = node.select_child(
                    min_max_stats, self.config.pb_c_base, self.config.pb_c_init
                )
                history.append(action)
                to_play = to_play + 1 % self.config.num_players
                search_path.append(node)

            # Turn of the leaf node
            parent = search_path[-2]
            reward, hidden_state, value, policy = (
                self.predict_single_recurrent_inference(
                    parent.hidden_state, history[-1]
                )
            )

            node.expand(to_play, self.num_actions, policy, hidden_state, reward)

            for node in search_path:
                node.value_sum += value if node.to_play != to_play else -value
                node.visits += 1
                min_max_stats.update(node.value())

                value = node.reward + self.config.discount_factor * value

        visit_counts = [
            (child.visits, action) for action, child in root.children.items()
        ]
        return root.value(), visit_counts

    def experience_replay(self):
        samples = self.replay_buffer.sample(
            self.config.unroll_steps, self.config.n_step
        )
        observations = samples["observations"]
        target_policies = samples["policy"]
        target_values = samples["values"]
        target_rewards = samples["rewards"]
        actions = samples["actions"]
        inputs = self.prepare_states(observations)
        for training_iteration in range(self.config.training_iterations):
            with tf.GradientTape() as tape:
                loss = 0
                for item in range(len(observations)):
                    value, policy, hidden_state = self.predict_single_initial_inference(
                        inputs[item]
                    )

                    # NORMALIZE POLICIES WITH ILLEGAL MOVES (DO WE DO THIS FOR MUZERO???)
                    # legal_moves_mask = (np.array(target_policies) > 0).astype(int)
                    # policies = tf.math.multiply(policies, legal_moves_mask)
                    # policies = tf.math.divide(
                    #     policies, tf.reduce_sum(policies, axis=1, keepdims=True)
                    # )

                    gradient_scales = [1.0]
                    values = [value]
                    policies = [policy]
                    rewards = [0]  # maybe this is from initial inference
                    for action in actions[item]:
                        reward, hidden_state, value, policy = (
                            self.predict_single_recurrent_inference(
                                hidden_state, action
                            )
                        )
                        gradient_scales.append(1.0 / len(actions[item]))
                        values.append(value)
                        policies.append(policy)
                        rewards.append(reward)

                        hidden_state = tf.scale_gradient(hidden_state, 0.5)

                    value_loss = (
                        self.config.value_loss_factor
                        * self.config.value_loss_function(target_values, values)
                    )
                    reward_loss = self.config.reward_loss_function(
                        target_rewards, rewards
                    )
                    policy_loss = self.config.policy_loss_function(
                        target_policies, policies
                    )
                    scaled_loss = tf.math.multiply(
                        value_loss + reward_loss + policy_loss, gradient_scales
                    )
                    print("Scaled Loss Shape ", scaled_loss.shape)
                    loss += tf.reduce_sum(scaled_loss)

                # compute losses
                loss = tf.divide(loss, self.config.replay_batch_size)
                l2_loss = sum(self.model.losses)
                loss += l2_loss
                # loss = tf.reduce_mean(loss)

            # print("Value Loss ", value_loss)
            # print("Policy Loss ", policy_loss)
            # print("L2 Loss ", l2_loss)
            # print("Loss ", loss)

            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.config.optimizer.apply_gradients(
                grads_and_vars=zip(gradients, self.model.trainable_variables)
            )
        # RIGHT NOW THIS RETURNS ONE BUT SHOULD PROBABLY RETURN A VALUE FOR EACH ITERATION
        return (
            tf.reduce_mean(value_loss),
            tf.reduce_mean(reward_loss),
            tf.reduce_mean(policy_loss),
            tf.reduce_mean(l2_loss),
            loss,
        )

    def predict_single_initial_inference(self, state):
        state_input = self.prepare_states(state)
        value, policy, hidden_state = self.model.initial_inference(state_input)
        policy = policy.numpy()[0]
        return value.numpy().item(), policy, hidden_state

    def predict_single_recurrent_inference(self, hidden_state, action):
        reward, hidden_state, value, policy = self.model.recurrent_inference(
            hidden_state, action
        )
        policy = policy.numpy()[0]
        value = value.numpy().item()
        return reward, hidden_state, value, policy

    def select_action(self, state, legal_moves=None, game=None):
        value, visit_counts = self.monte_carlo_tree_search(self.env, state, legal_moves)
        actions = [action for _, action in visit_counts]
        visit_counts = np.array([count for count, _ in visit_counts], dtype=np.float32)
        if (not self.is_test) and game.length < self.config.num_sampling_moves:
            temperature = self.config.exploration_temperature
        else:
            temperature = self.config.exploitation_temperature

        temperature_visit_counts = np.power(visit_counts, 1 / temperature)
        temperature_visit_counts /= np.sum(temperature_visit_counts)
        action = np.random.choice(actions, p=temperature_visit_counts)

        target_policy = np.zeros(self.num_actions)
        target_policy[actions] = visit_counts / np.sum(visit_counts)
        print("Target Policy", target_policy)
        if self.is_test:
            return action
        else:
            return action, target_policy, value

    def play_game(self):
        state, info = self.env.reset()
        game = Game()

        done = False
        while not done:
            action, target_policy = self.select_action(
                state,
                info["legal_moves"] if self.config.game.has_legal_moves else None,
                game=game,
            )
            print("Action ", action)
            next_state, reward, terminated, truncated, info = self.step(action)
            done = terminated or truncated
            game.append(state, reward, target_policy)
            state = next_state
            game.set_rewards()
            self.replay_buffer.store(game)
        return game.rewards[0], game.length
