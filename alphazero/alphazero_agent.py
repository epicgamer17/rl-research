import datetime
import gc
import sys
from time import time

from base_agent.agent import BaseAgent
from configs.agent_configs.alphazero_config import AlphaZeroConfig

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
from replay_buffers.alphazero_replay_buffer import ReplayBuffer, Game
import math
from alphazero.alphazero_mcts import Node
from alphazero.alphazero_network import Network
import matplotlib.pyplot as plt
import gymnasium as gym


class AlphaZeroAgent(BaseAgent):
    def __init__(
        self,
        env,
        config: AlphaZeroConfig,
        name=datetime.datetime.now().timestamp(),
    ):
        super(AlphaZeroAgent, self).__init__(env, config, name)

        # Add learning rate scheduler

        self.model = Network(config, self.observation_dimensions, self.num_actions)

        self.replay_buffer = ReplayBuffer(
            self.config.replay_buffer_size, self.config.minibatch_size
        )

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
            "loss": [],
            "test_score": [],
        }
        targets = {
            "score": self.env.spec.reward_threshold,
            "value_loss": 0,
            "policy_loss": 0,
            "l2_loss": 0,
            "loss": 0,
            "test_score": self.env.spec.reward_threshold,
        }

        for training_step in range(self.config.training_steps):
            print("Training Step ", training_step + 1)
            for training_game in range(self.config.games_per_generation):
                score, num_steps = self.play_game()
                total_environment_steps += num_steps
                stats["score"].append(score)  # score for player one

            # STAT TRACKING
            for minibatch in range(self.config.num_minibatches):
                value_loss, policy_loss, l2_loss, loss = self.experience_replay()
                stats["value_loss"].append(value_loss)
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

    def monte_carlo_tree_search(self, env, state, legal_moves):
        root = Node(0, state, legal_moves)
        value, policy = self.predict_single(state, legal_moves)
        print("Predicted Policy ", policy)
        print("Predicted Value ", value)
        root.to_play = int(
            state[0][0][2]
        )  ## FRAME STACKING ADD A DIMENSION TO THE FRONT
        # print("Root Turn", root.to_play)
        root.expand(policy, env)

        if not self.is_test:
            root.add_noise(
                self.config.root_dirichlet_alpha, self.config.root_exploration_fraction
            )

        for _ in range(self.config.num_simulations):
            node = root
            mcts_env = copy.deepcopy(env)
            search_path = [node]

            # GO UNTIL A LEAF NODE IS REACHED
            while node.expanded():
                action, node = node.select_child(
                    self.config.pb_c_base, self.config.pb_c_init
                )
                _, reward, terminated, truncated, info = mcts_env.step(action)
                search_path.append(node)
                legal_moves = (
                    info["legal_moves"] if self.config.game.has_legal_moves else None
                )

            # Turn of the leaf node
            leaf_node_turn = node.state[0][0][2]
            # print("Leaf Turn", leaf_node_turn)
            node.to_play = int(
                leaf_node_turn
            )  ## FRAME STACKING ADD A DIMENSION TO THE FRONT

            if terminated or truncated:
                value = -reward
            else:
                value, policy = self.predict_single(node.state, legal_moves)
                node.expand(policy, mcts_env)

            # UNCOMMENT FOR DEBUGGING
            for node in search_path:
                node.value_sum += value if node.to_play != leaf_node_turn else -value
                node.visits += 1

        visit_counts = [
            (child.visits, action) for action, child in root.children.items()
        ]
        return visit_counts

    def experience_replay(self):
        samples = self.replay_buffer.sample()
        observations = samples["observations"]
        target_policies = samples["policy"]
        target_values = samples["rewards"]
        inputs = self.prepare_states(observations)
        for training_iteration in range(self.config.training_iterations):
            with tf.GradientTape() as tape:
                values, policies = self.model(inputs)
                # Set illegal moves probability to zero and renormalize
                legal_moves_mask = (np.array(target_policies) > 0).astype(int)
                policies = tf.math.multiply(policies, legal_moves_mask)
                policies = tf.math.divide(
                    policies, tf.reduce_sum(policies, axis=1, keepdims=True)
                )

                # compute losses
                value_loss = self.config.value_loss_factor * tf.losses.MSE(
                    target_values, values
                )
                policy_loss = tf.losses.categorical_crossentropy(
                    target_policies, policies
                )
                l2_loss = sum(self.model.losses)
                loss = (value_loss + policy_loss) + l2_loss
                loss = tf.reduce_mean(loss)

            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.config.optimizer(
                learning_rate=self.config.learning_rate,
                epsilon=self.config.adam_epsilon,
                clipnorm=self.config.clipnorm,
            ).apply_gradients(
                grads_and_vars=zip(gradients, self.model.trainable_variables)
            )
        # RIGHT NOW THIS RETURNS THE LAST ITERATION OF THE LOSSES BUT SHOULD RETURN ONE FOR EACH ITERATION
        return (
            tf.reduce_mean(value_loss),
            tf.reduce_mean(policy_loss),
            tf.reduce_mean(l2_loss),
            loss,
        )

    def predict_single(self, state, legal_moves=None):
        state_input = self.prepare_states(state)
        value, policy = self.model(inputs=state_input)
        policy = policy.numpy()[0]
        # Set illegal moves probability to zero and renormalize
        if self.config.game.has_legal_moves:
            policy = self.action_mask(legal_moves, policy)
        value = value.numpy().item()
        return value, policy

    def select_action(self, state, legal_moves=None, game=None):
        visit_counts = self.monte_carlo_tree_search(self.env, state, legal_moves)
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
            return action, target_policy

    def action_mask(self, legal_moves, policy):
        illegal_moves = [a for a in range(self.num_actions) if a not in legal_moves]
        policy[illegal_moves] = 0
        policy /= np.sum(policy)
        return policy

    def play_game(self):
        state, info = self.env.reset()
        game = Game()
        legal_moves = info["legal_moves"] if self.config.game.has_legal_moves else None

        done = False
        while not done:
            action, target_policy = self.select_action(state, legal_moves, game=game)
            print("Action ", action)
            next_state, reward, terminated, truncated, info = self.step(action)
            done = terminated or truncated
            legal_moves = (
                info["legal_moves"] if self.config.game.has_legal_moves else None
            )
            game.append(state, reward, target_policy)
            state = next_state
        game.set_rewards()
        self.replay_buffer.store(game)
        return game.rewards[0], game.length
