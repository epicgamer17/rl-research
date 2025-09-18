import datetime
from time import time
from agent_configs import AlphaZeroConfig
import torch
from utils import (
    clip_low_prob_actions,
    normalize_policies,
    action_mask,
    get_legal_moves,
    CategoricalCrossentropyLoss,
    MSELoss,
)
from torch.optim.sgd import SGD
from torch.optim.adam import Adam

import sys

sys.path.append("../")
from base_agent.agent import BaseAgent

import copy
import numpy as np
from replay_buffers.alphazero_replay_buffer import AlphaZeroReplayBuffer, Game
from alphazero.alphazero_mcts import Node
from alphazero.alphazero_network import Network
from torch.nn.utils import clip_grad_norm_


class AlphaZeroAgent(BaseAgent):
    def __init__(
        self,
        env,
        config: AlphaZeroConfig,
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
        from_checkpoint=False,
    ):
        super(AlphaZeroAgent, self).__init__(
            env, config, name, device=device, from_checkpoint=from_checkpoint
        )

        # Add learning rate scheduler

        self.model = Network(
            config=config,
            output_size=self.num_actions,
            input_shape=(self.config.minibatch_size,) + self.observation_dimensions,
        )

        self.model.to(device)

        self.replay_buffer = AlphaZeroReplayBuffer(
            self.config.replay_buffer_size, self.config.minibatch_size
        )

        if self.config.optimizer == Adam:
            self.optimizer: torch.optim.Optimizer = self.config.optimizer(
                params=self.model.parameters(),
                lr=self.config.learning_rate,
                eps=self.config.adam_epsilon,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == SGD:
            print("Warning: SGD does not use adam_epsilon param")
            self.optimizer: torch.optim.Optimizer = self.config.optimizer(
                params=self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
            )

        self.stats = {
            "score": [],
            "policy_loss": [],
            "value_loss": [],
            # "l2_loss": [],
            "loss": [],
            "test_score": [],
        }
        self.targets = {
            "score": self.env.spec.reward_threshold,
            "value_loss": 0,
            "policy_loss": 0,
            # "l2_loss": 0,
            "loss": 0,
            "test_score": self.env.spec.reward_threshold,
        }

    def train(self):
        super().train()
        start_time = self.training_time - time()
        if self.training_step == 0:
            self.print_resume_training()

        while self.training_step < self.config.training_steps:
            if self.training_step % self.config.print_interval == 0:
                self.print_training_progress()
            for training_game in range(self.config.games_per_generation):
                print("Training Game ", training_game + 1)
                score, num_steps = self.play_game()
                self.total_environment_steps += num_steps
                self.stats["score"].append({"score": score})  # score for player one

            # STAT TRACKING
            for minibatch in range(self.config.num_minibatches):
                value_loss, policy_loss, loss = self.learn()
                self.stats["value_loss"].append({"loss": value_loss})
                self.stats["policy_loss"].append({"loss": policy_loss})
                # self.stats["l2_loss"].append(l2_loss)
                self.stats["loss"].append({"loss": loss})
                print("Losses", value_loss, policy_loss, loss)

            # CHECKPOINTING
            if (
                self.training_step % self.checkpoint_interval == 0
                and self.training_step > 0
            ):
                self.training_time = time() - start_time
                self.save_checkpoint()
            self.training_step += 1

        self.training_time = time() - start_time
        self.save_checkpoint()
        # save model to shared storage @Ezra

    def monte_carlo_tree_search(self, env, state, info):
        root = Node(0, state, info)
        value, policy = self.predict_no_mcts(state, info)
        policy = policy[0]
        value = value[0][0]
        # print("Predicted Policy ", policy)
        # print("Predicted Value ", value)
        root.to_play = int(
            state[0][0][2]
        )  ## FRAME STACKING ADD A DIMENSION TO THE FRONT
        # print("Root Turn", root.to_play)
        root.expand(policy, env)

        if env == self.env:  # ghetto way of checking if we are in test mode
            root.add_noise(
                self.config.root_dirichlet_alpha, self.config.root_exploration_fraction
            )

        for _ in range(self.config.num_simulations):
            # print(_)
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

            # Turn of the leaf node
            leaf_node_turn = node.info["player"]  # info[]
            # print("Leaf Turn", leaf_node_turn)
            node.to_play = int(
                leaf_node_turn
            )  ## FRAME STACKING ADD A DIMENSION TO THE FRONT

            if terminated or truncated:
                value = reward[
                    leaf_node_turn
                ]  # o instead of leaf_node_turn do info["player"]
                # value of a leaf node is always negative for the current player
                # print(value)
            else:
                value, policy = self.predict_no_mcts(node.state, info)
                policy = policy[0]
                value = value[0][0]
                node.expand(policy, mcts_env)

            # UNCOMMENT FOR DEBUGGING
            for node in search_path:
                node.value_sum += value if node.to_play != leaf_node_turn else -value
                node.visits += 1

            mcts_env.close()
            del mcts_env
            del node
            del search_path

        visit_counts = [
            (child.visits, action) for action, child in root.children.items()
        ]

        del root
        return visit_counts

    def learn(self):
        samples = self.replay_buffer.sample()
        observations = samples["observations"]
        target_policies = samples["policies"]
        target_values = samples["rewards"]
        infos = samples["infos"]
        # print("LEARNING LEARNING LEARNING LEARNING LEARNING")
        # print("Observations", observations)
        # print("Target Policies", target_policies)
        # print("Target Values", target_values)
        # print("Infos", infos)
        inputs = self.preprocess(observations)
        for training_iteration in range(self.config.training_iterations):
            values, policies = self.predict_no_mcts(inputs, infos)
            # print("Values:", values)
            # print("Policies:", policies)
            # compute losses
            value_loss = self.config.value_loss_factor * MSELoss()(
                values, torch.Tensor(target_values).to(self.device)
            )
            policy_loss = CategoricalCrossentropyLoss()(
                policies, torch.Tensor(target_policies).to(self.device)
            )

            # l2_loss = sum(self.model.losses)
            # loss = (value_loss + policy_loss) + l2_loss
            loss = value_loss + policy_loss
            loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        if self.config.clipnorm > 0:
            clip_grad_norm_(self.model.parameters(), self.config.clipnorm)

        self.optimizer.step()
        return (
            value_loss.mean().detach().cpu().numpy(),
            policy_loss.mean().detach().cpu().numpy(),
            # l2_loss.mean(),
            loss.detach().cpu().numpy(),
        )

    def predict_no_mcts(self, state, info: dict = None):
        state_input = self.preprocess(state)
        value, policy = self.model(inputs=state_input)
        # print("Value in predict_no_mcts", value)
        # print("Policy in predict_no_mcts", policy)
        if "legal_moves" in info:
            legal_moves = get_legal_moves(info)
            # print("Legal Moves", legal_moves)
            policy = action_mask(policy, legal_moves, mask_value=0, device=self.device)
            # print("Masked Policy", policy)
            policy = clip_low_prob_actions(policy, self.config.clip_low_prob)
            # print("Clipped Policy", policy)
            policy = normalize_policies(policy)
            # print("Normalized Policy", policy)
        # distribution = torch.distributions.Categorical(probs=policy)
        return value, policy

    def predict(
        self, state, info: dict = None, env=None, temperature=1.0, *args, **kwargs
    ):
        visit_counts = self.monte_carlo_tree_search(env, state, info)
        actions = [action for _, action in visit_counts]
        visit_counts = np.array([count for count, _ in visit_counts], dtype=np.float32)

        temperature_visit_counts = np.power(visit_counts, 1 / temperature)
        temperature_visit_counts /= np.sum(temperature_visit_counts)

        target_policy = np.zeros(self.num_actions)
        target_policy[actions] = visit_counts / np.sum(visit_counts)
        # print("Target Policy", target_policy)

        # SHOULD TARGET POLICY BE TEMPERATURE VISIT COUNTS???
        return temperature_visit_counts, target_policy, actions

    def select_actions(self, predictions, *args, **kwargs):
        action = np.random.choice(predictions[2], p=predictions[0])
        return action

    def play_game(self):
        state, info = self.env.reset()
        game = Game(self.config.game.num_players)

        done = False
        while not done:

            # we are doing this here instead of in predict to make sure this is not done in test mode
            # although.... maybe it should be done in test mode? because the last moves should be exploitive?
            if info["step"] < self.config.num_sampling_moves:  # and (not self.is_test)
                temperature = self.config.exploration_temperature
            else:
                temperature = self.config.exploitation_temperature

            prediction = self.predict(
                state, info, env=self.env, temperature=temperature
            )
            print("Target Policy", prediction[1])
            print("Temperature Policy ", prediction[0])
            action = self.select_actions(prediction)
            print("Action ", action)
            next_state, reward, terminated, truncated, next_info = self.env.step(action)

            done = terminated or truncated
            game.append(state, reward, prediction[1], info=info)
            state = next_state
            info = next_info
        game.set_rewards()
        self.replay_buffer.store(game)
        return game.rewards[0], game.length
