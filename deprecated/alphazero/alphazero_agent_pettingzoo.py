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
import pettingzoo


class AlphaZeroAgentPettingZoo(BaseAgent):
    """
    AlphaZero agent modified to work with PettingZoo environments.
    
    Key modifications:
    1. Handles PettingZoo AECEnv and ParallelEnv environments
    2. Manages multi-agent turn-based gameplay
    3. Properly handles PettingZoo observation and action spaces
    4. Adapts reward handling for multi-agent scenarios
    """
    
    def __init__(
        self,
        env,
        config: AlphaZeroConfig,
        name=datetime.datetime.now().timestamp(),
        device: torch.device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        ),
        from_checkpoint=False,
    ):
        # Check if environment is PettingZoo
        self.is_pettingzoo = isinstance(env, pettingzoo.AECEnv) or isinstance(env, pettingzoo.ParallelEnv)
        
        if self.is_pettingzoo:
            # For PettingZoo environments, we need to handle multi-agent setup
            self.agents = env.possible_agents if hasattr(env, 'possible_agents') else env.agents
            self.agent_id = self.agents[0] if self.agents else None
            
        super(AlphaZeroAgentPettingZoo, self).__init__(
            env, config, name, device=device, from_checkpoint=from_checkpoint
        )

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
            "loss": [],
            "test_score": [],
        }
        
        # Handle reward threshold for PettingZoo environments
        if self.is_pettingzoo:
            self.targets = {
                "score": 1.0,  # Default target for PettingZoo games
                "value_loss": 0,
                "policy_loss": 0,
                "loss": 0,
                "test_score": 1.0,
            }
        else:
            self.targets = {
                "score": self.env.spec.reward_threshold,
                "value_loss": 0,
                "policy_loss": 0,
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
                self.stats["score"].append({"score": score})

            # STAT TRACKING
            for minibatch in range(self.config.num_minibatches):
                value_loss, policy_loss, loss = self.learn()
                self.stats["value_loss"].append({"loss": value_loss})
                self.stats["policy_loss"].append({"loss": policy_loss})
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

    def get_current_player(self, env):
        """Helper method to get current player from PettingZoo environment."""
        if hasattr(env, 'agent_selection'):
            current_agent = env.agent_selection
            if current_agent and current_agent in self.agents:
                return self.agents.index(current_agent)
        return 0

    def get_observation(self, env, agent=None):
        """Helper method to get observation from PettingZoo environment."""
        if hasattr(env, 'observe'):
            if agent is None:
                agent = env.agent_selection
            return env.observe(agent) if agent else None
        return None

    def monte_carlo_tree_search(self, env, state, info):
        root = Node(0, state, info)
        value, policy = self.predict_no_mcts(state, info)
        policy = policy[0]
        value = value[0][0]
        
        # Handle player turn for PettingZoo environments
        if self.is_pettingzoo:
            root.to_play = self.get_current_player(env)
        else:
            root.to_play = int(state[0][0][2])  # Original frame stacking logic
            
        root.expand(policy, env)

        if env == self.env:  # Check if we are in training mode
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
                
                # Handle environment step for PettingZoo vs regular Gym
                if self.is_pettingzoo:
                    if hasattr(mcts_env, 'step'):
                        mcts_env.step(action)
                        
                        # Check termination status
                        if hasattr(mcts_env, 'terminations') and hasattr(mcts_env, 'truncations'):
                            terminated = any(mcts_env.terminations.values()) if mcts_env.terminations else False
                            truncated = any(mcts_env.truncations.values()) if mcts_env.truncations else False
                            
                            # Get rewards
                            reward = mcts_env.rewards if hasattr(mcts_env, 'rewards') else {}
                            
                            # Get info
                            current_agent = mcts_env.agent_selection
                            info = mcts_env.infos.get(current_agent, {}) if hasattr(mcts_env, 'infos') and current_agent else {}
                        else:
                            # Fallback
                            terminated, truncated, reward, info = False, False, {}, {}
                    else:
                        # Fallback for different PettingZoo API versions
                        _, reward, terminated, truncated, info = mcts_env.step(action)
                else:
                    _, reward, terminated, truncated, info = mcts_env.step(action)
                    
                search_path.append(node)

            # Handle leaf node turn for PettingZoo
            if self.is_pettingzoo:
                leaf_node_turn = self.get_current_player(mcts_env)
            else:
                leaf_node_turn = node.info.get("player", 0)
                
            node.to_play = int(leaf_node_turn)

            if terminated or truncated:
                # Handle rewards for PettingZoo vs regular environments
                if self.is_pettingzoo and isinstance(reward, dict):
                    # PettingZoo returns rewards as dict
                    agent_name = self.agents[leaf_node_turn] if leaf_node_turn < len(self.agents) else self.agents[0]
                    value = reward.get(agent_name, 0)
                elif isinstance(reward, (list, np.ndarray)):
                    value = reward[leaf_node_turn] if leaf_node_turn < len(reward) else reward[0]
                else:
                    value = float(reward) if reward is not None else 0.0
            else:
                # Get new state for expansion
                if self.is_pettingzoo:
                    new_state = self.get_observation(mcts_env)
                    if new_state is not None:
                        node.state = new_state
                        
                value, policy = self.predict_no_mcts(node.state, info)
                policy = policy[0]
                value = value[0][0]
                node.expand(policy, mcts_env)

            # Backpropagate values
            for path_node in search_path:
                path_node.value_sum += value if path_node.to_play != leaf_node_turn else -value
                path_node.visits += 1

            if hasattr(mcts_env, 'close'):
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
        
        inputs = self.preprocess(observations)
        for training_iteration in range(self.config.training_iterations):
            values, policies = self.predict_no_mcts(inputs, infos)
            
            # Compute losses
            value_loss = self.config.value_loss_factor * MSELoss()(
                values, torch.Tensor(target_values).to(self.device)
            )
            policy_loss = CategoricalCrossentropyLoss()(
                policies, torch.Tensor(target_policies).to(self.device)
            )

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
            loss.detach().cpu().numpy(),
        )

    def predict_no_mcts(self, state, info: dict = None):
        state_input = self.preprocess(state)
        value, policy = self.model(inputs=state_input)
        
        if info and "legal_moves" in info:
            legal_moves = get_legal_moves(info)
            policy = action_mask(policy, legal_moves, mask_value=0, device=self.device)
            policy = clip_low_prob_actions(policy, self.config.clip_low_prob)
            policy = normalize_policies(policy)
            
        return value, policy

    def predict(
        self, state, info: dict = None, env=None, temperature=1.0, *args, **kwargs
    ):
        visit_counts = self.monte_carlo_tree_search(env, state, info)
        actions = [action for _, action in visit_counts]
        visit_counts = np.array([count for count, _ in visit_counts], dtype=np.float32)

        if len(visit_counts) == 0:
            # Fallback if no visit counts
            visit_counts = np.ones(1, dtype=np.float32)
            actions = [0]

        temperature_visit_counts = np.power(visit_counts, 1 / temperature)
        temperature_visit_counts /= np.sum(temperature_visit_counts)

        target_policy = np.zeros(self.num_actions)
        target_policy[actions] = visit_counts / np.sum(visit_counts)

        return temperature_visit_counts, target_policy, actions

    def select_actions(self, predictions, *args, **kwargs):
        if len(predictions[2]) == 0:
            return 0  # Fallback action
        action = np.random.choice(predictions[2], p=predictions[0])
        return action

    def play_game(self):
        # Handle environment reset for PettingZoo vs regular Gym
        if self.is_pettingzoo:
            self.env.reset()
            # For AEC environments, we need to handle the agent turn system
            current_agent = self.env.agent_selection
            if current_agent is None:
                return 0, 0  # Game ended immediately
                
            state = self.get_observation(self.env, current_agent)
            if state is None:
                return 0, 0
                
            info = self.env.infos.get(current_agent, {}) if hasattr(self.env, 'infos') else {}
        else:
            state, info = self.env.reset()
            
        game = Game(self.config.game.num_players)
        step_count = 0

        while True:
            # Handle turn-based logic for PettingZoo
            if self.is_pettingzoo:
                current_agent = self.env.agent_selection
                if current_agent is None:
                    break
                    
                state = self.get_observation(self.env, current_agent)
                if state is None:
                    break
                    
                info = self.env.infos.get(current_agent, {}) if hasattr(self.env, 'infos') else {}
                info["step"] = step_count
                
                # Check if game is already terminated
                if hasattr(self.env, 'terminations') and hasattr(self.env, 'truncations'):
                    if (current_agent in self.env.terminations and self.env.terminations[current_agent]) or \
                       (current_agent in self.env.truncations and self.env.truncations[current_agent]):
                        break
            
            # Temperature scheduling
            if info.get("step", step_count) < self.config.num_sampling_moves:
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
            
            # Handle environment step
            if self.is_pettingzoo:
                self.env.step(action)
                
                # Check if game is done
                if hasattr(self.env, 'terminations') and hasattr(self.env, 'truncations'):
                    terminated = any(self.env.terminations.values()) if self.env.terminations else False
                    truncated = any(self.env.truncations.values()) if self.env.truncations else False
                    
                    # Get rewards
                    reward = self.env.rewards if hasattr(self.env, 'rewards') else {}
                else:
                    # Fallback
                    terminated, truncated, reward = False, False, {}
                    
                # Check if we should continue
                if terminated or truncated:
                    game.append(state, reward, prediction[1], info=info)
                    break
                    
            else:
                next_state, reward, terminated, truncated, next_info = self.env.step(action)
                if terminated or truncated:
                    game.append(state, reward, prediction[1], info=info)
                    break
                state = next_state
                info = next_info

            game.append(state, reward, prediction[1], info=info)
            step_count += 1
            
            # Safety check to prevent infinite loops
            if step_count > 1000:
                print("Warning: Game exceeded 1000 steps, terminating")
                break
            
        game.set_rewards()
        
        # Return appropriate score based on environment type
        if self.is_pettingzoo:
            # For PettingZoo, return the reward for the first agent or average reward
            if isinstance(game.rewards, dict):
                score = list(game.rewards.values())[0] if game.rewards else 0
            elif isinstance(game.rewards, (list, np.ndarray)) and len(game.rewards) > 0:
                score = game.rewards[0]
            else:
                score = 0
        else:
            score = game.rewards[0] if isinstance(game.rewards, (list, np.ndarray)) else game.rewards
            
        self.replay_buffer.store(game)
        return score, game.length
