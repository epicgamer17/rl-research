import gc
import os
import copy
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import dill as pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import pettingzoo

# Assuming these are custom modules provided in your project structure
from agents.random import RandomAgent
from stats.stats import StatTracker
from agent_configs import Config
from wrappers import record_video_wrapper, EpisodeTrigger


class BaseAgent:
    """
    Base Agent class handling generic RL training loops, checkpointing, and testing.
    """

    def __init__(
        self,
        env: gym.Env,
        config: Config,
        name: str,
        device: Optional[torch.device] = None,
        from_checkpoint: bool = False,
    ):
        self.from_checkpoint = from_checkpoint
        self.model_name = name
        self.config = config

        # 1. Device Setup
        if device:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")

        print(f"[{self.model_name}] Using device: {self.device}")

        # 2. Placeholders for Child Classes
        self.model: nn.Module = None
        self.optimizer: optim.Optimizer = None
        self.replay_buffer = None  # Placeholder

        # 3. Training Params
        self.player_id = "player_0"
        self.training_step = 0
        # Safety checks for config values
        total_steps = getattr(self.config, "training_steps", 1000)
        self.checkpoint_interval = max(total_steps // 30, 1)
        self.test_interval = max(total_steps // 30, 1)
        self.test_trials = 5

        # 4. Environment Setup
        self.env = env
        self.observation_dimensions, self.observation_dtype = (
            self.determine_observation_dimensions(env)
        )
        print(f"Observation dimensions: {self.observation_dimensions}")

        # 5. Action Space Setup
        self._setup_action_space(env)

        # 6. Test Environment
        self.test_env = self.make_test_env(env)

    def _setup_action_space(self, env):
        """Determines action space properties."""
        if isinstance(env.action_space, gym.spaces.Discrete):
            self.num_actions = int(env.action_space.n)
            self.discrete_action_space = True
        elif callable(env.action_space):  # PettingZoo
            self.num_actions = int(env.action_space(self.player_id).n)
            self.discrete_action_space = True
        else:  # Box/Continuous
            self.num_actions = int(env.action_space.shape[0])
            self.discrete_action_space = False

        print(
            f"Num actions: {self.num_actions} (Discrete: {self.discrete_action_space})"
        )

    def make_test_env(self, env: gym.Env):
        """Creates a separate environment for testing, handling video recording if applicable."""
        print("Making test env...")
        try:
            # Deepcopy can fail on certain C-based envs (MuJoCo, Atari)
            # Ideally, pass a factory function instead of an env instance to __init__
            env_copy = copy.deepcopy(env)
        except Exception as e:
            print(
                f"Warning: Could not deepcopy environment ({e}). Reusing reference (unsafe for concurrent access) or recreate manually."
            )
            env_copy = env

        render_mode = getattr(env, "render_mode", None)

        if render_mode == "rgb_array":
            print("Test env configured for video recording.")
            video_path = str(Path("checkpoints", self.model_name, "videos"))

            if isinstance(env, gym.Env):
                return gym.wrappers.RecordVideo(
                    env_copy,
                    video_folder=video_path,
                    name_prefix=f"{self.model_name}",
                    disable_logger=True,
                )
            elif isinstance(env, (pettingzoo.AECEnv, pettingzoo.ParallelEnv)):
                return record_video_wrapper(env_copy, video_path)

        return env_copy

    def determine_observation_dimensions(self, env):
        """Infers input dimensions for the neural network."""
        obs_space = env.observation_space

        # Handle PettingZoo callable observation spaces
        if callable(obs_space):
            obs_space = obs_space(self.player_id)

        if isinstance(obs_space, gym.spaces.Box):
            return obs_space.shape, obs_space.dtype
        elif isinstance(obs_space, gym.spaces.Discrete):
            return (1,), np.int32
        elif isinstance(obs_space, gym.spaces.Tuple):
            return (len(obs_space.spaces),), np.int32
        else:
            return obs_space.shape, obs_space.dtype

    def preprocess(self, states) -> torch.Tensor:
        """
        Converts states to torch tensors on the correct device.
        Adds batch dimension if input is a single observation.
        """
        # 1. Convert to numpy (efficiently)
        if torch.is_tensor(states):
            states = states.cpu().numpy()

        np_states = np.array(states, copy=False)

        # 2. Convert to Tensor
        prepared_state = torch.tensor(
            np_states, dtype=torch.float32, device=self.device
        )

        # 3. Handle Batch Dimensions
        # If it's a scalar (0-dim tensor), make it 1D
        if prepared_state.ndim == 0:
            prepared_state = prepared_state.unsqueeze(0)

        # If the shape matches the single observation shape exactly, implies missing batch dim
        # e.g., Obs is (4,), Tensor is (4,) -> make it (1, 4)
        if prepared_state.shape == torch.Size(self.observation_dimensions):
            prepared_state = prepared_state.unsqueeze(0)

        return prepared_state

    # --- Abstract Methods ---
    def train(self):
        raise NotImplementedError

    def predict(self, state: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def select_actions(self, prediction, info) -> torch.Tensor:
        raise NotImplementedError

    def learn(self):
        pass

    # --- Checkpointing ---
    def load_optimizer_state(self, checkpoint):
        if self.optimizer and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

    def load_replay_buffers(self, checkpoint):
        if "replay_buffer" in checkpoint:
            self.replay_buffer = checkpoint["replay_buffer"]

    def load_model_weights(self, checkpoint):
        if self.model and "model" in checkpoint:
            self.model.load_state_dict(checkpoint["model"])

    def make_checkpoint_dict(self):
        checkpoint = {
            "training_step": self.training_step,
            "model_name": self.model_name,
        }
        if self.optimizer:
            checkpoint["optimizer"] = self.optimizer.state_dict()
        if self.replay_buffer:
            checkpoint["replay_buffer"] = self.replay_buffer
        if self.model:
            checkpoint["model"] = self.model.state_dict()

        # NOTE: We specifically DO NOT save the environment.
        # It is not picklable in most robust use cases.
        return checkpoint

    def save_checkpoint(self, save_weights=False):
        base_dir = Path("checkpoints", self.model_name)
        step_dir = base_dir / f"step_{self.training_step}"
        os.makedirs(step_dir, exist_ok=True)

        if save_weights:
            weights_dir = step_dir / "model_weights"
            os.makedirs(weights_dir, exist_ok=True)
            weights_path = weights_dir / "weights.pt"
            checkpoint = self.make_checkpoint_dict()
            torch.save(checkpoint, weights_path)

        # Save Config
        config_dir = base_dir / "configs"
        os.makedirs(config_dir, exist_ok=True)
        if hasattr(self.config, "dump"):
            self.config.dump(f"{config_dir}/config.yaml")

        # Save Stats
        stats_dir = step_dir / "graphs_stats"
        os.makedirs(stats_dir, exist_ok=True)

        if hasattr(self, "stats"):
            with open(stats_dir / "stats.pkl", "wb") as f:
                pickle.dump(self.stats.get_data(), f)

            # Plot graphs
            graph_dir = base_dir / "graphs"
            os.makedirs(graph_dir, exist_ok=True)
            self.stats.plot_graphs(dir=graph_dir)

        gc.collect()

    @classmethod
    def load(cls, *args, **kwargs):
        """Alias for load_from_checkpoint."""
        return cls.load_from_checkpoint(*args, **kwargs)

    @classmethod
    def load_from_checkpoint(
        cls, env, agent_class, config_class, dir_path: str, training_step, device
    ):
        """
        Loads an agent from a checkpoint.
        IMPORTANT: 'env' must be passed fresh, it is not loaded from disk.
        """
        dir_path = Path(dir_path)
        step_dir = dir_path / f"step_{training_step}"
        weights_path = (
            step_dir / "model_weights/weights.pt"
        )  # Standardized extension to .pt
        config_path = dir_path / "configs/config.yaml"

        # 1. Load Config
        if not config_path.exists():
            print(
                f"Config not found at {config_path}, attempting to find in step dir..."
            )
            # Fallback logic if needed
        config = config_class.load(config_path)

        # 2. Load Checkpoint Data
        # weights_only=False is needed because we might be loading the replay buffer (arbitrary object)
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        model_name = checkpoint.get("model_name", "unnamed_model")

        # 3. Instantiate Agent
        agent = agent_class(
            env=env, config=config, name=model_name, device=device, from_checkpoint=True
        )

        agent.training_step = checkpoint.get("training_step", 0)
        agent.load_model_weights(checkpoint)
        agent.load_optimizer_state(checkpoint)
        agent.load_replay_buffers(checkpoint)

        # 4. Load Stats
        stats_path = step_dir / "graphs_stats/stats.pkl"
        if stats_path.exists():
            with open(stats_path, "rb") as f:
                agent.stats = pickle.load(f)
        else:
            print("Warning: No stats file found for checkpoint.")

        return agent

    def test(self, num_trials, dir="./checkpoints") -> dict:
        if num_trials == 0:
            return {}

        print(f"--- Starting Test: {self.model_name} ({num_trials} episodes) ---")

        with torch.no_grad():
            average_score = 0
            max_score = float("-inf")
            min_score = float("inf")

            # Setup video recording triggers if needed
            if hasattr(self.test_env, "episode_trigger"):
                # Ensure we record appropriately based on wrapper implementation
                pass

            for _ in range(num_trials):
                state, info = self.test_env.reset()
                done = False
                score = 0

                while not done:
                    # For gym envs, state is usually the obs
                    prediction = self.predict(state, info)
                    action = self.select_actions(prediction, info=info).item()

                    state, reward, terminated, truncated, info = self.test_env.step(
                        action
                    )
                    done = terminated or truncated

                    # Handle different reward structures (vector vs scalar)
                    r = reward[0] if isinstance(reward, (list, np.ndarray)) else reward
                    score += r

                average_score += score
                max_score = max(max_score, score)
                min_score = min(min_score, score)

            self.test_env.close()

            average_score /= num_trials
            print(f"Test Complete. Avg Score: {average_score:.2f}")

            return {
                "score": average_score,
                "max_score": max_score,
                "min_score": min_score,
            }

    def run_tests(self, stats):
        dir_path = Path("checkpoints", self.model_name)
        step_dir = dir_path / f"step_{self.training_step}"

        test_score = self.test(self.test_trials, dir=step_dir)

        if isinstance(test_score, float):
            stats.append("test_score", test_score)
        elif isinstance(test_score, dict):
            for key in test_score:
                stats.append("test_score", test_score[key], subkey=key)

    def __getstate__(self):
        state = self.__dict__.copy()
        # Don't pickle the environment or stats when saving the agent class instance
        if "env" in state:
            del state["env"]
        if "test_env" in state:
            del state["test_env"]
        if "stats" in state:
            del state["stats"]
        return state


class MARLBaseAgent(BaseAgent):
    def __init__(
        self,
        env,
        config: Config,  # Fixed type hint and argument order logic
        name: str,
        test_agents: List[Any] = None,
        device: Optional[torch.device] = None,
        from_checkpoint=False,
    ):
        if test_agents is None:
            test_agents = [RandomAgent()]

        # FIX: BaseAgent init expects (env, config, name...)
        # The previous code passed 'model' which likely didn't exist or was in wrong place
        super().__init__(
            env, config, name, device=device, from_checkpoint=from_checkpoint
        )
        self.test_agents = test_agents
        print(
            f"MARL Agent '{self.model_name}' initialized. Test agents: {[a.model_name for a in self.test_agents]}"
        )

    def test(self, num_trials, player=0, dir="./checkpoints") -> dict:
        # If single player game disguised as MARL, fallback
        if (
            hasattr(self.config.game, "num_players")
            and self.config.game.num_players == 1
        ):
            return super().test(num_trials, dir)

        if num_trials == 0:
            return {}

        with torch.no_grad():
            results = []

            # Setup video path logic (simplified)
            if hasattr(self.test_env, "video_folder"):
                self.test_env.video_folder = str(Path(dir) / "videos" / self.model_name)
                os.makedirs(self.test_env.video_folder, exist_ok=True)

            for trial in range(num_trials):
                self.test_env.reset()
                # Petting Zoo AEC Loop
                state, reward, terminated, truncated, info = self.test_env.last()
                done = terminated or truncated
                score = 0

                # We need to know which agent string maps to our player index
                # This assumes standard PettingZoo agents list ["player_0", "player_1"]
                target_agent_id = (
                    self.test_env.agents[player]
                    if player < len(self.test_env.agents)
                    else "player_0"
                )

                while not done:
                    # Determine action based on current agent selection
                    # In self-play test, we usually play all sides, or we play one side against random?
                    # This generic test method usually implies self-play or solo-play in MARL

                    prediction = self.predict(state, info, env=self.test_env.env)
                    action = self.select_actions(prediction, info=info).item()

                    self.test_env.step(action)

                    state, reward, terminated, truncated, info = self.test_env.last()
                    done = terminated or truncated

                    # Accumulate reward only for the specific player we are tracking
                    # Note: In AEC, rewards are often stored in a dictionary attribute
                    if hasattr(self.test_env, "rewards"):
                        score += self.test_env.rewards.get(target_agent_id, 0)
                    else:
                        # Fallback for simpler envs
                        score += reward

                results.append(score)

            self.test_env.close()

            if not results:
                return {"score": 0}

            average_score = sum(results) / len(results)
            return {
                "score": average_score,
                "max_score": max(results),
                "min_score": min(results),
            }

    def test_vs_agent(self, num_trials, opponent_agent, dir="./checkpoints"):
        """
        Test the trained agent against a specific opponent agent.
        Assumes a 2-player or multi-player setup where we rotate positions.
        """
        num_players = self.config.game.num_players
        final_rewards = {p: [] for p in range(num_players)}
        results = {}

        print(f"--- Testing: {self.model_name} vs {opponent_agent.model_name} ---")

        with torch.no_grad():
            # For each player position (0 and 1, usually)
            for player_idx in range(num_players):

                # Video setup
                if getattr(self.test_env, "render_mode", "") == "rgb_array":
                    video_folder = (
                        Path(dir) / "videos" / f"vs_{opponent_agent.model_name}"
                    )
                    os.makedirs(video_folder, exist_ok=True)
                    if hasattr(self.test_env, "video_folder"):
                        self.test_env.video_folder = str(video_folder)

                # Run trials for this configuration
                trials_per_config = max(1, num_trials // num_players)

                for _ in range(trials_per_config):
                    self.test_env.reset()

                    # AEC Loop
                    state, reward, termination, truncation, info = self.test_env.last()
                    done = termination or truncation

                    while not done:
                        agent_id = self.test_env.agent_selection
                        # Find index of current agent_id in the agents list
                        current_player_idx = self.test_env.agents.index(agent_id)

                        if current_player_idx == player_idx:
                            # It is OUR turn
                            prediction = self.predict(
                                state, info, env=self.test_env.env
                            )
                            action = self.select_actions(prediction, info=info).item()
                        else:
                            # It is OPPONENT'S turn
                            prediction = opponent_agent.predict(
                                state, info, env=self.test_env.env
                            )
                            action = opponent_agent.select_actions(
                                prediction, info=info
                            ).item()

                        self.test_env.step(action)
                        state, reward, termination, truncation, info = (
                            self.test_env.last()
                        )
                        done = termination or truncation

                    # End of Episode: Record rewards
                    for p_id in range(num_players):
                        agent_str = self.test_env.agents[p_id]
                        r = self.test_env.rewards.get(agent_str, 0)
                        final_rewards[p_id].append(r)

                # Calculate stats for this player configuration
                avg_score = sum(final_rewards[player_idx]) / len(
                    final_rewards[player_idx]
                )
                win_pct = sum(1 for r in final_rewards[player_idx] if r > 0) / len(
                    final_rewards[player_idx]
                )

                results[f"player_{player_idx}_score"] = avg_score
                results[f"player_{player_idx}_win%"] = win_pct

                print(
                    f"As Player {player_idx}: Win% {win_pct*100:.1f} | Avg Score {avg_score:.2f}"
                )

        # Aggregate overall score
        total_score = sum(results[f"player_{p}_score"] for p in range(num_players))
        results["score"] = total_score / num_players

        return results

    def run_tests(self, stats: StatTracker):
        # 1. Run generic tests (BaseAgent)
        # Note: We manually call the logic here because super().run_tests might not fit the vs_agent flow
        dir_path = Path("checkpoints", self.model_name)
        step_dir = dir_path / f"step_{self.training_step}"

        # 2. Test against specific opponents
        for test_agent in self.test_agents:
            results = self.test_vs_agent(self.test_trials, test_agent, dir=step_dir)

            # Log results
            for key, value in results.items():
                stats.append(
                    f"test_score_vs_{test_agent.model_name}",
                    value,
                    subkey=key,
                )
