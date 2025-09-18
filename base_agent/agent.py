import gc
import os
from pathlib import Path
import random
import numpy as np
import pettingzoo
import torch
import gymnasium as gym
import copy

from tqdm import tqdm
from agent_configs import Config
import pickle
from torch.optim import Optimizer
from torch.nn import Module

from utils import (
    make_stack,
    plot_graphs,
    process_petting_zoo_obs,
    record_video_wrapper,
    get_legal_moves,
)

import time
from pynput import keyboard

# Every model should have:
# 1. A network
# 2. An optimizer
# 3. A loss function
# 4. A training method
#       this method should have training iterations, minibatches, and training steps
# 6. A select_action method
# 7. A predict method


class BaseAgent:
    def __init__(
        self,
        env,  # :gym.Env
        config: Config,
        name,
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
        if from_checkpoint:
            self.from_checkpoint = True
        self.model: Module = None
        self.optimizer: Optimizer = None
        self.model_name = name
        self.config = config
        self.device = device

        self.player_id = "player_0"
        self.training_time = 0
        self.training_step = 0
        self.total_environment_steps = 0
        self.training_steps = self.config.training_steps
        self.checkpoint_interval = max(self.training_steps // 30, 1)
        self.checkpoint_trials = 5

        self.env = env
        self.test_env = self.make_test_env(env)
        print("Test env:", self.test_env)
        self.observation_dimensions, self.observation_dtype = (
            self.determine_observation_dimensions(env)
        )

        print("Observation dimensions:", self.observation_dimensions)
        print("Observation dtype:", self.observation_dtype)

        if isinstance(env.action_space, gym.spaces.Discrete):
            self.num_actions = env.action_space.n
            self.discrete_action_space = True
        elif callable(env.action_space):
            self.num_actions = env.action_space(self.player_id).n
            self.discrete_action_space = True
        else:
            self.num_actions = env.action_space.shape[0]
            self.discrete_action_space = False

        print("num_actions: ", self.num_actions)

        self.stop_flag = False
        # Track which keys are pressed
        pressed_keys = set()

        def on_press(key):
            try:
                if key == keyboard.Key.shift:
                    pressed_keys.add("shift")
                elif key == keyboard.Key.esc:
                    pressed_keys.add("esc")

                # Check if both are pressed
                if "shift" in pressed_keys and "esc" in pressed_keys:
                    self.stop_flag = True
            except Exception as e:
                print(f"Error: {e}")

        def on_release(key):
            try:
                if key == keyboard.Key.shift:
                    pressed_keys.discard("shift")
                elif key == keyboard.Key.esc:
                    pressed_keys.discard("esc")
            except Exception as e:
                print(f"Error: {e}")

        # Run listener in the background
        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()

        # Set up the keyboard listener

        # TODO: NEED TO MAKE AN END OF TRAIN LOOP FUNCTION TO END THE LISTENER THREAD
        # listener.join()

    def make_test_env(self, env: gym.Env):
        print("making test env")
        # self.test_env = copy.deepcopy(env)
        if hasattr(env, "render_mode") and env.render_mode == "rgb_array":
            print("Test env with record video")
            if isinstance(env, gym.Env):
                print("gym env")
                return gym.wrappers.RecordVideo(
                    copy.deepcopy(env),
                    ".",
                    name_prefix="{}".format(self.model_name),
                )
            elif isinstance(env, pettingzoo.AECEnv) or isinstance(
                env, pettingzoo.ParallelEnv
            ):
                print("petting zoo env")
                return record_video_wrapper(copy.deepcopy(env), ".")
        else:
            print(
                "Warning: test_env will not record videos as render_mode is not 'rgb_array'"
            )
            return copy.deepcopy(env)

    def determine_observation_dimensions(self, env):
        print(type(env.observation_space))
        if isinstance(env.observation_space, gym.spaces.Box):
            return env.observation_space.shape, env.observation_space.dtype
        elif isinstance(env.observation_space, gym.spaces.Discrete):
            return (1,), np.int32
        elif isinstance(env.observation_space, gym.spaces.Tuple):
            return (
                len(env.observation_space.spaces),
            ), np.int32  # for tuple of discretes
        elif callable(env.observation_space):
            # Petting Zoo MARL
            # if isinstance(self.env.observation_space(self.player_id), Dict):
            try:
                print("petting zoo")
                if self.config.game.is_image:
                    # change from H, W, C to C, H, W
                    return (
                        env.observation_space(self.player_id)["observation"].shape[2],
                        env.observation_space(self.player_id)["observation"].shape[0],
                        env.observation_space(self.player_id)["observation"].shape[1],
                    ), env.observation_space(self.player_id)["observation"].dtype
                return (
                    env.observation_space(self.player_id)["observation"].shape,
                    env.observation_space(self.player_id)["observation"].dtype,
                )
            except KeyError:
                print("custom petting zoo")
                if self.config.game.is_image:
                    # change from H, W, C to C, H, W
                    return (
                        env.observation_space(self.player_id).shape[2],
                        env.observation_space(self.player_id).shape[0],
                        env.observation_space(self.player_id).shape[1],
                    ), env.observation_space(self.player_id).dtype
                return (
                    env.observation_space(self.player_id).shape,
                    env.observation_space(self.player_id).dtype,
                )
            except TypeError:
                if self.config.game.is_image:
                    # change from H, W, C to C, H, W
                    return (
                        env.observation_space(self.player_id).shape[2],
                        env.observation_space(self.player_id).shape[0],
                        env.observation_space(self.player_id).shape[1],
                    ), env.observation_space(self.player_id).dtype
                return (
                    env.observation_space(self.player_id).shape,
                    env.observation_space(self.player_id).dtype,
                )
        else:
            print("fallback case for observation dimension")
            return env.observation_space.shape, env.observation_space.dtype

    def train(self):
        if self.training_steps != 0:
            self.print_resume_training()

        pass

    def preprocess(self, states) -> torch.Tensor:
        """Applies necessary preprocessing steps to a batch of environment observations or a single environment observation
        Does not alter the input state parameter, instead creating a new Tensor on the inputted device (default cpu)

        Args:
            state (Any): A or a list of state returned from self.env.step
        Returns:
            Tensor: The preprocessed state, a tensor of floats. If the input was a single environment step,
                    the returned tensor is returned as outputed as if a batch of states with a length of a batch size of 1
        """

        # always convert to np.array first for performance, recoommnded by pytorchx
        # special case: list of compressed images (which are LazyFrames)
        # if isinstance(states[0], gym.wrappers.frame_stack.LazyFrames):
        #     np_states = np.array([np.array(state) for state in states])
        # else:
        # single observation, could be compressed or not compressed
        # print("Single state")
        # np_states = np.array(states)
        # FRAME STACK IS NO LONGER IN GYMNASIUM WRAPPERS

        np_states = np.array(states)

        # print("Numpyified States", np_states)
        prepared_state = (
            torch.from_numpy(
                np_states,
            )
            .to(torch.float32)
            .to(self.device)
        )
        # if self.config.game.is_image:
        # normalize_images(prepared_state)

        # if the state is a single number, add a dimension (not the batch dimension!, just wrapping it in []s basically)
        if prepared_state.shape == torch.Size([]):
            prepared_state = prepared_state.unsqueeze(0)

        if prepared_state.shape == self.observation_dimensions:
            prepared_state = make_stack(prepared_state)
        return prepared_state

    def predict(
        self, state: torch.Tensor, *args
    ) -> torch.Tensor:  # args is for info for player counts or legal move masks
        """Run inference on 1 or a batch of environment states, applying necessary preprocessing steps

        Returns:
            Tensor: The predicted values, e.g. Q values for DQN or Q distributions for Categorical DQN
        """
        raise NotImplementedError

    def select_actions(self, predicted, info) -> torch.Tensor:
        """Return actions determined from the model output, appling postprocessing steps such as masking beforehand

        Args:
            state (_type_): _description_
            legal_moves (_type_, optional): _description_. Defaults to None.

        Raises:
            NotImplementedError: _description_

        Returns:
            Tensor: _description_
        """
        raise NotImplementedError

    def learn(self):
        # raise NotImplementedError, "Every agent should have a learn method. (Previously experience_replay)"
        pass

    def load_optimizer_state(self, checkpoint):
        self.optimizer.load_state_dict(checkpoint["optimizer"])

    def load_replay_buffers(self, checkpoint):
        self.replay_buffer = checkpoint["replay_buffer"]

    def load_model_weights(self, checkpoint):
        self.model.load_state_dict(checkpoint["model"])

    def checkpoint_base(self, checkpoint):
        checkpoint["training_time"] = self.training_time
        checkpoint["training_step"] = self.training_step
        checkpoint["total_environment_steps"] = self.total_environment_steps
        return checkpoint

    def checkpoint_environment(self, checkpoint):
        checkpoint["enviroment"] = self.env
        return checkpoint

    def checkpoint_optimizer_state(self, checkpoint):
        checkpoint["optimizer"] = self.optimizer.state_dict()
        return checkpoint

    def checkpoint_replay_buffers(self, checkpoint):
        checkpoint["replay_buffer"] = self.replay_buffer
        return checkpoint

    def checkpoint_model_weights(self, checkpoint):
        checkpoint["model"] = self.model.state_dict()
        return checkpoint

    def checkpoint_extra(self, checkpoint) -> dict:
        return checkpoint

    @classmethod
    def load(cls, *args, **kwargs):
        cls.loaded_from_checkpoint = True
        return cls.load_from_checkpoint(*args, **kwargs)

    def load_from_checkpoint(agent_class, config_class, dir: str, training_step):
        # load the config and checkpoint
        training_step_dir = Path(dir, f"step_{training_step}")
        weights_dir = Path(training_step_dir, "model_weights")
        weights_path = str(Path(training_step_dir, f"model_weights/weights.keras"))
        config = config_class.load(Path(dir, "configs/config.yaml"))
        checkpoint = torch.load(weights_path)
        env = checkpoint["enviroment"]
        model_name = checkpoint["model_name"]

        # construct the agent
        agent = agent_class(env, config, model_name, from_checkpoint=True)

        # load the model state (weights, optimizer, replay buffer, training time, training step, total environment steps)
        os.makedirs(weights_dir, exist_ok=True)

        agent.training_time = checkpoint["training_time"]
        agent.training_step = checkpoint["training_step"]
        agent.total_environment_steps = checkpoint["total_environment_steps"]

        agent.load_model_weights(checkpoint)
        agent.load_optimizer_state(checkpoint)
        agent.load_replay_buffers(checkpoint)

        # load the graph stats and targets
        with open(Path(training_step_dir, f"graphs_stats/stats.pkl"), "rb") as f:
            agent.stats = pickle.load(f)
        with open(Path(training_step_dir, f"graphs_stats/targets.pkl"), "rb") as f:
            agent.targets = pickle.load(f)

        return agent

    def save_checkpoint(
        self,
        frames_seen=None,
        training_step=None,
        time_taken=None,
        save_weights=False,
    ):

        if not frames_seen is None:
            print(
                "warning: frames_seen option is deprecated, update self.total_environment_steps instead"
            )

        if not time_taken is None:
            print(
                "warning: time_taken option is deprecated, update self.training_time instead"
            )

        if not training_step is None:
            print(
                "warning: training_step option is deprecated, update self.training_step instead"
            )

        dir = Path("checkpoints", self.model_name)
        training_step_dir = Path(dir, f"step_{self.training_step}")
        os.makedirs(dir, exist_ok=True)

        # test model
        test_score = self.test(
            self.checkpoint_trials, step=self.training_step, dir=training_step_dir
        )
        self.stats["test_score"].append(test_score)

        # save the model state
        if save_weights:
            weights_path = str(Path(training_step_dir, f"model_weights/weights.keras"))
            os.makedirs(Path(training_step_dir, "model_weights"), exist_ok=True)
            checkpoint = self.make_checkpoint_dict()
            torch.save(checkpoint, weights_path)

        if self.env.render_mode == "rgb_array":
            os.makedirs(Path(training_step_dir, "videos"), exist_ok=True)

        # save config
        os.makedirs(Path(dir, "configs"), exist_ok=True)
        self.config.dump(f"{dir}/configs/config.yaml")
        os.makedirs(Path(dir, "stats"), exist_ok=True)
        pickle.dump(self.stats, open(f"{dir}/stats/stats.pkl", "wb"))

        # save the graph stats and targets
        os.makedirs(
            Path(training_step_dir, f"graphs_stats", exist_ok=True), exist_ok=True
        )
        with open(Path(training_step_dir, f"graphs_stats/stats.pkl"), "wb") as f:
            pickle.dump(self.stats, f)
        with open(Path(training_step_dir, f"graphs_stats/targets.pkl"), "wb") as f:
            pickle.dump(self.targets, f)

        # to periodically clear uneeded memory, if it is drastically slowing down training you can comment this out, checkpoint less often, or do less trials
        gc.collect()

        # plot the graphs (and save the graph)
        os.makedirs(Path(dir, "graphs"), exist_ok=True)
        plot_graphs(
            self.stats,
            self.targets,
            self.training_step if training_step is None else training_step,
            self.total_environment_steps if frames_seen is None else frames_seen,
            self.training_time if time_taken is None else time_taken,
            self.model_name,
            f"{dir}/graphs",
        )

    def make_checkpoint_dict(self):
        checkpoint = self.checkpoint_base({})
        checkpoint = self.checkpoint_environment(checkpoint)
        checkpoint = self.checkpoint_optimizer_state(checkpoint)
        checkpoint = self.checkpoint_replay_buffers(checkpoint)
        checkpoint = self.checkpoint_model_weights(checkpoint)
        checkpoint = self.checkpoint_extra(checkpoint)
        return checkpoint

    def test(self, num_trials, step, dir="./checkpoints") -> None:
        if num_trials == 0:
            return
        with torch.no_grad():
            """Test the agent."""
            average_score = 0
            max_score = float("-inf")
            min_score = float("inf")
            # self.test_env.reset()
            if self.test_env.render_mode == "rgb_array":
                self.test_env.episode_trigger = lambda x: (x + 1) % num_trials == 0
                self.test_env.video_folder = "{}/videos/{}/{}".format(
                    dir, self.model_name, step
                )
                if not os.path.exists(self.test_env.video_folder):
                    os.makedirs(self.test_env.video_folder)
            for trials in range(num_trials):
                state, info = self.test_env.reset()

                done = False
                score = 0

                while not done:
                    prediction = self.predict(
                        state,
                        info,
                    )
                    action = self.select_actions(prediction, info).item()
                    state, reward, terminated, truncated, info = self.test_env.step(
                        action
                    )
                    # self.test_env.render()
                    done = terminated or truncated
                    score += reward[0] if isinstance(reward, list) else reward
                average_score += score
                max_score = max(max_score, score)
                min_score = min(min_score, score)
                print("score: ", score)

            # reset
            # if self.test_env.render_mode != "rgb_array":
            #     self.test_env.render()
            self.test_env.close()
            average_score /= num_trials
            return {
                "score": average_score,
                "max_score": max_score,
                "min_score": min_score,
            }

    def print_training_progress(self):
        print(f"Training step: {self.training_step + 1}/{self.training_steps}")

    def print_resume_training(self):
        print(
            f"Resuming training at step {self.training_step + 1} / {self.training_steps}"
        )

    def print_stats(self):
        print(f"")


class MARLBaseAgent(BaseAgent):
    def __init__(
        self,
        env,
        model,
        name: str,
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
        super().__init__(
            env, model, name, device=device, from_checkpoint=from_checkpoint
        )

        self.test_agents = ["random"]
        print("Test agents:", self.test_agents)

    def test(self, num_trials, player=0, step=None, dir="./checkpoints") -> None:
        if num_trials == 0:
            return
        with torch.no_grad():
            """Test the agent."""
            average_score = 0
            max_score = float("-inf")
            min_score = float("inf")
            # self.test_env.reset()
            if self.test_env.render_mode == "rgb_array":
                self.test_env.episode_trigger = lambda x: (x + 1) % num_trials == 0
                self.test_env.video_folder = "{}/videos/{}/{}".format(
                    dir, self.model_name, step
                )
                if not os.path.exists(self.test_env.video_folder):
                    os.makedirs(self.test_env.video_folder)
            for trials in tqdm(range(num_trials)):
                self.test_env.reset()
                state, reward, terminated, truncated, info = self.test_env.last()
                agent_id = self.test_env.agent_selection
                current_player = self.test_env.agents.index(agent_id)
                state, info = process_petting_zoo_obs(state, info, current_player)
                done = terminated or truncated
                score = 0

                while not done:
                    prediction = self.predict(state, info, env=self.test_env.env)
                    action = self.select_actions(prediction, info).item()

                    self.test_env.step(action)
                    state, reward, terminated, truncated, info = self.test_env.last()
                    agent_id = self.test_env.agent_selection
                    current_player = self.test_env.agents.index(agent_id)
                    state, info = process_petting_zoo_obs(state, info, current_player)
                    done = terminated or truncated

                    if done:
                        score += self.test_env.rewards[f"player_{player}"]
                average_score += score
                max_score = max(max_score, score)
                min_score = min(min_score, score)

            # reset
            # if self.test_env.render_mode != "rgb_array":
            #     self.test_env.render()
            self.test_env.close()
            average_score /= num_trials
            print("average score:", average_score)
            return {
                "score": average_score,
                "max_score": max_score,
                "min_score": min_score,
            }

    def test_vs_agent(
        self, num_trials, agent, player=None, step=None, dir="./checkpoints"
    ):
        """
        Test the trained NFSP agent against a random agent
        """
        if agent is None:
            agent = "random"
        print("Testing Player {} vs Agent {}".format(player, agent))
        if self.test_env.render_mode == "rgb_array":
            self.test_env.episode_trigger = lambda x: (x + 1) % num_trials == 0
            self.test_env.video_folder = "{}/videos/random/{}/{}".format(
                dir, self.model_name, step
            )
            if not os.path.exists(self.test_env.video_folder):
                os.makedirs(self.test_env.video_folder)

        test_player_win_percentage = 0
        test_player_average_score = 0

        with torch.no_grad():  # No gradient computation during testing
            for trial in tqdm(range(num_trials)):
                # Reset environment
                self.test_env.reset()
                state, reward, termination, truncation, info = self.test_env.last()
                done = termination or truncation
                agent_id = self.test_env.agent_selection
                current_player = self.test_env.agents.index(agent_id)
                state, info = process_petting_zoo_obs(state, info, current_player)
                agent_names = self.test_env.agents.copy()

                episode_length = 0
                while not done and episode_length < 1000:  # Safety limit
                    # Get current agent and player
                    episode_length += 1

                    # Get action from average strategy
                    if current_player == player:
                        prediction = self.predict(state, info, env=self.test_env.env)
                        if trial == 0:
                            print(f"Player {current_player} Prediction: {prediction}")
                        action = self.select_actions(prediction, info).item()
                    else:
                        if agent == "random":
                            if "legal_moves" in info:
                                # print("legal moves found")
                                legal_moves = get_legal_moves(info)[0]

                                action = torch.multinomial(
                                    torch.ones_like(torch.tensor(legal_moves)).float()
                                    / len(legal_moves),
                                    1,
                                ).item()
                                action = legal_moves[action]
                            else:
                                action = self.test_env.action_space(agent_id).sample()
                        else:
                            prediction = self.predict(
                                state, info, env=self.test_env.env
                            )
                            action = self.select_actions(prediction, info).item()

                        if trial == 0:
                            print(f"Player {current_player} Random Action: {action}")

                    # Step environment
                    self.test_env.step(action)
                    state, reward, termination, truncation, info = self.test_env.last()
                    agent_id = self.test_env.agent_selection
                    current_player = self.test_env.agents.index(agent_id)
                    state, info = process_petting_zoo_obs(state, info, current_player)
                    done = termination or truncation

                episode_rewards = self.test_env.rewards
                if episode_rewards:
                    max_reward = max(episode_rewards.values())
                    winners = [
                        agent
                        for agent, reward in episode_rewards.items()
                        if reward == max_reward
                    ]
                    if len(winners) == 1:
                        winner = winners[0]
                        if agent_names.index(winner) == player:
                            test_player_win_percentage += 1
                        test_player_average_score += episode_rewards[
                            self.test_env.agents[player]
                        ]

        test_player_win_percentage /= num_trials
        test_player_average_score /= num_trials

        return test_player_win_percentage, test_player_average_score

    def save_checkpoint(
        self, frames_seen=None, training_step=None, time_taken=None, save_weights=False
    ):
        dir = Path("checkpoints", self.model_name)
        training_step_dir = Path(dir, f"step_{self.training_step}")
        os.makedirs(dir, exist_ok=True)

        # exploitability /= self.config.num_players
        for test_agent in self.test_agents:
            avg_score_vs_agent = 0
            for p in range(self.config.game.num_players):
                test_player_win_percentage, test_player_average_score = (
                    self.test_vs_agent(
                        self.checkpoint_trials,
                        test_agent,
                        player=p,
                        step=training_step,
                        dir=training_step_dir,
                    )
                )
                avg_score_vs_agent += test_player_average_score
            avg_score_vs_agent /= self.config.game.num_players
            self.stats["test_score_vs_{}".format(test_agent)].append(
                {"score": avg_score_vs_agent}
            )

        return super().save_checkpoint(
            frames_seen, training_step, time_taken, save_weights=save_weights
        )
