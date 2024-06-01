import os
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym
import copy
import dill
from agent_configs import Config
import pickle

from utils import make_stack, normalize_images_, get_legal_moves, plot_graphs

# Every model should have:
# 1. A network
# 2. An optimizer
# 3. A loss function
# 4. A training method
#       this method should have training iterations, minibatches, and training steps
# 5. A step method
# 6. A select_action method
# 7. A predict method


class BaseAgent:
    def __init__(
        self,
        env: gym.Env,
        config: Config,
        name,
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
        self.model_name = name
        self.config = config
        self.device = device

        self.env = env
        # self.test_env = copy.deepcopy(env)
        if hasattr(self.env, "render_mode") and self.env.render_mode == "rgb_array":
            self.test_env = gym.wrappers.RecordVideo(
                copy.deepcopy(env),
                ".",
                name_prefix="{}".format(self.model_name),
            )
        else:
            self.test_env = copy.deepcopy(env)

        if isinstance(env.observation_space, gym.spaces.Box):
            self.observation_dimensions = env.observation_space.shape
        elif isinstance(env.observation_space, gym.spaces.Discrete):
            self.observation_dimensions = (env.observation_space.n,)
        else:
            raise ValueError("Observation space not supported")

        print("observation_dimensions: ", self.observation_dimensions)
        if isinstance(env.action_space, gym.spaces.Discrete):
            self.num_actions = env.action_space.n
            self.discrete_action_space = True
        else:
            self.num_actions = env.action_space.shape[0]
            self.discrete_action_space = False

        print("num_actions: ", self.num_actions)

        self.start_training_step = 0
        self.training_steps = self.config.training_steps
        self.checkpoint_interval = max(self.training_steps // 30, 1)

    def train(self):
        raise NotImplementedError

    def preprocess(self, states, device="cpu", dtype=torch.float32) -> torch.Tensor:
        """Applies necessary preprocessing steps to a batch of environment observations or a single environment observation
        Does not alter the input state parameter, instead creating a new Tensor on the inputted device (default cpu)

        Args:
            state (Any): A or a list of state returned from self.env.step
            dtype (dtype, optional): The datatype for the returned tensor
            device (str, optional): The device to send the preprocessed states to. Defaults to "cpu".

        Returns:
            Tensor: The preprocessed state, a tensor of floats. If the input was a single environment step,
                    the returned tensor is returned as outputed as if a batch of states with a length of a batch size of 1
        """

        # convert to np.array first for performance, recoommnded by pytorch
        prepared_state = torch.from_numpy(np.array(states)).to(dtype).to(device)
        if self.config.game.is_image:
            normalize_images_(prepared_state)
        if prepared_state.shape == self.observation_dimensions:
            prepared_state = make_stack(prepared_state)
        # print(prepared_state)
        return prepared_state

    def predict(self, state: torch.Tensor) -> torch.Tensor:
        """Run inference on 1 or a batch of environment states, applying necessary preprocessing steps

        Returns:
            Tensor: The predicted values, e.g. Q values for DQN or Q distributions for Categorical DQN
        """
        raise NotImplementedError

    def select_actions(self, predicted, legal_moves=None) -> torch.Tensor:
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

    def calculate_loss(self, batch):
        pass

    def learn(self):
        # raise NotImplementedError, "Every agent should have a learn method. (Previously experience_replay)"
        pass

    def load(self, dir, training_step):
        """load the model from a directory and training step. The name of the directory will be the name of the model, and should contain the following files:
        - episode_{training_step}_optimizer.dill
        - config.yaml
        """

        # dir = Path("model_weights", self.model_name)
        name = ""
        optimizer_path = Path(dir, f"episode_{training_step}_optimizer.dill"), "wb"
        config_path = Path(dir, f"episode_{training_step}_optimizer.dill"), "wb"
        weights_path = str(Path(dir, f"episode_{training_step}.keras"))

        self.config = self.config.__class__.load(config_path)
        with open(optimizer_path, "rb") as f:
            self.config.optimizer = dill.load(f)

        self.mode.load(weights_path)

        self.on_load()

    def load_replay_buffers(self, dir):
        with open(
            Path(
                dir,
                f"replay_buffers/replay_buffer.pkl",
            ),
            "rb",
        ) as f:
            self.replay_buffer = pickle.load(f)

    def load_model_weights(self, weights_path: str):
        raise NotImplementedError

    def load_from_checkpoint(self, dir: str, training_step):
        training_step_dir = Path(dir, f"step_{training_step}")
        # load the model weights
        weights_path = str(Path(training_step_dir, f"model_weights/weights.keras"))
        self.load_model_weights(weights_path)

        # load the config
        self.config = self.config.__class__.load(Path(dir, "configs/config.yaml"))

        # load optimizer (pickle doesn't work but dill does)
        with open(Path(training_step_dir, f"optimizers/optimizer.dill"), "rb") as f:
            self.config.optimizer = dill.load(f)

        # load replay buffer
        self.load_replay_buffers(training_step_dir)

        # load the graph stats and targets
        with open(Path(training_step_dir, f"graphs_stats/stats.pkl"), "rb") as f:
            self.stats = pickle.load(f)
        with open(Path(training_step_dir, f"graphs_stats/targets.pkl"), "rb") as f:
            self.targets = pickle.load(f)

        self.start_training_step = training_step

    def save_replay_buffers(self, dir):
        with open(
            Path(
                dir,
                f"replay_buffers/replay_buffer.pkl",
            ),
            "wb",
        ) as f:
            pickle.dump(self.replay_buffer, f)

    def save_checkpoint(
        self,
        num_trials,
        training_step,
        frames_seen,
        time_taken,
    ):
        # test model

        dir = Path("checkpoints", self.model_name)
        training_step_dir = Path(dir, f"step_{training_step}")
        os.makedirs(dir, exist_ok=True)
        os.makedirs(Path(dir, "graphs"), exist_ok=True)
        os.makedirs(Path(dir, "configs"), exist_ok=True)
        if self.config.save_intermediate_weights:
            weights_path = str(Path(training_step_dir, f"model_weights/weights.keras"))
            os.makedirs(Path(training_step_dir, "model_weights"), exist_ok=True)
            os.makedirs(Path(training_step_dir, "optimizers"), exist_ok=True)
            os.makedirs(Path(training_step_dir, "replay_buffers"), exist_ok=True)
            os.makedirs(Path(training_step_dir, "graphs_stats"), exist_ok=True)
            os.makedirs(Path(training_step_dir, "videos"), exist_ok=True)

            # save the model weights
            torch.save(self.model.state_dict(), weights_path)

            # save optimizer (pickle doesn't work but dill does)
            with open(Path(training_step_dir, f"optimizers/optimizer.dill"), "wb") as f:
                dill.dump(self.config.optimizer, f)

            # save replay buffer
            self.save_replay_buffers(training_step_dir)

        # save config
        self.config.dump(f"{dir}/configs/config.yaml")

        test_score = self.test(num_trials, training_step, training_step_dir)
        self.stats["test_score"].append(test_score)
        # save the graph stats and targets
        stats_path = Path(training_step_dir, f"graphs_stats", exist_ok=True)
        os.makedirs(stats_path, exist_ok=True)
        with open(Path(training_step_dir, f"graphs_stats/stats.pkl"), "wb") as f:
            pickle.dump(self.stats, f)
        with open(Path(training_step_dir, f"graphs_stats/targets.pkl"), "wb") as f:
            pickle.dump(self.targets, f)

        # plot the graphs (and save the graph)
        plot_graphs(
            self.stats,
            self.targets,
            training_step,
            frames_seen,
            time_taken,
            self.model_name,
            f"{dir}/graphs",
        )

    def test(self, num_trials, step, dir="./checkpoints") -> None:
        with torch.no_grad():
            """Test the agent."""
            average_score = 0
            max_score = float("-inf")
            min_score = float("inf")
            if self.test_env.render_mode == "rgb_array":
                self.test_env.episode_trigger = lambda x: (x + 1) % num_trials == 0
                self.test_env.video_folder = "{}/videos/{}/{}".format(
                    dir, self.model_name, step
                )
                if not os.path.exists(self.test_env.video_folder):
                    os.makedirs(self.test_env.video_folder)
            for trials in range(num_trials):
                state, info = self.test_env.reset()
                # self.test_env.render()
                legal_moves = get_legal_moves(info)

                done = False
                score = 0

                while not done:
                    prediction = self.predict(state)
                    action = self.select_actions(prediction, legal_moves).item()
                    next_state, reward, terminated, truncated, info = (
                        self.test_env.step(action)
                    )
                    # self.test_env.render()
                    done = terminated or truncated
                    legal_moves = get_legal_moves(info)
                    state = next_state
                    score += reward
                average_score += score
                max_score = max(max_score, score)
                min_score = min(min_score, score)
                print("score: ", score)

            # reset
            if self.test_env.render_mode != "rgb_array":
                self.test_env.render()
            self.test_env.close()
            average_score /= num_trials
            return {
                "score": average_score,
                "max_score": max_score,
                "min_score": min_score,
            }
