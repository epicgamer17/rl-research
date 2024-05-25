import os
from pathlib import Path
from torch import Tensor
import gymnasium as gym
import copy
import dill
from agent_configs import Config

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
    def __init__(self, env: gym.Env, config: Config, name):
        self.model_name = name
        self.config = config

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

        self.training_steps = self.config.training_steps
        self.checkpoint_interval = self.training_steps // 30

        self.is_test = False

    def step(self, action):
        if not self.is_test:
            next_state, reward, terminated, truncated, info = self.env.step(action)
        else:
            next_state, reward, terminated, truncated, info = self.test_env.step(action)

        return next_state, reward, terminated, truncated, info

    def train(self):
        raise NotImplementedError

    def preprocess(self, state) -> Tensor:
        """Applies necessary preprocessing steps to a batch of environment observations or a single environment observation
        Does not alter the input state parameter, instead creating a new Tensor.
        If the input is a single environment step, the returned tensor is outputed as if B = 1

        Args:
            state (Any): The state returned from self.env.step

        Returns:
            Tensor: The preprocessed state, a tensor of floats
        """
        prepared_state = Tensor(state)
        if self.config.game.is_image:
            normalize_images_(prepared_state)
        if prepared_state.shape == self.observation_dimensions:
            prepared_state = make_stack(prepared_state)
        return prepared_state

    def predict_single(self, state: Tensor) -> Tensor:
        """Run inference on an environment state, applying necessary preprocessing steps
        and/or necessary postprocessing steps (e.g. masking) to the output of inference

        Args:
            state (Tensor): A preprocessed environment state

        Returns:
            Tensor: The predicted values, e.g. Q values for DQN
        """
        raise NotImplementedError

    def select_action(self, state, legal_moves=None) -> int:
        raise NotImplementedError

    def calculate_loss(self, batch):
        pass

    def learn(self):
        # raise NotImplementedError, "Every agent should have a learn method. (Previously experience_replay)"
        pass

    def collect_experience(self):
        # raise NotImplementedError
        pass

    def on_save(self):
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

    def on_load(self):
        pass

    def save_checkpoint(
        self,
        stats,
        targets,
        num_trials,
        training_step,
        frames_seen,
        time_taken,
    ):
        if self.config.save_intermediate_weights:
            dir = Path("checkpoints", self.model_name)
            os.makedirs(dir, exist_ok=True)
            os.makedirs(Path(dir, "model_weights"), exist_ok=True)
            os.makedirs(Path(dir, "optimizers"), exist_ok=True)
            os.makedirs(Path(dir, "configs"), exist_ok=True)
            os.makedirs(Path(dir, "replay_buffers"), exist_ok=True)
            os.makedirs(Path(dir, "graphs"), exist_ok=True)
            os.makedirs(Path(dir, "videos"), exist_ok=True)

            # save the model weights
            weights_path = str(Path(dir, f"model_weights/step_{training_step}.keras"))
            self.model.save(weights_path)

            # save optimizer (pickle doesn't work but dill does)
            with open(
                Path(dir, f"optimizers/step_{training_step}_optimizer.dill"), "wb"
            ) as f:
                dill.dump(self.config.optimizer, f)

            # save config
            self.config.dump(f"{dir}/configs/config.yaml")

            # save replay buffer
            with open(
                Path(dir, f"replay_buffers/step_{training_step}_replay_buffer.pkl"),
                "wb",
            ) as f:
                pickle.dump(self.replay_buffer, f)

        # test model
        test_score = self.test(num_trials, training_step, dir)
        stats["test_score"].append(test_score)
        # plot the graphs
        plot_graphs(
            stats,
            targets,
            training_step,
            frames_seen,
            time_taken,
            self.model_name,
            f"{dir}/graphs",
        )

    def test(self, num_trials, step, dir="./checkpoints") -> None:
        """Test the agent."""
        self.is_test = True
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
                action = self.select_action(state, legal_moves)
                next_state, reward, terminated, truncated, info = self.step(action)
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
        self.is_test = False
        average_score /= num_trials
        return {"score": average_score, "max_score": max_score, "min_score": min_score}
