import unittest
import torch
import ray
import numpy as np
import shutil
import gymnasium as gym

from agents.muzero import MuZeroAgent
from agent_configs.muzero_config import MuZeroConfig
from agents.muzero import MuZeroAgent
from agent_configs.muzero_config import MuZeroConfig
from modules.world_models.muzero_world_model import MuzeroWorldModel


class MockGameConfig:
    def __init__(self):
        self.num_players = 1
        self.is_deterministic = True
        self.is_discrete = True
        self.is_image = True
        self.reward_threshold = 100

    def make_env(self):
        return MockEnv()


class MockSpec:
    def __init__(self):
        self.reward_threshold = 100
        self.id = "MockEnv-v0"


class MockEnv(gym.Env):
    def __init__(self):
        self.spec = MockSpec()
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(3, 32, 32), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(4)
        self.steps = 0
        self.max_steps = 5

    def reset(self, seed=None, options=None):
        self.steps = 0
        return np.zeros((3, 32, 32), dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        obs = np.zeros((3, 32, 32), dtype=np.float32)
        reward = 1.0
        terminated = self.steps >= self.max_steps
        truncated = False
        return obs, reward, terminated, truncated, {}

    def render(self):
        pass

    def close(self):
        pass


class TestRayTrain(unittest.TestCase):
    def setUp(self):
        if ray.is_initialized():
            ray.shutdown()
        # Clean up any previous runs
        shutil.rmtree("./results/test_ray", ignore_errors=True)

    def tearDown(self):
        if ray.is_initialized():
            ray.shutdown()

    def test_train_loop(self):
        """
        Runs the training loop for a few steps to verify Ray integration.
        """

        # 0. Config
        # Mock Config Dict
        config_dict = {
            "model_name": "test_ray",
            "compile": False,
            "use_quantization": False,
            "qat": False,
            "use_mixed_precision": False,
            "multi_process": True,
            "num_workers": 2,
            "min_replay_buffer_size": 2,  # Trigger learning quickly
            "minibatch_size": 2,
            "training_steps": 5,  # Run very briefly
            "checkpoint_interval": 100,
            "games_per_generation": 100,  # Not used in async mode basically
            "transfer_interval": 2,
            "device": "cpu",
            "world_model_cls": MuzeroWorldModel,  # Required by logic
            "num_simulations": 5,  # Speed up test
            # Tiny model architecture for fast CPU testing
            "residual_layers": [(32, 3, 1)],
            "representation_residual_layers": [(32, 3, 1)],
            "dynamics_residual_layers": [(32, 3, 1)],
            "reward_conv_layers": [(16, 3, 1)],
            "reward_dense_layer_widths": [32],
            "critic_conv_layers": [(16, 3, 1)],
            "critic_dense_layer_widths": [32],
            "actor_conv_layers": [(16, 3, 1)],
            "actor_dense_layer_widths": [32],
        }

        game_config = MockGameConfig()
        config = MuZeroConfig(config_dict, game_config)

        # Disable storage of heavy objects if possible or just rely on MockEnv being small
        # config.action_space_size = 4

        # 1. Initialize Agent
        # Note: we need to patch standard libraries if they are missing in environment,
        # but here we assume imports work.

        agent = MuZeroAgent(
            env=MockEnv(), config=config, device="cpu", name="test_ray_train"
        )

        # 2. Run Train
        # This should launch workers, play a few games, learn, and exit after training_steps
        print("Starting training loop...")
        try:
            agent.train()
        except Exception as e:
            self.fail(f"Training loop failed with exception: {e}")

        print("Training loop finished.")

        # 3. Verify side effects
        # Verify stats were logged
        self.assertGreater(agent.stats.get_num_steps(), 0, "Should have logged steps")
        # Verify replay buffer has data
        self.assertGreater(
            agent.replay_buffer.size, 0, "Replay buffer should not be empty"
        )


if __name__ == "__main__":
    unittest.main()
