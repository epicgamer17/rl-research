import ray
import torch
import torch.nn as nn
from agent_configs.muzero_config import MuZeroConfig
from agents.ray_workers import MuZeroWorker
from modules.agent_nets.muzero import Network
import unittest


class MockEnv:
    def __init__(self):
        import gymnasium as gym
        import numpy as np

        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(3, 32, 32), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(2)
        self.possible_agents = ["player_0"]
        self.agents = ["player_0"]
        self.metadata = {"render_modes": []}
        self.render_mode = None
        self.steps = 0

    def reset(self):
        self.steps = 0
        return torch.randn(3, 32, 32), {}

    def step(self, action):
        self.steps += 1
        return torch.randn(3, 32, 32), 1.0, self.steps >= 10, False, {}

    def close(self):
        pass

    def last(self):
        return torch.randn(3, 32, 32), 0, False, False, {}


class MockGame:
    def __init__(self):
        self.num_actions = 2
        self.num_players = 1
        self.is_discrete = True
        self.has_legal_moves = False

    def make_env(self, render_mode=None):
        return MockEnv()


from modules.world_models.muzero_world_model import MuzeroWorldModel


class TestRayWorker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ray.init(ignore_reinit_error=True)

    @classmethod
    def tearDownClass(cls):
        ray.shutdown()

    def test_worker_init_and_play(self):
        # 1. Create Config
        # MuZeroConfig requires config_dict containing 'world_model_cls' and game_config
        config = MuZeroConfig(
            config_dict={"world_model_cls": MuzeroWorldModel}, game_config={}
        )
        config.game = MockGame()
        # Ensure config matches Cartpole dims roughly or minimal
        config.observation_shape = (3, 32, 32)
        config.action_space_size = 2
        config.game.num_actions = 2

        # Speed up test
        config.num_simulations = 2
        config.minibatch_size = 1
        config.inference_batch_size = 1

        # 2. Instantiate Worker
        worker = MuZeroWorker.remote(
            config=config,
            worker_id=0,
            model_name="test_model",
            checkpoint_interval=10,
            device="cpu",
        )

        # 3. Create dummy weights
        # We need a model structure that matches
        local_model = Network(
            config=config,
            num_actions=2,
            input_shape=(1, 3, 32, 32),
            channel_first=True,
            world_model_cls=config.world_model_cls,
        )
        weights = local_model.state_dict()

        # 4. Set Weights
        ray.get(worker.set_weights.remote(weights, training_step=1))

        # 5. Play Game
        # This might fail if play_game logic is complex.
        # But we want to ensure basic connectivity
        result = ray.get(worker.continuous_self_play.remote())

        print("Worker Result:", result)
        self.assertIn("score", result)
        self.assertIn("game", result)


if __name__ == "__main__":
    unittest.main()
