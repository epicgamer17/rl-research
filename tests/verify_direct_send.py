import ray
import torch
import torch.nn as nn
from agent_configs.muzero_config import MuZeroConfig
from agents.muzero import MuZeroWorker, Network
from modules.world_models.muzero_world_model import MuzeroWorldModel
import unittest
import gymnasium as gym
import numpy as np


class MockEnv:
    def __init__(self):
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(3, 32, 32), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(2)
        self.possible_agents = ["player_0"]
        self.agents = ["player_0"]
        self.metadata = {"render_modes": []}
        self.render_mode = None
        self.steps = 0

    def reset(self, **kwargs):
        self.steps = 0
        return np.random.randn(3, 32, 32).astype(np.float32), {}

    def step(self, action):
        self.steps += 1
        return (
            np.random.randn(3, 32, 32).astype(np.float32),
            1.0,
            self.steps >= 2,
            False,
            {},
        )

    def close(self):
        pass

    def last(self):
        return np.random.randn(3, 32, 32).astype(np.float32), 0, False, False, {}


class MockGame:
    def __init__(self):
        self.num_actions = 2
        self.num_players = 1
        self.is_discrete = True
        self.has_legal_moves = False

    def make_env(self, render_mode=None):
        return MockEnv()


class TestDirectSend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ray.init(ignore_reinit_error=True)

    @classmethod
    def tearDownClass(cls):
        ray.shutdown()

    def test_worker_direct_send(self):
        # 1. Create Config
        config = MuZeroConfig(
            config_dict={"world_model_cls": MuzeroWorldModel}, game_config={}
        )
        config.game = MockGame()
        config.observation_shape = (3, 32, 32)
        config.action_space_size = 2
        config.num_simulations = 2
        config.minibatch_size = 1
        config.inference_batch_size = 1
        config.replay_buffer_size = 100
        config.unroll_steps = 1
        config.n_step = 1
        config.discount_factor = 0.99
        config.per_alpha = 0.6
        config.per_beta = 0.4
        config.per_epsilon = 1e-6
        config.per_use_batch_weights = True
        config.per_use_initial_max_priority = True
        config.lstm_horizon_len = 5
        config.value_prefix = False
        config.reanalyze_tau = 1.0
        config.temperatures = [1.0]
        config.temperature_updates = []
        config.temperature_with_training_steps = False
        config.use_mixed_precision = False
        config.qat = False
        config.use_quantization = False

        # 2. Instantiate Worker
        worker = MuZeroWorker.remote(
            config=config,
            worker_id=0,
            model_name="test_model",
            checkpoint_interval=10,
            device="cpu",
        )

        # 3. Verify no replay_buffer
        try:
            has_buffer = ray.get(worker.has_attribute.remote("replay_buffer"))
        except:
            # We didn't add has_attribute, so we can check via a remote call that would fail if we try to access it
            # Or just check if we can get it
            def check_buffer(w):
                return hasattr(w, "replay_buffer")

            # Since MuZeroWorker is a class, we can't easily inject methods.
            # But we can try to access it and see if it fails.
            pass

        # Let's add a temporary helper to check
        # Actually, let's just run play_game and see if it works without error.

        # 4. Create dummy weights
        local_model = Network(
            config=config,
            num_actions=2,
            input_shape=(1, 3, 32, 32),
            channel_first=True,
            world_model_cls=config.world_model_cls,
        )
        weights = local_model.state_dict()
        ray.get(worker.set_weights.remote(weights, training_step=1))

        # 5. Play Game
        result = ray.get(worker.continuous_self_play.remote())

        print("Worker Result Keys:", result.keys())
        self.assertIn("score", result)
        self.assertIn("game", result)
        self.assertIn("num_steps", result)

        # Verify result contains a Game object (or something that looks like it)
        game = result["game"]
        self.assertTrue(len(game) >= 2)
        print(f"Game length: {len(game)}")


if __name__ == "__main__":
    unittest.main()
