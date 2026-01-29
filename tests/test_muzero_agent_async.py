import pytest
import ray
import torch
import sys
import os
import numpy as np
import gymnasium as gym

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.muzero import MuZeroAgent, MuZeroWorker
from replay_buffers.modular_buffer import ModularReplayBuffer


from modules.world_models.muzero_world_model import MuzeroWorldModel


from agent_configs.muzero_config import MuZeroConfig


def test_muzero_async_initialization():
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    # 1. Setup Mock Config
    class MockGame:
        def __init__(self):
            self.num_players = 1
            self.observation_dimensions = (3, 64, 64)
            self.action_space_size = 5
            self.is_deterministic = True
            self.is_discrete = True
            self.has_legal_moves = False

        def make_env(self, render_mode=None):
            # Mock Env
            class MockEnv:
                def __init__(self):
                    # Use a mock object that mimics gym.spaces.Box
                    class MockSpace:
                        def __init__(self):
                            self.shape = (3, 64, 64)
                            self.dtype = np.float32

                    self.observation_space = MockSpace()
                    self.action_space = gym.spaces.Discrete(5)
                    self.possible_agents = ["player_0"]
                    self.spec = type("spec", (object,), {"reward_threshold": 100})()

                def reset(self):
                    return np.zeros((3, 64, 64), dtype=np.float32), {}

                def step(self, action):
                    return (
                        np.zeros((3, 64, 64), dtype=np.float32),
                        0.0,
                        False,
                        False,
                        {},
                    )

                def close(self):
                    pass

                def last(self):
                    return (
                        np.zeros((3, 64, 64), dtype=np.float32),
                        0.0,
                        False,
                        False,
                        {},
                    )

            return MockEnv()

    game = MockGame()

    config_dict = {
        "world_model_cls": MuzeroWorldModel,
        "multi_process": True,
        "num_workers": 1,
        "replay_buffer_size": 100,
        "minibatch_size": 4,
        "unroll_steps": 5,
        "n_step": 5,
        "discount_factor": 0.99,
        "learning_rate": 0.001,
        "num_minibatches": 1,
        "training_steps": 10,
        "action_embedding_dim": 16,
        "lstm_hidden_size": 16,
        "support_range": 10,
    }

    config = MuZeroConfig(config_dict, game)
    config.game = game  # Verify game is attached
    env = game.make_env()

    # 2. Instantiate Agent
    # This invokes __init__ which spawns workers
    try:
        agent = MuZeroAgent(env, config, name="test_agent", device="cpu")

        # Verify Actors exist
        assert hasattr(agent, "storage")
        assert agent.storage is not None
        assert hasattr(agent, "replay_buffer")
        assert agent.replay_buffer is not None
        assert hasattr(agent, "workers")
        assert len(agent.workers) == 1

        # We can't easily verify pass-through to workers without inspecting them, but if they spawned without error, it's good.

        # 3. Test Train Loop (Short run)
        # This is risky as it might hang if workers assume full functionality.
        # But we updated train() to just start workers and loop.
        # The workers calling run() might crash if play_game fails.
        # But let's verify agent.train() can be entered.
        # To avoid infinite loop, we can't call agent.train() directly unless we override training_steps to 0 or something.

        # But agent.train() has a while loop: while self.training_step < self.config.training_steps
        # We set training_steps = 10.
        # If workers crash, agent.train will sit there waiting for buffer size to increase.
        # So we shouldn't call it fully in this unit test unless we mock buffer fill.

        # Just verifying Initialization for now is a big win.
        print("Initialization passed.")

    except Exception as e:
        pytest.fail(f"Agent initialization failed: {e}")
    finally:
        if "agent" in locals():
            agent.shutdown()
        ray.shutdown()


if __name__ == "__main__":
    test_muzero_async_initialization()
