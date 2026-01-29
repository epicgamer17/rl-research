import sys
import os
import torch
import gymnasium as gym

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from game_configs.tictactoe_config import TicTacToeConfig
from agent_configs.muzero_config import MuZeroConfig
from agents.muzero import MuZeroAgent
from modules.world_models.muzero_world_model import MuzeroWorldModel
import ray


def test_lstm_shape():
    print("Starting LSTM Shape Test...")

    # 1. Config with search_batch_size > 1
    game_config = TicTacToeConfig()
    params = {
        "multi_process": True,
        "num_workers": 1,
        "search_batch_size": 4,  # Key parameter to trigger batch logic
        "minibatch_size": 4,
        "training_steps": 10,
        "min_replay_buffer_size": 10,
        "replay_buffer_size": 100,
        "use_mixed_precision": False,  # Keep it simple
        "compile": False,
        "world_model_cls": MuzeroWorldModel,
        "lstm_horizon_len": 5,
        "value_prefix": True,
        "num_simulations": 8,  # Enough to trigger batched loop
    }

    config = MuZeroConfig(params, game_config)
    env = game_config.make_env()

    agent = None
    try:
        agent = MuZeroAgent(
            env=env,
            config=config,
            name="test_lstm_worker",
            device=torch.device("cpu"),
        )

        print("Agent initialized. Running training step...")

        # Run a bit of training to trigger workers and play_game
        # The workers start autonomously. We just need to wait a bit or call learn.
        # But we need to ensure play_game happens.

        import time

        t_start = time.time()
        while time.time() - t_start < 10:
            if agent.replay_buffer.get_size.remote():
                size = ray.get(agent.replay_buffer.get_size.remote())
                print(f"Buffer size: {size}")
                if size >= 4:
                    print("Data collected! LSTM shape seems correct.")
                    break
            time.sleep(1)

    except Exception as e:
        print(f"FAILED: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if agent:
            agent.shutdown()
        import ray

        ray.shutdown()


if __name__ == "__main__":
    import ray

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    test_lstm_shape()
