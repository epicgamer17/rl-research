import sys
import os
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import time
from functools import reduce

# Add project root to path
sys.path.append(os.path.abspath("."))

from agents.muzero import MuZeroAgent
from agent_configs.muzero_config import MuZeroConfig
from modules.world_models.muzero_world_model import MuzeroWorldModel


# Setup Mocks (Same as verification)
class MockGame:
    def __init__(self):
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(0, 1, shape=(4,))
        self.num_players = 1
        self.is_discrete = True
        self.is_deterministic = True

    def make_env(self, render_mode=None):
        return MockEnv()


class MockSpec:
    def __init__(self):
        self.reward_threshold = 100


class MockEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(0, 1, shape=(4,))
        self.possible_agents = ["player_0"]
        self.agents = ["player_0"]
        self.spec = MockSpec()

    def reset(self, seed=None):
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(4, dtype=np.float32), 0, False, False, {}

    def close(self):
        pass


def benchmark_speedup():
    print("--- Setting up Benchmark ---")
    # Increase layer sizes to make computation significant
    # Dynamic Quantization overhead is high. Needs large matricies to win against hardware float acceleration.
    hidden_size = 2048

    base_config_dict = {
        "world_model_cls": MuzeroWorldModel,
        "observation_shape": (4,),
        "action_space_size": 4,
        "minibatch_size": 1,
        "use_mixed_precision": False,
        "multi_process": False,
        "residual_layers": [],
        "conv_layers": [],
        "dense_layer_widths": [hidden_size, hidden_size],
        "representation_residual_layers": [],
        "representation_conv_layers": [],
        "representation_dense_layer_widths": [hidden_size, hidden_size],
        "dynamics_residual_layers": [],
        "dynamics_conv_layers": [],
        "dynamics_dense_layer_widths": [hidden_size, hidden_size],
        "reward_conv_layers": [],
        "reward_dense_layer_widths": [hidden_size],
        "to_play_conv_layers": [],
        "to_play_dense_layer_widths": [hidden_size],
        "actor_conv_layers": [],
        "actor_dense_layer_widths": [hidden_size],
        "critic_conv_layers": [],
        "critic_dense_layer_widths": [hidden_size],
    }

    # 1. Instantiate Float Agent (Default: quantize=False)
    print("Instantiating Float Agent (quantize=False)...")
    config_float = MuZeroConfig(base_config_dict.copy(), {})
    config_float.game = MockGame()
    config_float.quantize = False

    agent_float = MuZeroAgent(
        env=MockEnv(), config=config_float, name="bench_float", device="cpu"
    )
    float_model = (
        agent_float.target_model
    )  # Use target model for fairness, though model is same structure if not quantized
    float_model.eval()

    # Verify it is float
    is_q = False
    for m in float_model.modules():
        if isinstance(
            m, (torch.nn.quantized.dynamic.Linear, torch.ao.nn.quantized.dynamic.Linear)
        ):
            is_q = True
    print(f"Float Agent has quantized layers: {is_q} (Expected: False)")

    # 2. Instantiate Quantized Agent (quantize=True)
    print("Instantiating Quantized Agent (quantize=True)...")
    config_quant = MuZeroConfig(base_config_dict.copy(), {})
    config_quant.game = MockGame()
    config_quant.quantize = True

    agent_quant = MuZeroAgent(
        env=MockEnv(), config=config_quant, name="bench_quant", device="cpu"
    )
    quantized_model = agent_quant.target_model
    quantized_model.eval()

    # Verify it is quantized
    is_q = False
    for m in quantized_model.modules():
        if isinstance(
            m, (torch.nn.quantized.dynamic.Linear, torch.ao.nn.quantized.dynamic.Linear)
        ):
            is_q = True
    print(f"Quantized Agent has quantized layers: {is_q} (Expected: True)")

    # Input data
    batch_sizes = [1, 32, 64, 128]
    num_iterations = 50

    print("\n--- Starting Benchmark Loops ---")

    for bs in batch_sizes:
        print(f"\nBatch Size: {bs}")
        # Create dummy observation input
        obs = torch.randn(bs, 4)

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                float_model.initial_inference(obs)
                quantized_model.initial_inference(obs)

        # Benchmark Float
        start_time = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_iterations):
                float_model.initial_inference(obs)

        end_time = time.perf_counter()
        float_duration = end_time - start_time
        float_avg = (float_duration / num_iterations) * 1000  # ms

        # Benchmark Quantized
        start_time = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_iterations):
                quantized_model.initial_inference(obs)
        end_time = time.perf_counter()
        quantized_duration = end_time - start_time
        quantized_avg = (quantized_duration / num_iterations) * 1000  # ms

        speedup = float_duration / quantized_duration

        print(f"Float Model Avg Time:     {float_avg:.4f} ms")
        print(f"Quantized Model Avg Time: {quantized_avg:.4f} ms")
        print(f"Speedup:                  {speedup:.2f}x")


if __name__ == "__main__":
    benchmark_speedup()
