
import sys
import os
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from functools import reduce
import copy

# Add project root to path
sys.path.append(os.path.abspath("."))

from agents.muzero import MuZeroAgent
from agent_configs.muzero_config import MuZeroConfig
from modules.world_models.muzero_world_model import MuzeroWorldModel

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

def verify_quantization(enable_quantization=True):
    print(f"\n--- Verifying Quantization with quantize={enable_quantization} ---")
    config_dict = {
        "world_model_cls": MuzeroWorldModel,
        "observation_shape": (4,),
        "action_space_size": 4,
        "minibatch_size": 2,
        "use_mixed_precision": False,
        "multi_process": False, 
        "residual_layers": [],
        "conv_layers": [],
        "dense_layer_widths": [64],
        "representation_residual_layers": [],
        "representation_conv_layers": [],
        "representation_dense_layer_widths": [64],
        "dynamics_residual_layers": [],
        "dynamics_conv_layers": [],
        "dynamics_dense_layer_widths": [64],
        "reward_conv_layers": [],
        "to_play_conv_layers": [],
        "actor_conv_layers": [],
        "critic_conv_layers": [],
        "quantize": enable_quantization
    }
    
    config = MuZeroConfig(config_dict, {})
    config.game = MockGame()
    
    print("Instantiating MuZeroAgent...")
    agent = MuZeroAgent(
        env=MockEnv(),
        config=config,
        name="test_quant",
        device="cpu"
    )
    
    # Check target_model structure
    quantized_linear_count = 0
    float_linear_count = 0
    for name, mod in agent.target_model.named_modules():
        if isinstance(mod, (torch.nn.quantized.dynamic.Linear, torch.ao.nn.quantized.dynamic.Linear)):
            quantized_linear_count += 1
        elif isinstance(mod, nn.Linear):
            float_linear_count += 1
            
    print(f"Found {quantized_linear_count} quantized, {float_linear_count} float linear layers in target_model.")
    
    if enable_quantization:
        assert quantized_linear_count > 0, "target_model should have quantized layers when enabled!"
        assert float_linear_count == 0, "target_model should not have float linear layers when quantized!"
    else:
        assert quantized_linear_count == 0, "target_model should NOT have quantized layers when disabled!"
        assert float_linear_count > 0, "target_model should have float linear layers when disabled!"

    # Verify update works (no crashes) and basic correctness
    print("Testing update_target_model...")
    input_tensor = torch.randn(2, 4)
    
    with torch.no_grad():
        # Modify weights to ensure update happens
        # Use first linear layer found in model
        first_linear = None
        for mod in agent.model.modules():
            if isinstance(mod, nn.Linear):
                first_linear = mod
                break
        if first_linear:
             first_linear.weight.fill_(0.5)
             first_linear.bias.fill_(0.1)

    agent.update_target_model()
    
    # If quantized, verify forward pass logic
    if enable_quantization and first_linear:
         # Find corresponding target layer
         target_linear = None
         # We assume structure is preserved, so find via name if possible, or just first quantized linear
         for mod in agent.target_model.modules():
             if isinstance(mod, (torch.nn.quantized.dynamic.Linear, torch.ao.nn.quantized.dynamic.Linear)):
                 target_linear = mod
                 break
         
         if target_linear:
             in_features = target_linear.in_features
             batch_size = 2
             x = torch.randn(batch_size, in_features)
             y_quant = target_linear(x)
             
             weights_expected = torch.full((target_linear.out_features, in_features), 0.5)
             bias_expected = torch.full((target_linear.out_features,), 0.1)
             y_expected = torch.nn.functional.linear(x, weights_expected, bias_expected)
             
             diff = (y_quant - y_expected).abs().max()
             print(f"Max Output Diff: {diff.item()}")
             assert diff < 0.1, f"Quantized output too far from expected! Diff: {diff}"
             print("Quantization update verification passed.")

    print(f"Success for quantize={enable_quantization}")

if __name__ == "__main__":
    verify_quantization(enable_quantization=True)
    verify_quantization(enable_quantization=False)
