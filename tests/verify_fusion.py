import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent_configs.muzero_config import MuZeroConfig
from modules.agent_nets.muzero import Network


# Mock the Config
class MockConfig:
    def __init__(self):
        self.conv_layers = [(16, 3, 2), (32, 3, 2)]
        self.residual_layers = [
            (16, 3, 1),
            (16, 3, 1),
        ]  # This creates 2 residual blocks
        self.dense_layer_widths = [16]
        self.input_shape = (3, 64, 64)
        self.num_actions = 4
        self.qat = True  # Enable quantization stubs
        self.game = type("obj", (object,), {"is_discrete": True, "num_players": 1})
        self.stochastic = False
        self.num_chance = 0
        self.activation = nn.ReLU()
        self.noisy_sigma = 0
        self.norm_type = "batch"
        self.use_efficientzero = False
        self.bn_momentum = 0.1
        self.value_support = None
        self.reward_support = None
        self.fp32_inference = False
        self.value_prefix = False
        self.action_embedding_dim = 16
        self.dynamics_state_dim = 16  # potential
        self.reward_dim = 1  # potential
        self.discount = 0.99
        self.max_grad_norm = 10
        self.momentum = 0.9
        self.lr_init = 0.01
        self.lr_decay_rate = 0.1
        self.lr_decay_steps = 1000
        self.weight_decay = 0.0001
        self.fc_reward_layers = [16]
        self.fc_value_layers = [16]
        self.fc_policy_layers = [16]
        self.support_range = None
        self.projector_hidden_dim = 64
        self.projector_output_dim = 64
        self.predictor_hidden_dim = 64
        self.predictor_output_dim = 64


def test_fusion():
    print("Initializing Network...")
    config = MockConfig()
    # Mock game specific stuff usually passed or handled in init
    # Network expects config, num_actions, input_shape.

    # We need to ensure residual.py and others are importable.
    # Provided enviroment should have pythonpath set or running from root.

    net = Network(config, num_actions=4, input_shape=(1, 3, 64, 64))
    net.eval()

    print("\n--- Before Fusion ---")
    # Check Conv2dStack structure
    conv_stack = net.prediction.net.conv_layers
    print("Conv2dStack Layer 0:", conv_stack._layers[0])

    # Check ResidualStack structure
    res_stack = net.prediction.net.residual_layers
    print("ResidualStack Layer 0:", res_stack._layers[0])

    # Check DenseStack structure
    dense_stack = net.prediction.net.dense_layers
    print("DenseStack Layer 0:", dense_stack._layers[0])

    print("\n--- Fusing Model ---")
    net.fuse_model()

    print("\n--- After Fusion ---")

    # Verify Conv2dStack fusion
    # Flattened Conv2dStack sequential should have fused module at index 0
    # Original: Sequential(Conv, Norm, Act) -> Fused: Sequential(ConvBnReLU, Identity, Identity)
    # OR depending on fusion implementation details in fuse_modules

    conv_layer_0 = conv_stack._layers[0]
    print("Conv2dStack Layer 0:", conv_layer_0)

    # Check if first element is an intrinsic fused module
    is_fused_conv = (
        hasattr(conv_layer_0[0], "qconfig")
        or "Intr" in str(type(conv_layer_0[0]))
        or "Fused" in str(type(conv_layer_0[0]))
        or isinstance(
            conv_layer_0[0],
            (torch.nn.intrinsic.ConvReLU2d, torch.nn.intrinsic.ConvBnReLU2d),
        )
    )
    # Note: torch.ao.quantization.fuse_modules replaces modules in place.
    # If fused, the type changes.

    if isinstance(conv_layer_0[0], torch.nn.intrinsic.ConvBnReLU2d) or isinstance(
        conv_layer_0[0], torch.nn.intrinsic.ConvReLU2d
    ):
        print("✅ Conv2dStack Fusion Successful")
    else:
        print("❌ Conv2dStack Fusion Failed")

    # Verify ResidualBlock fusion
    res_block = res_stack._layers[0]
    print("ResidualBlock:", res_block)

    if isinstance(res_block.conv1, torch.nn.intrinsic.ConvBnReLU2d) or isinstance(
        res_block.conv1, torch.nn.intrinsic.ConvReLU2d
    ):
        print("✅ ResidualBlock Conv1 Fusion Successful")
    else:
        print("❌ ResidualBlock Conv1 Fusion Failed")

    if isinstance(res_block.conv2, torch.nn.intrinsic.ConvBn2d) or isinstance(
        res_block.conv2, torch.nn.Conv2d
    ):  # Conv2 fused with BN only
        print("✅ ResidualBlock Conv2 Fusion Successful")
    else:
        print("❌ ResidualBlock Conv2 Fusion Failed")

    # Verify DenseStack fusion
    dense_layer_0 = dense_stack._layers[0]
    print("DenseStack Layer 0:", dense_layer_0)

    if isinstance(dense_layer_0[0], torch.nn.intrinsic.LinearReLU):
        print("✅ DenseStack Fusion Successful")
    else:
        print("❌ DenseStack Fusion Failed")

    # Run a forward pass to ensure no breakage
    print("\nRunning dummy forward pass...")
    try:
        x = torch.randn(1, 3, 64, 64)
        net.initial_inference(x)
        print("✅ Forward pass successful")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_fusion()
