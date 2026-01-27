import torch
import torch.nn as nn
import torch.ao.nn.intrinsic as nni
from agent_configs.muzero_config import MuZeroConfig
from modules.agent_nets.muzero import Network
import traceback


class MockConfig:
    def __init__(self, norm_type="batch"):
        # Basic MuZero Config Setup
        self.game = type(
            "obj",
            (object,),
            {
                "is_discrete": True,
                "num_players": 1,
                "observation_shape": (3, 64, 64),
                "action_space_size": 4,
            },
        )
        self.observation_dimensions = (3, 64, 64)
        self.minibatch_size = 1
        self.world_model_cls = None  # Will rely on default if needed or import
        self.stochastic = False
        self.qat = True
        self.use_quantization = False
        self.num_chance = 0
        self.norm_type = norm_type

        # Network Architecture params
        self.resnet_transition = False
        self.max_grad_norm = 5
        self.momentum = 0.9
        self.weight_decay = 1e-4

        # Define layers for NetworkBlock (representation)
        # Using lists of tuples: (filters, kernel_size, stride)
        self.representation_residual_layers = [(16, 3, 1), (16, 3, 1)]
        self.representation_conv_layers = []  # Optional
        self.representation_dense_layer_widths = []

        # Define layers for Dynamics (dynamics)
        self.dynamics_residual_layers = [(16, 3, 1)]
        self.dynamics_conv_layers = []
        self.dynamics_dense_layer_widths = []

        self.blocks = 1  # Number of residual blocks (legacy?)
        self.channels = 16  # Number of channels
        self.reduced_channels_reward = 16
        self.reduced_channels_value = 16
        self.reduced_channels_policy = 16
        self.fc_reward_layers = [16]
        self.fc_value_layers = [16]
        self.fc_policy_layers = [16]
        self.downsample = False  # For simplicity
        self.latent_state_channels = 16

        # Add missing attributes that might be accessed by Network or its submodules
        self.value_support = None
        self.reward_support = None
        self.action_space_size = 4
        self.action_embedding_dim = 16
        self.dynamics_state_dim = 16
        self.reward_dim = 1
        self.discount = 0.99
        self.projector_hidden_dim = 64
        self.projector_output_dim = 64
        self.predictor_hidden_dim = 64
        self.predictor_output_dim = 64
        self.lr_init = 0.01
        self.lr_decay_rate = 0.1
        self.lr_decay_steps = 1000
        self.fp32_inference = False
        self.value_prefix = False
        self.support_range = None
        self.use_efficientzero = False
        self.bn_momentum = 0.1
        self.noisy_sigma = 0
        self.activation = torch.nn.ReLU()


def test_fusion_safe():
    print("========================================")
    print("TEST 1: norm_type='batch'")
    print("========================================")
    try:
        config_bn = MockConfig(norm_type="batch")
        # Re-import to avoid stale references if any
        from modules.agent_nets.muzero import Network

        net_bn = Network(config_bn, num_actions=4, input_shape=(1, 3, 64, 64))
        net_bn.eval()
        print(f"DEBUG: Net BN training mode: {net_bn.training}")

        # Check Pre-fusion
        if hasattr(net_bn.world_model.representation.net, "residual_layers"):
            res_stack = net_bn.world_model.representation.net.residual_layers
            if isinstance(res_stack, nn.Identity):
                print("⚠️  ResidualStack is Identity! Check config.")
            else:
                res_block = res_stack._layers[0]
                print(f"DEBUG: Pre-fusion Norm1 type: {type(res_block.norm1)}")
                print(f"DEBUG: Pre-fusion Conv1 type: {type(res_block.conv1)}")

        # Fuse
        print("Fusing model with BatchNorm...")
        net_bn.fuse_model()

        if hasattr(net_bn.world_model.representation.net, "residual_layers"):
            res_block = net_bn.world_model.representation.net.residual_layers._layers[0]
            print(f"Residual Block Conv1 type: {type(res_block.conv1)}")

            # Expecting ConvBnReLU or ConvReLU(eval)
            possible_types = (
                torch.nn.intrinsic.ConvBnReLU2d,
                nni.ConvBnReLU2d,
                torch.nn.intrinsic.ConvReLU2d,
                nni.ConvReLU2d,
            )
            if isinstance(res_block.conv1, possible_types):
                print("✅ SUCCESS: Conv1 fused (ConvBnReLU2d or ConvReLU2d)")
            else:
                print(f"❌ FAILURE: Conv1 is {type(res_block.conv1)}")
                raise AssertionError("Test 1 Failed")

        print("\n")

    except Exception:
        traceback.print_exc()

    print("========================================")
    print("TEST 2: norm_type='none' (Identity)")
    print("========================================")
    try:
        config_id = MockConfig(norm_type="none")

        net_id = Network(config_id, num_actions=4, input_shape=(1, 3, 64, 64))
        net_id.eval()
        print(f"DEBUG: Net ID training mode: {net_id.training}")

        # Check pre-fusion
        res_block_pre = net_id.world_model.representation.net.residual_layers._layers[0]
        print(f"Pre-fusion Norm1 type: {type(res_block_pre.norm1)}")
        if isinstance(res_block_pre.norm1, nn.Identity):
            print("Confirmed Norm1 is Identity.")

        # Fuse
        print("Fusing model with Identity Norm...")
        net_id.fuse_model()

        # Check a ResidualBlock
        res_block = net_id.world_model.representation.net.residual_layers._layers[0]
        print(f"Residual Block Conv1 type: {type(res_block.conv1)}")

        # Expecting ConvReLU (fused conv+act), NOT ConvBnReLU
        if isinstance(res_block.conv1, torch.nn.intrinsic.ConvReLU2d) or isinstance(
            res_block.conv1, nni.ConvReLU2d
        ):
            # Also check if norm1 is still there (Identity) but not fused into conv
            print("✅ SUCCESS: Conv1 became ConvReLU2d (skipped BN fusion)")
        elif isinstance(res_block.conv1, torch.nn.intrinsic.ConvBnReLU2d) or isinstance(
            res_block.conv1, nni.ConvBnReLU2d
        ):
            print(
                f"❌ FAILURE: Conv1 became ConvBnReLU2d (should not happen with Identity)"
            )
        else:
            print(
                f"⚠️  Conv1 is {type(res_block.conv1)}. Standard Conv2d means no fusion happened at all."
            )

        print("\nRunning dummy forward pass to ensure no runtime crashes...")
        x = torch.randn(1, 3, 64, 64)
        net_id.initial_inference(x)
        print("✅ Forward pass successful")

    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    test_fusion_safe()
