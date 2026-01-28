import sys

sys.path.append(".")
import torch
from agent_configs.muzero_config import MuZeroConfig
from modules.world_models.muzero_world_model import Representation, Dynamics


def test_representation_init():
    # Mock config dict
    config_dict = {"world_model_cls": "MuzeroWorldModel", "game": {"is_discrete": True}}

    # Mock game config
    class MockGameConfig:
        is_discrete = True
        num_players = 2

    game_config = MockGameConfig()

    config = MuZeroConfig(config_dict, game_config)
    # Minimal config setup if needed
    input_shape = (1, 3, 96, 96)
    print("Initializing Representation...")
    rep = Representation(config, input_shape)

    assert hasattr(rep, "dequant"), "Representation missing dequant attribute"
    assert hasattr(rep, "out_quant"), "Representation missing out_quant attribute"
    print("Representation initialization check passed!")


def test_dynamics_init():
    # Mock config dict
    config_dict = {
        "world_model_cls": "MuzeroWorldModel",
        "value_prefix": True,
        "game": {"is_discrete": True},
    }

    # Mock game config
    class MockGameConfig:
        is_discrete = True
        num_players = 2

    game_config = MockGameConfig()

    config = MuZeroConfig(config_dict, game_config)
    input_shape = (1, 256, 6, 6)  # typical hidden state
    print("Initializing Dynamics...")
    # num_actions, action_embedding_dim
    dyn = Dynamics(config, input_shape, 18, 64)

    assert hasattr(dyn, "dequant"), "Dynamics missing dequant attribute"
    assert hasattr(dyn, "out_quant"), "Dynamics missing out_quant attribute"
    assert hasattr(dyn, "ff"), "Dynamics missing ff attribute"
    print("Dynamics initialization check passed!")


if __name__ == "__main__":
    try:
        test_representation_init()
        test_dynamics_init()
        print("\nAll checks passed successfully!")
    except Exception as e:
        print(f"\nFAILED: {e}")
        exit(1)
