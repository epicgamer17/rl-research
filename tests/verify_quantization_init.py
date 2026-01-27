import torch
import torch.ao.quantization
import torch.nn as nn
from agents.muzero import MuZeroAgent
from agent_configs.muzero_config import MuZeroConfig
from game_configs.cartpole_config import CartPoleConfig
from modules.world_models.muzero_world_model import MuzeroWorldModel
from losses.basic_losses import CategoricalCrossentropyLoss


def action_as_onehot(action, num_actions):
    if isinstance(action, torch.Tensor):
        action = action.item()
    one_hot = torch.zeros(num_actions)
    one_hot[action] = 1.0
    return one_hot


def get_base_config_dict():
    return {
        "world_model_cls": MuzeroWorldModel,
        "residual_layers": [],
        "representation_dense_layer_widths": [16],
        "dynamics_dense_layer_widths": [16],
        "actor_dense_layer_widths": [16],
        "critic_dense_layer_widths": [16],
        "reward_dense_layer_widths": [16],
        "actor_conv_layers": [],
        "critic_conv_layers": [],
        "reward_conv_layers": [],
        "to_play_conv_layers": [],
        "num_simulations": 2,
        "minibatch_size": 2,
        "min_replay_buffer_size": 2,
        "replay_buffer_size": 10,
        "unroll_steps": 2,
        "n_step": 2,
        "multi_process": False,
        "training_steps": 1,
        "games_per_generation": 1,
        "learning_rate": 0.001,
        "action_function": action_as_onehot,
        "value_loss_function": CategoricalCrossentropyLoss(),
        "reward_loss_function": CategoricalCrossentropyLoss(),
        "policy_loss_function": CategoricalCrossentropyLoss(),
        "support_range": 31,
    }


def test_ptq_init():
    print("Testing PTQ Late Init...")
    game_config = CartPoleConfig()
    env = game_config.make_env()

    config_dict = get_base_config_dict()
    config_dict["use_quantization"] = True
    config_dict["qat"] = False

    config = MuZeroConfig(config_dict, game_config)
    agent = MuZeroAgent(env, config, name="test_ptq", device="cpu")

    # Verify NO observers in __init__
    for module in agent.model.modules():
        assert not hasattr(
            module, "activation_post_process"
        ), "Model should NOT have observers in __init__"

    # Simulate Late Init (PTQ)
    torch.backends.quantized.engine = "qnnpack"
    agent.model.qconfig = torch.ao.quantization.get_default_qconfig("qnnpack")
    agent.model.fuse_model()
    torch.ao.quantization.prepare(agent.model, inplace=True)

    # Verify Fusion/Preparation: Check if Observers are attached
    has_observer = False
    for name, module in agent.model.named_modules():
        if hasattr(module, "activation_post_process"):  # Observers
            has_observer = True
            print(f"Found observer in {name}")
            break
    assert has_observer, "Model should be prepared with observers after late init"
    print("PTQ late check passed!")


def test_qat_init():
    print("Testing QAT Late Init...")
    game_config = CartPoleConfig()
    env = game_config.make_env()

    config_dict = get_base_config_dict()
    config_dict["use_quantization"] = False
    config_dict["qat"] = True

    config = MuZeroConfig(config_dict, game_config)
    agent = MuZeroAgent(env, config, name="test_qat", device="cpu")

    # Verify NO observers in __init__
    for module in agent.model.modules():
        assert not hasattr(
            module, "activation_post_process"
        ), "Model should NOT have observers in __init__"

    # Simulate Late Init (QAT)
    torch.backends.quantized.engine = "qnnpack"
    agent.model.qconfig = torch.ao.quantization.get_default_qat_qconfig("qnnpack")
    agent.model.fuse_model()
    torch.ao.quantization.prepare_qat(agent.model, inplace=True)

    # QAT Preparation check
    has_fake_quant = False
    for name, module in agent.model.named_modules():
        if hasattr(module, "weight_fake_quant") or hasattr(
            module, "activation_post_process"
        ):
            if isinstance(
                getattr(module, "activation_post_process", None),
                torch.ao.quantization.FakeQuantize,
            ):
                has_fake_quant = True
                print(f"Found fake quant in {name}")
                break

    assert (
        has_fake_quant
    ), "Model should be prepared for QAT with fake quantization after late init"
    print("QAT late check passed!")


if __name__ == "__main__":
    try:
        test_ptq_init()
        test_qat_init()
        print("ALL TESTS PASSED")
    except Exception as e:
        import traceback

        traceback.print_exc()
        print("TEST FAILED")
