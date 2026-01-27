import torch
import torch.nn as nn
import sys
import os

# Add the project root to sys.path
sys.path.append(os.getcwd())

from agents.muzero import MuZeroAgent
from agent_configs.muzero_config import MuZeroConfig
from game_configs.tictactoe_config import TicTacToeConfig
from modules.world_models.muzero_world_model import MuzeroWorldModel
from losses.basic_losses import CategoricalCrossentropyLoss, MSELoss


def action_as_plane(action, num_actions, height, width):
    if isinstance(action, torch.Tensor):
        action = action.item()
    plane = torch.zeros(num_actions, height, width)
    plane[action, :, :] = 1.0
    return plane


def test_update_target_model():
    game_config = TicTacToeConfig()
    env = game_config.make_env()

    config_dict = {
        "world_model_cls": MuzeroWorldModel,
        "residual_layers": [(16, 3, 1)],
        "actor_conv_layers": [(8, 1, 1)],
        "critic_conv_layers": [(8, 1, 1)],
        "reward_conv_layers": [],
        "num_simulations": 2,
        "minibatch_size": 2,
        "min_replay_buffer_size": 2,
        "replay_buffer_size": 10,
        "unroll_steps": 2,
        "n_step": 3,
        "multi_process": False,
        "training_steps": 1,
        "games_per_generation": 1,
        "action_function": action_as_plane,
        "value_loss_function": MSELoss(),
        "reward_loss_function": MSELoss(),
        "policy_loss_function": CategoricalCrossentropyLoss(),
        "support_range": None,
        "compile": True,
    }

    config = MuZeroConfig(config_dict, game_config)

    agent = MuZeroAgent(env, config, name="verify_update", device="cpu")

    # Manually compile the model to simulate what happens in train()
    # We use a simple compile here
    agent.model = torch.compile(agent.model)

    # Change some weights in agent.model
    # We can access the base model via _orig_mod if we know it's compiled,
    # but state_dict() on the compiled model already has the prefixes.

    # Call update_target_model
    try:
        agent.update_target_model()
        print("update_target_model called successfully.")
    except Exception as e:
        print(f"update_target_model failed: {e}")
        import traceback

        traceback.print_exc()
        return

    # Check if target_model weights match model weights
    source_state_dict = agent.model.state_dict()
    target_state_dict = agent.target_model.state_dict()

    # Check that all target keys exist in cleaned source keys
    clean_source_keys = [
        k[10:] if k.startswith("_orig_mod.") else k for k in source_state_dict.keys()
    ]
    target_keys = list(target_state_dict.keys())

    if set(clean_source_keys) == set(target_keys):
        print("Keys match perfectly after stripping _orig_mod.")
    else:
        print("Keys mismatch!")
        print(f"Clean source extra: {set(clean_source_keys) - set(target_keys)}")
        print(f"Target extra: {set(target_keys) - set(clean_source_keys)}")
        # Some backends might add extra keys, but usually it should match the original model
        return

    # Check values
    for k, v in target_state_dict.items():
        source_key = "_orig_mod." + k if "_orig_mod." + k in source_state_dict else k
        if not torch.allclose(v, source_state_dict[source_key]):
            print(f"Value mismatch for key {k}")
            return

    print(
        "Verification successful: Weights correctly transferred from compiled model to raw target model."
    )


if __name__ == "__main__":
    test_update_target_model()
