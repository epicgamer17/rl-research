import sys
import os
import torch
import torch.multiprocessing as mp

# Add project root to path
sys.path.append(os.path.abspath("."))

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


def reproduce():
    game_config = TicTacToeConfig()
    env = game_config.make_env()

    config_dict = {
        "world_model_cls": MuzeroWorldModel,
        "residual_layers": [],
        "actor_conv_layers": [],
        "critic_conv_layers": [],
        "reward_conv_layers": [],
        "num_simulations": 1,
        "minibatch_size": 1,
        "min_replay_buffer_size": 1,
        "replay_buffer_size": 10,
        "unroll_steps": 1,
        "n_step": 1,
        "multi_process": True,
        "num_workers": 1,
        "training_steps": 1,
        "games_per_generation": 1,
        "action_function": action_as_plane,
        "value_loss_function": MSELoss(),
        "reward_loss_function": MSELoss(),
        "policy_loss_function": CategoricalCrossentropyLoss(),
        "support_range": None,
    }

    config = MuZeroConfig(config_dict, game_config)

    # Use CPU for reproduction
    agent = MuZeroAgent(env, config, name="reproduce_error", device="cpu")

    # We want to catch the error from the worker process
    # The error should happen almost immediately when train() is called
    try:
        agent.train()
    except Exception as e:
        print(f"\nCaught expected exception: {e}")
        import traceback

        traceback.print_exc()
        return True

    print("\nNo exception caught. The bug might be gone or not reproduced.")
    return False


if __name__ == "__main__":
    # Ensure multiprocessing works in script
    mp.set_start_method("spawn", force=True)
    reproduce()
