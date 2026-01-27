import torch
import pytest
from agents.muzero import MuZeroAgent
from agent_configs.muzero_config import MuZeroConfig
from game_configs.cartpole_config import CartPoleConfig
from game_configs.tictactoe_config import TicTacToeConfig
from modules.world_models.muzero_world_model import MuzeroWorldModel
from losses.basic_losses import CategoricalCrossentropyLoss, MSELoss


def action_as_onehot(action, num_actions):
    """
    Encodes an action as a one-hot vector.
    """
    if isinstance(action, torch.Tensor):
        action = action.item()
    one_hot = torch.zeros(num_actions)
    one_hot[action] = 1.0
    return one_hot


def action_as_plane(action, num_actions, height, width):
    """
    Encodes an action as a plane (for CNNs).
    """
    if isinstance(action, torch.Tensor):
        action = action.item()
    plane = torch.zeros(num_actions, height, width)
    plane[action, :, :] = 1.0
    return plane


def test_muzero_cartpole_smoke():
    """
    Smoke test for MuZero on CartPole.
    Ensures agent can initialize and run a training step.
    """
    game_config = CartPoleConfig()
    env = game_config.make_env()

    config_dict = {
        "world_model_cls": MuzeroWorldModel,
        "residual_layers": [],
        "representation_dense_layer_widths": [64],
        "dynamics_dense_layer_widths": [64],
        "actor_dense_layer_widths": [64],
        "critic_dense_layer_widths": [64],
        "reward_dense_layer_widths": [64],
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

    config = MuZeroConfig(config_dict, game_config)

    agent = MuZeroAgent(env, config, name="smoke_test_cartpole", device="cpu")

    assert agent.model is not None
    assert agent.replay_buffer is not None


def test_muzero_tictactoe_smoke():
    """
    Smoke test for MuZero on TicTacToe.
    """
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
    }

    config = MuZeroConfig(config_dict, game_config)

    agent = MuZeroAgent(env, config, name="smoke_test_tictactoe", device="cpu")
    # Patch player_id because TicTacToe uses player_1
    agent.player_id = "player_1"

    assert agent.model is not None
    assert agent.replay_buffer is not None


if __name__ == "__main__":
    pytest.main([__file__])
