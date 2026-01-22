import torch
import pytest
import numpy as np
from agents.muzero import MuZeroAgent
from agents.ppo import PPOAgent
from agent_configs.muzero_config import MuZeroConfig
from agent_configs.ppo_config import PPOConfig
from agent_configs.actor_config import ActorConfig
from agent_configs.critic_config import CriticConfig
from game_configs.cartpole_config import CartPoleConfig
from modules.world_models.muzero_world_model import MuzeroWorldModel
from losses.basic_losses import CategoricalCrossentropyLoss, MSELoss

def action_as_onehot(action, num_actions):
    one_hot = torch.zeros(num_actions)
    one_hot[action] = 1.0
    return one_hot

class MockGame:
    def __init__(self, obs_shape):
        self.observation_history = [np.random.randn(*obs_shape) for _ in range(5)]
        self.action_history = [0, 1, 0, 1]
        self.rewards = [0.0, 1.0, 0.0, 1.0]
        self.value_history = [0.0, 0.5, 0.0, 0.5, 0.0]
        self.policy_history = [torch.tensor([0.5, 0.5]) for _ in range(5)]
        self.info_history = [{"legal_moves": [0, 1], "player": 0, "chance": 0, "done": False} for _ in range(4)]
        self.info_history.append({"legal_moves": [0, 1], "player": 0, "chance": 0, "done": True})

def test_muzero_mixed_precision_cpu():
    print("Testing MuZero Mixed Precision on CPU...")
    game_config = CartPoleConfig()
    env = game_config.make_env()
    
    config_dict = {
        "world_model_cls": MuzeroWorldModel,
        "use_mixed_precision": True,
        "training_steps": 1,
        "games_per_generation": 1,
        "min_replay_buffer_size": 2,
        "minibatch_size": 2,
        "num_simulations": 2,
        "unroll_steps": 1,
        "multi_process": False,
        "action_function": action_as_onehot,
        "value_loss_function": MSELoss(),
        "reward_loss_function": MSELoss(),
        "policy_loss_function": CategoricalCrossentropyLoss(),
        # "support_range": 31,
        
        # MLP Config for CartPole
        "representation_residual_layers": [],
        "representation_conv_layers": [],
        "representation_dense_layer_widths": [64, 64],
        
        "dynamics_residual_layers": [],
        "dynamics_conv_layers": [],
        "dynamics_dense_layer_widths": [64, 64],
        
        "reward_conv_layers": [],
        "reward_dense_layer_widths": [64],
        
        "to_play_conv_layers": [],
        "to_play_dense_layer_widths": [64],
        
        "critic_conv_layers": [],
        "critic_dense_layer_widths": [64],
        
        "actor_conv_layers": [],
        "actor_dense_layer_widths": [64],

        "afterstate_residual_layers": [],
        "afterstate_conv_layers": [],
        "afterstate_dense_layer_widths": [64],

        "chance_conv_layers": [],
        "chance_dense_layer_widths": [64],
        
        "residual_layers": [],
        "conv_layers": [],
        "dense_layer_widths": [64, 64],
    }
    config = MuZeroConfig(config_dict, game_config)
    agent = MuZeroAgent(env, config, device=torch.device("cpu"))
    
    # Store dummy data
    for _ in range(2):
        game = MockGame((4,))
        agent.replay_buffer.store_aggregate(game)
    
    agent.learn()
    assert hasattr(agent, "scaler")
    print("MuZero Mixed Precision CPU test passed!")

def test_ppo_mixed_precision_cpu():
    print("Testing PPO Mixed Precision on CPU...")
    game_config = CartPoleConfig()
    env = game_config.make_env()
    
    actor_config = ActorConfig({"learning_rate": 0.001, "clipnorm": 0.0})
    critic_config = CriticConfig({"learning_rate": 0.001, "clipnorm": 0.0})
    
    config_dict = {
        "use_mixed_precision": True,
        "training_steps": 1,
        "steps_per_epoch": 10,
        "train_policy_iterations": 1,
        "train_value_iterations": 1,
        "multi_process": False,
    }
    
    config = PPOConfig(config_dict, game_config, actor_config, critic_config)
    agent = PPOAgent(env, config, device=torch.device("cpu"))
    
    # Store dummy data
    for _ in range(10):
        agent.replay_buffer.store(
            observations=env.observation_space.sample(),
            info={"legal_moves": [0, 1]},
            actions=0,
            values=0.0,
            log_probabilities=0.0,
            rewards=0.0
        )
    agent.replay_buffer.finalize_trajectory(0.0)
    
    agent.learn()
    assert hasattr(agent, "scaler")
    print("PPO Mixed Precision CPU test passed!")

if __name__ == "__main__":
    test_muzero_mixed_precision_cpu()
    test_ppo_mixed_precision_cpu()
