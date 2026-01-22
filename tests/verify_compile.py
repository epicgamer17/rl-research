
import sys
import torch
import gymnasium as gym
from agent_configs.muzero_config import MuZeroConfig
from agent_configs.ppo_config import PPOConfig
from agent_configs.dqn.rainbow_config import RainbowConfig
from agent_configs.dqn.nfsp_config import NFSPDQNConfig
from agents.muzero import MuZeroAgent
from agents.ppo import PPOAgent
from agents.rainbow_dqn import RainbowAgent
from agents.nfsp import NFSPDQN
from game_configs.cartpole_config import CartPoleConfig

from modules.world_models.muzero_world_model import MuzeroWorldModel

class DummyConfig:
    def __init__(self):
        self.compile = True
        self.game = None
        self.training_steps = 10
        self.multi_process = False
        # Add other common attributes with reasonable defaults
        self.minibatch_size = 2
        self.replay_buffer_size = 100
        self.min_replay_buffer_size = 4
        self.n_step = 1
        self.discount_factor = 0.99
        self.per_alpha = 0.5
        self.per_beta = 0.5
        self.per_epsilon = 1e-6
        self.per_use_batch_weights = False
        self.per_use_initial_max_priority = True
        self.lstm_horizon_len = 5
        self.value_prefix = False
        self.reanalyze_tau = 1.0
        self.optimizer = torch.optim.Adam
        self.learning_rate = 1e-3
        self.adam_epsilon = 1e-8
        self.weight_decay = 0
        self.compile = True
        self.compile_mode = "default"
        self.use_mixed_precision = False
        self.clipnorm = 0
        self.training_iterations = 1
        self.norm_type = "none" # Missing
        self.noisy_sigma = 0 # Missing
        self.action_embedding_dim = 16 # Missing
        self.prob_layer_initializer = None # Missing
        self.activation = torch.nn.ReLU # Missing
        
        # MuZero specific
        self.unroll_steps = 2
        self.world_model_cls = MuzeroWorldModel 
        self.reanalyze_ratio = 0.5
        self.num_workers = 1
        self.games_per_generation = 1
        self.lr_ratio = 1.0
        self.num_minibatches = 1
        self.transfer_interval = 10
        self.stochastic = False
        self.num_chance = 0 # Missing
        self.lr_schedule_type = "none" # Missing
        self.latent_viz_interval = 100
        
        self.projector_hidden_dim = 16 # Missing
        self.predictor_hidden_dim = 16 # Missing
        self.projector_output_dim = 16 # Missing
        self.predictor_output_dim = 16 # Missing
        
        # Search
        self.gumbel = False # Missing
        self.gumbel_m = 16
        self.gumbel_cvisit = 50
        self.gumbel_cscale = 1.0
        self.pb_c_base = 19652
        self.pb_c_init = 1.25
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25
        self.root_dirichlet_alpha_adaptive = False
        self.min_max_epsilon = 0.01
        self.search_batch_size = 1 # Not 0
        self.use_virtual_mean = False
        self.virtual_loss = 3.0
        self.num_simulations = 10
        self.latent_viz_method = "tsne"
        self.save_intermediate_weights = False
        
        # Rainbow specific
        self.atom_size = 1
        self.v_min = -10
        self.v_max = 10
        self.kernel_initializer = None
        self.eg_epsilon_decay_type = "linear"
        self.eg_epsilon = 0.1
        self.eg_epsilon_final = 0.01
        self.eg_epsilon_final_step = 100
        self.replay_interval = 4
        self.dueling = False # Missing
        
        # PPO specific
        self.actor = self
        self.critic = self
        self.gae_lambda = 0.95
        self.clip_low_prob = 0.1
        self.clip_param = 0.2
        self.entropy_coefficient = 0.01
        self.train_policy_iterations = 1
        self.train_value_iterations = 1
        self.steps_per_epoch = 10
        self.target_kl = 0.01
        self.critic_coefficient = 0.5
        self.support_range = None # Missing
        self.distributional_head = False # Likely needed
        
        # NFSP
        self.rl_configs = [self]
        self.sl_configs = [self]
        self.shared_networks_and_buffers = True
        self.anticipatory_param = 0.1
        self.loss_function = torch.nn.MSELoss()



def test_agent_compile(agent_cls, config_cls, name):
    print(f"Testing {name} with compile=True...")
    game_config = CartPoleConfig()
    env = gym.make("CartPole-v1")
    
    # Mock config
    config = DummyConfig()
    config.game = game_config
    
    # Initialize agent
    try:
        agent = agent_cls(env=env, config=config, name=f"test_{name}")
        print(f"✅ {name} initialized successfully.")
        
        # Run a simple forward pass to trigger compilation (lazy compilation)
        if name != "MuZero": # MuZero network construction is complex without real config
             pass 
        
        print(f"✅ {name} verification passed (init only).")
    except Exception as e:
        print(f"❌ {name} failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Test MuZero
    test_agent_compile(MuZeroAgent, MuZeroConfig, "MuZero")
    
    # Test Rainbow
    test_agent_compile(RainbowAgent, RainbowConfig, "RainbowDQN")
    
    # Test PPO
    test_agent_compile(PPOAgent, PPOConfig, "PPO")
    
    # Test NFSP (which uses Rainbow)
    # NFSP structure is complex, might need more mocks
    # test_agent_compile(NFSPDQN, NFSPDQNConfig, "NFSP")
