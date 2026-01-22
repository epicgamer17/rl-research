
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import torch
import numpy as np
from agents.muzero import MuZeroAgent
from agents.rainbow_dqn import RainbowAgent
from agents.ppo import PPOAgent
from modules.world_models.muzero_world_model import MuzeroWorldModel
import gymnasium as gym

# Mock Config Classes (reused from verify_compile.py)
class DummyGame:
    def __init__(self):
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(4,))
        self.action_space = gym.spaces.Discrete(2)
        self.num_players = 1
        self.is_deterministic = True
        self.is_discrete = True # Added

class SubConfig:
    def __init__(self, optimizer): 
         self.optimizer = optimizer
         self.clipnorm = 0
         self.learning_rate = 1e-3
         self.noisy_sigma = 0 # Added

class DummyConfig:
    def __init__(self, compile_enabled=False, compile_mode="default"):
        self.game = DummyGame()
        # ... (rest of attributes)
        self.minibatch_size = 32
        self.observation_dimensions = (4,)
        self.mixed_precision_dtype = torch.float16
        self.training_steps = 100
        self.optimizer = torch.optim.Adam
        self.learning_rate = 1e-3
        self.adam_epsilon = 1e-8
        self.weight_decay = 0
        self.compile = compile_enabled
        self.compile_mode = compile_mode
        self.use_mixed_precision = False
        self.clipnorm = 0
        self.training_iterations = 1
        self.num_minibatches = 1
        self.replay_buffer_size = 100
        self.min_replay_buffer_size = 10
        self.n_step = 1
        self.discount_factor = 0.99
        self.per_alpha = 0.6
        self.per_beta = 0.4
        self.per_beta_final = 1.0
        self.per_epsilon = 1e-6
        self.per_use_batch_weights = False
        self.per_use_initial_max_priority = False
        self.lstm_horizon_len = 5
        self.value_prefix = False
        self.reanalyze_tau = 1.0
        self.multi_process = False
        self.games_per_generation = 1
        self.lr_ratio = 1.0
        self.transfer_interval = 100
        self.stochastic = False
        self.action_embedding_dim = 16 # Added
        self.kernel_initializer = None # Added
        
        # MuZero specific
        self.world_model_cls = MuzeroWorldModel
        self.unroll_steps = 5
        self.reanalyze_ratio = 0.5
        self.num_workers = 1
        
        # Rainbow specific
        self.atom_size = 1
        self.v_min = -10
        self.v_max = 10
        self.eg_epsilon = 0.1
        self.eg_epsilon_final = 0.01
        self.eg_epsilon_decay_type = "linear"
        self.eg_epsilon_final_step = 100
        self.replay_interval = 4
        self.test_interval = 1000
        self.save_intermediate_weights = False
        self.checkpoint_interval = 1000
        self.steps_per_epoch = 100
        self.dueling = False # Added

        # OptimizationConfig
        self.lr_schedule_type = "none"
        self.lr_schedule_steps = []
        self.lr_schedule_values = []

        # Recurrent/MCTS
        self.projector_hidden_dim = 16
        self.predictor_hidden_dim = 16
        self.projector_output_dim = 16
        self.predictor_output_dim = 16
        self.gumbel = False
        self.pb_c_base = 19652
        self.pb_c_init = 1.25
        self.num_simulations = 5  # Low for test
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25
        self.root_dirichlet_alpha_adaptive = False
        self.use_virtual_mean = False # Added default
        self.virtual_loss = 3.0 # Added default
        self.search_batch_size = 0 # Added default

        # PPO
        self.gae_lambda = 0.95
        self.clip_low_prob = 1e-5
        self.target_kl = 0.01
        self.entropy_coefficient = 0.01
        self.clip_param = 0.2
        self.train_policy_iterations = 1
        self.train_value_iterations = 1
        self.critic_coefficient = 0.5
        self.noisy_sigma = 0 # Added

        self.actor = SubConfig(torch.optim.Adam)
        self.critic = SubConfig(torch.optim.Adam)
        
        # Policy Imitation
        self.loss_function = torch.nn.MSELoss()
        
        # Dummy Num chance/support range
        self.num_chance = 0
        self.support_range = None

def benchmark_agent(agent_cls, name, run_fn, config_updates={}, input_shape=None):
    print(f"\n--- Benchmarking {name} ---")
    
    modes = [
        ("Baseline (No Compile)", False, "default"),
        # ("Compiled (Default)", True, "default"),
        ("Compiled (Reduce Overhead)", True, "reduce-overhead"),
        # ("Compiled (Max Autotune)", True, "max-autotune"),
    ]
    
    results = {}

    for mode_name, compile_enabled, compile_mode in modes:
        print(f"Running: {mode_name}")
        config = DummyConfig(compile_enabled=compile_enabled, compile_mode=compile_mode)
        # Apply specific config updates
        for k, v in config_updates.items():
            setattr(config, k, v)
            
        try:
             # Hack for PPO sub-configs which are instances
            if name == "PPO":
                 config.actor = SubConfig(torch.optim.Adam)
                 config.critic = SubConfig(torch.optim.Adam)

            agent = agent_cls(gym.make("CartPole-v1"), config, device=torch.device("cpu"))
            
            # WARMUP
            print("  Warmup...")
            warmup_start = time.time()
            for _ in range(10): # Run a few times to trigger JIT
                run_fn(agent)
            warmup_time = time.time() - warmup_start
            print(f"  Warmup time: {warmup_time:.4f}s")
            
            # BENCHMARK
            print("  Benchmarking...")
            start_time = time.time()
            iterations = 100
            for _ in range(iterations):
                run_fn(agent)
            total_time = time.time() - start_time
            avg_time = total_time / iterations * 1000 # ms
            
            print(f"  Avg Inference Time: {avg_time:.4f} ms")
            results[mode_name] = avg_time
            
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            results[mode_name] = None
    
    # Calculate Speedup
    baseline = results.get("Baseline (No Compile)")
    if baseline:
        for mode_name, val in results.items():
            if mode_name != "Baseline (No Compile)" and val:
                speedup = baseline / val
                print(f"Speedup {mode_name}: {speedup:.2f}x")

def run_muzero_inference(agent):
    # Test initial_inference
    obs = torch.randn(1, *agent.observation_dimensions).to(agent.device)
    with torch.no_grad():
        agent.model.initial_inference(obs)

def run_rainbow_inference(agent):
    obs = torch.randn(1, *agent.observation_dimensions).to(agent.device)
    with torch.no_grad():
        agent.model(obs)

def run_ppo_inference(agent):
    obs = torch.randn(1, *agent.observation_dimensions).to(agent.device)
    with torch.no_grad():
        # PPO predict calls model.actor and model.critic
        # Let's call the model's forward or just actor/critic directly if that's what predict does
        agent.predict(obs, info={"legal_moves": []}, mask_actions=False) 

if __name__ == "__main__":
    benchmark_agent(MuZeroAgent, "MuZero", run_muzero_inference)
    benchmark_agent(RainbowAgent, "Rainbow", run_rainbow_inference)
    benchmark_agent(PPOAgent, "PPO", run_ppo_inference)
