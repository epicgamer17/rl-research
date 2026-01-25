import sys
import os
import time
import torch
import multiprocessing as mp
# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.muzero import MuZeroAgent
from agent_configs.muzero_config import MuZeroConfig
from game_configs.catan_config import CatanConfig
from modules.world_models.muzero_world_model import MuzeroWorldModel
from torch.optim import Adam

def benchmark(num_envs_per_worker):
    print(f"\nBenchmarking with {num_envs_per_worker} envs per worker...")
    
    params = {
        "num_simulations": 2, # Reduce sim count to focus on env/inference throughput
        "num_workers": 1,
        "num_envs_per_worker": num_envs_per_worker,
        "minibatch_size": 32,
        "replay_buffer_size": 1000,
        "min_replay_buffer_size": 1000, # prevent learning
        "dense_layer_widths": [128, 128],
        "optimizer": Adam,
        "learning_rate": 0.001,
        "use_mixed_precision": False, # CPU
        "compile": False,
        "device": "cpu",
        "turn_based": True, # Catan
        "turn_based": True, # Catan
        "world_model_cls": MuzeroWorldModel,
        "residual_layers": [],
        "conv_layers": [],
        "dense_layer_widths": [128, 128],
        
        "representation_dense_layer_widths": [128, 128],
        "representation_conv_layers": [], 
        "representation_residual_layers": [],
        
        "dynamics_dense_layer_widths": [128, 128],
        "dynamics_conv_layers": [],
        "dynamics_residual_layers": [],
        
        "reward_dense_layer_widths": [32],
        "reward_conv_layers": [],
        "reward_residual_layers": [],
        
        "chance_dense_layer_widths": [32],
        "chance_conv_layers": [],
        "chance_residual_layers": [],
        
        "actor_dense_layer_widths": [32],
        "actor_conv_layers": [],
        "actor_residual_layers": [],
        
        "critic_dense_layer_widths": [32],
        "critic_conv_layers": [],
        "critic_residual_layers": [],
        
        "to_play_dense_layer_widths": [32],
        "to_play_conv_layers": [],
        "to_play_residual_layers": [],
        
        "known_bounds": [-1, 1],
        "support_range": None, 
        "lr_ratio": 1.0,
        "transfer_interval": 1,
        "reanalyze_ratio": 0.0,
        "reanalyze_method": "mcts", 
        "reanalyze_tau": 0.3,
        "reanalyze_noise": False,
        "reanalyze_update_priorities": False,
        "consistency_loss_factor": 0.0,
        "projector_output_dim": 128,
        "projector_hidden_dim": 128,
        "predictor_output_dim": 128,
        "predictor_hidden_dim": 64,
        "mask_absorbing": True,
        "value_prefix": False,
        "lstm_horizon_len": 5,
        "lstm_hidden_size": 64,
        "q_estimation_method": "v_mix",
        "stochastic": True, # Catan is stochastic
        "use_true_chance_codes": False,
        "num_chance": 32,
        "afterstate_residual_layers": [],
        "afterstate_conv_layers": [],
        "afterstate_dense_layer_widths": [128],
        "vqvae_commitment_cost_factor": 1.0,
        "action_embedding_dim": 32,
        "latent_viz_method": "umap",
        "latent_viz_interval": 1000,
        "search_batch_size": 8,
        "use_virtual_mean": True,
        "virtual_loss": 3.0,
        "root_dirichlet_alpha": 0.03,
        "gumbel": False,
        "value_loss_factor": 1.0, 
        "to_play_loss_factor": 2.0,
        "injection_frac": 0.25,
        "temperature_updates": [30],
        "temperatures": [1.0, 0.0],
    }
    
    game_config = CatanConfig()
    # Need to populate full config defaults or use a base config
    config = MuZeroConfig(config_dict=params, game_config=game_config)
    
    # Initialize agent
    env = CatanConfig().make_env()
    agent = MuZeroAgent(
        env=env,
        config=config,
        name=f"bench_{num_envs_per_worker}",
        device="cpu",
    )
    
    stop_flag = mp.Value("i", 0)
    error_queue = mp.Queue()
    
    # Start 1 worker
    # Ensure stats are setup for multiprocessing
    if not hasattr(agent.stats, "get_client"):
         # Force server initialization if needed?
         pass
         
    stats_client = agent.stats.get_client()
    p = mp.Process(target=agent.worker_fn, args=(0, stop_flag, stats_client, error_queue))
    p.start()
    
    start_time = time.time()
    initial_steps = 0
    target_steps = 1000
    
    try:
        while True:
            # Drain queue to receive updates
            agent.stats.drain_queue()
            
            # Check for errors
            if not error_queue.empty():
                err = error_queue.get()
                print("Worker error:", err)
                break
                
            current_steps = agent.stats.get_num_steps()
            if current_steps >= target_steps:
                break
            time.sleep(1)
            print(f"Steps: {current_steps}/{target_steps}", end="\r")
            
        duration = time.time() - start_time
        fps = current_steps / duration
        print(f"\nCollected {current_steps} frames in {duration:.2f}s -> {fps:.2f} FPS")
        
    finally:
        stop_flag.value = 1
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()
            
    return fps

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
        
    results = {}
    for n in [1, 2, 4, 8]:
        fps = benchmark(n)
        results[n] = fps
    
    print("\nBenchmark Results (FPS):")
    for n, fps in results.items():
        print(f"{n} Envs: {fps:.2f}")
