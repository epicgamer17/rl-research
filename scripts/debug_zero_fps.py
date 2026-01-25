import sys
import os
import time
import torch
import torch.multiprocessing as mp
import traceback

# Add project root and custom gym envs package to path
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)
sys.path.append(os.path.join(root_path, 'custom_gym_envs_pkg'))

from agents.muzero import MuZeroAgent
from agent_configs.muzero_config import MuZeroConfig
from game_configs.catan_config import CatanConfig
from modules.world_models.muzero_world_model import MuzeroWorldModel
from torch.optim import Adam

def run_debug_config(name, config_params):
    print(f"\n--- Debugging Config: {name} ---")
    print(f"Params: {config_params}")
    
    game_config = CatanConfig()
    config = MuZeroConfig(config_dict=config_params, game_config=game_config)
    env = game_config.make_env()
    
    try:
        agent = MuZeroAgent(
            env=env,
            config=config,
            name="debug_run",
            device=torch.device("cpu"),
        )
        
        stop_flag = mp.Value("i", 0)
        error_queue = mp.Queue()
        stats_client = agent.stats.get_client()
        
        workers = []
        for i in range(config.num_workers):
            p = mp.Process(target=agent.worker_fn, args=(i, stop_flag, stats_client, error_queue))
            p.start()
            workers.append(p)
        
        start_time = time.time()
        timeout = 120 # Increased timeout
        steps = 0
        
        while time.time() - start_time < timeout:
            agent.stats.drain_queue()
            if not error_queue.empty():
                err, tb = error_queue.get()
                print(f"WORKER ERROR detected:\n{err}\n{tb}")
                break
                
            steps = agent.stats.get_num_steps()
            if steps > 0:
                print(f"Progress: {steps} steps at {time.time() - start_time:.2f}s")
                if steps >= 50:
                    break
            
            # Check if workers are still alive
            all_dead = True
            for i, p in enumerate(workers):
                if p.is_alive():
                    all_dead = False
                else:
                    if stop_flag.value == 0:
                        print(f"Worker {i} died unexpectedly!")
            
            if all_dead:
                print("All workers died.")
                break
                
            time.sleep(2)
            
        print(f"Debug Result: {steps} steps in {time.time() - start_time:.2f}s")
        
    except Exception as e:
        print(f"MAIN PROCESS ERROR:\n{e}")
        traceback.print_exc()
    finally:
        stop_flag.value = 1
        for p in workers:
            p.join(timeout=2)
            if p.is_alive():
                p.terminate()
        env.close()

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
    # Common base params
    base = {
        "num_simulations": 10,
        "world_model_cls": MuzeroWorldModel,
        "min_replay_buffer_size": 100,
        "dense_layer_widths": [128, 128],
        "conv_layers": [],
        "residual_layers": [],
        "representation_dense_layer_widths": [128, 128],
        "representation_conv_layers": [],
        "representation_residual_layers": [],
        "dynamics_dense_layer_widths": [128, 128],
        "dynamics_conv_layers": [],
        "dynamics_residual_layers": [],
        "reward_dense_layer_widths": [32],
        "reward_conv_layers": [],
        "reward_residual_layers": [],
        "actor_dense_layer_widths": [32],
        "actor_conv_layers": [],
        "actor_residual_layers": [],
        "critic_dense_layer_widths": [32],
        "critic_conv_layers": [],
        "critic_residual_layers": [],
        "to_play_dense_layer_widths": [32],
        "to_play_conv_layers": [],
        "to_play_residual_layers": [],
        "multi_process": True,
        "device": "cpu",
    }
    
    # Config A: High Concurrency + Compile
    config_a = {**base, "search_batch_size": 0, "compile": True, "use_mixed_precision": False, "quantize": False, "num_workers": 4, "num_envs_per_worker": 8}
    
    # Config B: Compile + Quantize
    config_b = {**base, "search_batch_size": 0, "compile": True, "use_mixed_precision": False, "quantize": True, "num_workers": 1, "num_envs_per_worker": 1}

    # Config C: Quantize + Mixed Precision
    config_c = {**base, "search_batch_size": 0, "compile": False, "use_mixed_precision": True, "quantize": True, "num_workers": 1, "num_envs_per_worker": 1}

    # run_debug_config("HIGH_CONCUR_COMPILE", config_a)
    # run_debug_config("COMPILE_PLUS_QUANTIZE", config_b)
    run_debug_config("QUANTIZE_PLUS_MIXED_PRECISION", config_c)
