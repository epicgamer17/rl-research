import sys
import os
import time
import torch
import torch.multiprocessing as mp
import pandas as pd
import argparse
from tabulate import tabulate

# Add project root and custom gym envs package to path
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)
sys.path.append(os.path.join(root_path, 'custom_gym_envs_pkg'))

# Set quantization engine globally
try:
    if 'fbgemm' in torch.backends.quantized.supported_engines:
        torch.backends.quantized.engine = 'fbgemm'
    elif 'qnnpack' in torch.backends.quantized.supported_engines:
        torch.backends.quantized.engine = 'qnnpack'
except:
    pass

from agents.muzero import MuZeroAgent
from agent_configs.muzero_config import MuZeroConfig
from game_configs.catan_config import CatanConfig
from modules.world_models.muzero_world_model import MuzeroWorldModel
from torch.optim import Adam

def run_repro():
    print("Running reproduction for failing config: 0,True,False,True,1,1,cpu")
    
    # Configuration that failed
    config_params = {
        "search_batch_size": 0,
        "compile": True,
        "use_mixed_precision": False,
        "quantize": True,
        "num_workers": 1,
        "num_envs_per_worker": 1,
        "device": "cpu",
        
        # Base params
        "num_simulations": 10, 
        "minibatch_size": 32,
        "replay_buffer_size": 10000,
        "min_replay_buffer_size": 5000,
        "optimizer": Adam,
        "learning_rate": 0.001,
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
        "lr_ratio": float('inf'),
        "transfer_interval": 1000,
        "multi_process": True,
        "stochastic": True,
    }

    game_config = CatanConfig()
    env = game_config.make_env()
    config = MuZeroConfig(config_dict=config_params, game_config=game_config)
    
    agent = MuZeroAgent(
        env=env,
        config=config,
        name="repro_run",
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
        
    # Identical warmup logic to optimize_muzero_throughput.py
    warmup_steps = 40 # compile=True
    print(f"  Warming up for {warmup_steps} steps...")
    warmup_start = time.time()
    
    # 3-minute timeout + 30s buffer
    timeout = 210 
    
    while True:
        agent.stats.drain_queue()
        if not error_queue.empty():
            err, tb = error_queue.get()
            print(f"WARMUP ERROR: {err}\n{tb}")
            break
        
        current_steps = agent.stats.get_num_steps()
        if current_steps >= warmup_steps:
            print(f"Warmup complete at {time.time() - warmup_start:.2f}s")
            break
            
        if time.time() - warmup_start > timeout:
            print(f"  Warmup timed out after {time.time() - warmup_start:.2f}s. Steps: {current_steps}")
            break
        time.sleep(1)

    # Cleanup
    stop_flag.value = 1
    for p in workers:
        p.join(timeout=2)
        if p.is_alive():
            p.terminate()
            p.join()
        try:
            p.close()
        except:
            pass
    env.close()

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
        mp.set_sharing_strategy('file_system')
    except RuntimeError:
        pass
    run_repro()
