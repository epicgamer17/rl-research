import sys
import os
import time
import torch
import multiprocessing as mp

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

def test_quantization():
    print("Testing MuZero with Quantization and Multiprocessing...")
    game_config = CatanConfig()
    
    config_params = {
        "num_simulations": 10,
        "search_batch_size": 0,
        "num_workers": 2,
        "num_envs_per_worker": 1,
        "multi_process": True,
        "quantize": True,
        "compile": False,
        "use_mixed_precision": False,
        "device": "cpu",
        "world_model_cls": MuzeroWorldModel,
        "min_replay_buffer_size": 100,
        "dense_layer_widths": [64, 64],
        "conv_layers": [],
        "residual_layers": [],
        
        "representation_dense_layer_widths": [64, 64],
        "representation_conv_layers": [],
        "representation_residual_layers": [],
        
        "dynamics_dense_layer_widths": [64, 64],
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
    }
    
    config = MuZeroConfig(config_dict=config_params, game_config=game_config)
    env = game_config.make_env()
    
    agent = MuZeroAgent(
        env=env,
        config=config,
        name="quant_test",
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
    steps = 0
    while time.time() - start_time < 30: # 30s test
        agent.stats.drain_queue()
        if not error_queue.empty():
            err, tb = error_queue.get()
            print(f"Error: {err}\n{tb}")
            break
        steps = agent.stats.get_num_steps()
        if steps > 50:
            break
        time.sleep(1)
        
    print(f"Test finished. Steps: {steps}, Time: {time.time() - start_time:.2f}s")
    
    stop_flag.value = 1
    for p in workers:
        p.join(timeout=2)
        if p.is_alive():
            p.terminate()
    env.close()

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    test_quantization()
