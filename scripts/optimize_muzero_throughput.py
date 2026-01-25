import sys
import os
import time
import torch
import torch.multiprocessing as mp
import pandas as pd
import itertools
from tabulate import tabulate
import argparse

# Set quantization engine globally for stable multiprocessing with quantized models
try:
    if 'fbgemm' in torch.backends.quantized.supported_engines:
        torch.backends.quantized.engine = 'fbgemm'
    elif 'qnnpack' in torch.backends.quantized.supported_engines:
        torch.backends.quantized.engine = 'qnnpack'
except:
    pass

# Add project root and custom gym envs package to path
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)
sys.path.append(os.path.join(root_path, 'custom_gym_envs_pkg'))

from agents.muzero import MuZeroAgent
from agent_configs.muzero_config import MuZeroConfig
from game_configs.catan_config import CatanConfig
from modules.world_models.muzero_world_model import MuzeroWorldModel
from torch.optim import Adam

def run_benchmark(config_params, game_config, target_steps=200, timeout=60):
    """Runs a single benchmark configuration and returns FPS."""
    
    # Increase base timeout for compilation overhead
    if config_params.get("compile", False):
        timeout = 300
    
    # Ensure CPU for these benchmarks as requested
    config_params["device"] = "cpu"
    
    # Base params for all runs to ensure stability
    base_params = {
        "num_simulations": 10, 
        "minibatch_size": 32,
        "replay_buffer_size": 10000,
        "min_replay_buffer_size": 5000, # prevent learning start during benchmark
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
        "lr_ratio": float('inf'), # prevent learning from blocking
        "transfer_interval": 1000,
        "multi_process": True,
        "stochastic": True,
    }
    
    # Merge base params with config grid params
    params = {**base_params, **config_params}
    
    # Initialize config and agent
    config = MuZeroConfig(config_dict=params, game_config=game_config)
    env = game_config.make_env()
    
    agent = MuZeroAgent(
        env=env,
        config=config,
        name="benchmark_run",
        device=torch.device("cpu"),
    )
    
    stop_flag = mp.Value("i", 0)
    error_queue = mp.Queue()
    stats_client = agent.stats.get_client()
    
    # Use a single worker for benchmark to avoid oversubscribing during grid search
    # but the grid search itself can test multiple workers if requested
    workers = []
    for i in range(config.num_workers):
        p = mp.Process(target=agent.worker_fn, args=(i, stop_flag, stats_client, error_queue))
        p.start()
        workers.append(p)
    
    # Warmup phase to handle compilation/initialization overhead
    warmup_steps = 10
    if config_params.get("compile", False):
        warmup_steps = 40 # Compilation needs more time/steps to settle
        
    print(f"  Warming up for {warmup_steps} steps...")
    warmup_start = time.time()
    
    while True:
        agent.stats.drain_queue()
        if not error_queue.empty():
            break # Let main loop handle error
        
        current_steps = agent.stats.get_num_steps()
        if current_steps >= warmup_steps:
            break
            
        if time.time() - warmup_start > timeout + 30:
            print("  Warmup timed out.")
            break
        time.sleep(0.5)

    print("  Benchmark started...")
    start_bench = time.time()
    start_steps = agent.stats.get_num_steps() # Record steps at start of benchmark
    
    try:
        while time.time() - start_bench < timeout:
            agent.stats.drain_queue()
            
            if not error_queue.empty():
                err, tb = error_queue.get()
                print(f"  Worker Error: {err}")
                break
                
            current_steps = agent.stats.get_num_steps()
            total_steps = current_steps - start_steps # Calculate steps ONLY during benchmark window
            
            if total_steps >= target_steps:
                break
            time.sleep(0.5)
            
        duration = time.time() - start_bench
        fps = total_steps / duration if duration > 0 else 0
        
    except Exception as e:
        print(f"Benchmark Failed: {e}")
    finally:
        stop_flag.value = 1
        for p in workers:
            p.join(timeout=2)
            if p.is_alive():
                p.terminate()
                p.join()
            try:
                p.close()
            except AttributeError:
                pass # Python < 3.7
        env.close()
        import gc
        gc.collect()
        
    return fps, total_steps

def main():
    parser = argparse.ArgumentParser(description="MuZero Throughput Grid Search")
    parser.add_argument("--test-run", action="store_true", help="Run a minimal grid for verification")
    args = parser.parse_args()

    if args.test_run:
        grid = {
            "search_batch_size": [0, 8],
            "compile": [False],
            "use_mixed_precision": [False],
            "quantize": [True],
            "num_workers": [1],
            "num_envs_per_worker": [1],
        }
        target_steps = 50
    else:
        grid = {
            "search_batch_size": [0, 8, 16],
            "compile": [True, False],
            "use_mixed_precision": [True, False],
            "quantize": [True, False],
            "num_workers": [1, 2, 4],
            "num_envs_per_worker": [1, 4, 8],
        }
        target_steps = 150

    keys = grid.keys()
    values = grid.values()
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"Starting Grid Search with {len(combinations)} combinations...")
    
    game_config = CatanConfig()
    results = []
    
    # Ensure 'benchmarks' directory exists
    os.makedirs("benchmarks", exist_ok=True)
    
    for i, config_params in enumerate(combinations):
        # Skip combinations known to cause issues on CPU or that are redundant
        # 1. Compile + Mixed Precision on CPU is unstable and often slower
        # if config_params["compile"] and config_params["use_mixed_precision"]:
        #     print(f"[{i+1}/{len(combinations)}] Skipping: {config_params} (Compile + Mixed Precision on CPU is unstable)")
        #     continue
            
        # # 2. Quantize + Mixed Precision on CPU is redundant/problematic
        # if config_params["quantize"] and config_params["use_mixed_precision"]:
        #     print(f"[{i+1}/{len(combinations)}] Skipping: {config_params} (Quantize + Mixed Precision is redundant)")
        #     continue

        # 3. Quantize + Compile is now enabled for testing
        # if config_params["quantize"] and config_params["compile"]:
        #     print(f"[{i+1}/{len(combinations)}] Skipping: {config_params} (Quantize + Compile is often problematic)")
        #     continue
            
        # 4. Mixed precision is now enabled for testing
        # if config_params["use_mixed_precision"]:
        #     print(f"[{i+1}/{len(combinations)}] Skipping: {config_params} (Mixed Precision on CPU is unstable)")
        #     continue
            
        print(f"[{i+1}/{len(combinations)}] Testing: {config_params}")
        
        # Mixed precision and quantization usually don't work together or have specific requirements
        # On CPU, mixed precision (autocast) might not give speedup depending on torch version/hardware
        # Quantize usually requires CPU.
        
        fps, steps = run_benchmark(config_params, game_config, target_steps=target_steps)
        
        result_entry = {**config_params, "fps": fps, "steps": steps}
        results.append(result_entry)
        print(f"  Result: {fps:.2f} FPS")
        
        # Save intermediate results
        pd.DataFrame(results).to_csv("benchmarks/muzero_optimization_steps.csv", index=False)
        
        # Give the OS a moment to reclaim semaphores and shared memory
        time.sleep(1)

    df = pd.DataFrame(results)
    df = df.sort_values(by="fps", ascending=False)
    
    print("\n" + "="*50)
    print("GRID SEARCH RESULTS (Sorted by FPS)")
    print("="*50)
    print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))
    
    df.to_csv("benchmarks/muzero_optimization_final.csv", index=False)
    print(f"\nFinal results saved to benchmarks/muzero_optimization_final.csv")

if __name__ == "__main__":
    # Use spawn for multiprocessing compatibility
    try:
        mp.set_start_method('spawn', force=True)
        # file_system is generally safer for many small processes on macOS
        mp.set_sharing_strategy('file_system')
    except RuntimeError:
        pass
    main()
