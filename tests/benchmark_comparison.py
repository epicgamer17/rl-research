import sys
import os
import time
import torch
import gymnasium as gym
from tabulate import tabulate
import copy

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Imports
from game_configs.tictactoe_config import TicTacToeConfig
from agent_configs.muzero_config import MuZeroConfig
from agents.tictactoe_expert import TicTacToeBestAgent
from agents.random import RandomAgent

# Dynamic imports for agents
from agents.muzero import MuZeroAgent as MuZeroRay
import ray
from modules.world_models.muzero_world_model import MuzeroWorldModel


import itertools
import csv
import datetime
import gc


class Benchmark:
    def __init__(self):
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            print("Warning: CUDA available but using CPU as per user snippet request")

        self.game_config = TicTacToeConfig()

        # Fixed parameters
        self.fixed_params = {
            "minibatch_size": 16,
            "training_steps": 200,
            "min_replay_buffer_size": 50,
            "replay_buffer_size": 2000,
            "games_per_generation": 1,
            "optimizer": torch.optim.Adam,
            "learning_rate": 0.001,
            "adam_epsilon": 1e-8,
            "weight_decay": 0,
            "momentum": 0.9,
            "clipnorm": 10,
            "training_iterations": 1,
            "num_minibatches": 1,
            "n_step": 5,
            "discount_factor": 0.997,
            "per_alpha": 0.0,
            "per_beta": 0.0,
            "per_epsilon": 1e-6,
            "per_use_batch_weights": False,
            "per_use_initial_max_priority": False,
            "lstm_horizon_len": 5,
            "value_prefix": True,
            "reanalyze_tau": 1,
            "lr_ratio": 1000,
            "unroll_steps": 5,
            "reanalyze_ratio": 0.0,
            "projector_hidden_dim": 16,
            "predictor_hidden_dim": 16,
            "projector_output_dim": 16,
            "predictor_output_dim": 16,
            "root_dirichlet_alpha": 0.25,
            "root_exploration_fraction": 0.25,
            "residual_layers": [(16, 3, 1)] * 2,
            "conv_layers": [(16, 3, 1)],
            "dense_layer_widths": [],
            "world_model_cls": MuzeroWorldModel,
            "multi_process": True,
            "transfer_interval": 50,
        }

    def run_grid_search(self):
        # 1. Define Parameter Grid
        self.grid_params = {
            "num_workers": [1, 2, 4],  # [1, 2, 4, 8, 16],
            "search_batch_size": [1, 4, 8],  # [1, 2, 4, 8],
            "quantization_mode": ["none", "ptq", "qat"],
            "use_torch_compile": [False, True],
            "use_mixed_precision": [False, True],
            "num_simulations": [32],
        }

        keys, values = zip(*self.grid_params.items())
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

        results = []
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"muzero_benchmark_results_{timestamp}.csv"

        print(f"Starting Grid Search with {len(combinations)} configurations...")

        # Prepare CSV
        fieldnames = list(self.grid_params.keys()) + [
            "duration_seconds",
            "total_frames",
            "fps",
        ]
        with open(csv_filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

        try:
            for i, config_dict in enumerate(combinations):
                print(f"\n--- Run {i+1}/{len(combinations)} ---")
                print(f"Config: {config_dict}")

                # Map quantization_mode
                q_mode = config_dict.pop("quantization_mode")

                # Prepare full params
                run_params = self.fixed_params.copy()
                run_params.update(config_dict)

                # Handle Quantization
                run_params["qat"] = q_mode == "qat"
                run_params["use_quantization"] = q_mode == "ptq"

                # Handle Compile mapping
                run_params["compile"] = run_params.pop("use_torch_compile")

                # Ensure nice naming
                config_dict["quantization_mode"] = q_mode
                config_dict["use_torch_compile"] = run_params["compile"]

                # Run specific config
                metrics = self._run_single_config(run_params)

                # Record Result
                result_row = {**config_dict, **metrics}
                results.append(result_row)

                # Append to CSV immediately
                with open(csv_filename, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writerow(result_row)

        except KeyboardInterrupt:
            print("Benchmark Interrupted")
        finally:
            print("Shutting down Ray cluster...")
            ray.shutdown()

        self._print_summary(results)
        return csv_filename

    def _run_single_config(self, params):
        # Setup Config
        # We'll run for a fixed number of training steps for the benchmark
        # 100 steps should be enough to get a stable throughput measurement
        params["training_steps"] = 100

        config = MuZeroConfig(params, self.game_config)
        env = self.game_config.make_env()
        test_agents = [RandomAgent(), TicTacToeBestAgent()]

        # Instantiate Agent
        agent = MuZeroRay(
            env=env,
            config=config,
            name=f"bench_worker{params['num_workers']}",
            device=torch.device("cpu"),
            test_agents=test_agents,
        )

        # Disable heavy testing/checkpointing for benchmark
        agent.test_interval = 1000000
        agent.checkpoint_interval = 1000000

        print(f"  Starting training for {params['training_steps']} steps...")

        # Start timer
        start_time = time.time()

        # Run standard training loop
        agent.train()

        # End timer
        end_time = time.time()
        duration = end_time - start_time

        # Calculate Metrics
        # agent.training_step should be equal to params["training_steps"] if successful
        total_samples = agent.training_step * agent.config.minibatch_size
        fps = total_samples / duration if duration > 0 else 0

        print(
            f"  Benchmark complete: {fps:.2f} samples/sec ({total_samples} samples in {duration:.2f}s)"
        )

        # Final cleanup just in case (though train() calls shutdown())
        agent.shutdown()
        gc.collect()

        return {
            "duration_seconds": duration,
            "total_frames": total_samples,
            "fps": fps,
        }

    def _print_summary(self, results):
        print("\n=== GRID SEARCH SUMMARY ===")
        print(tabulate(results, headers="keys", tablefmt="pretty", floatfmt=".2f"))


def main():
    bench = Benchmark()
    csv_file = bench.run_grid_search()
    print(f"\nResults saved to {csv_file}")


if __name__ == "__main__":
    main()
