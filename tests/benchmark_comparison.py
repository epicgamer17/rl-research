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
from agents.muzero_tmp import MuZeroAgent as MuZeroTorchMP
from modules.world_models.muzero_world_model import MuzeroWorldModel


class Benchmark:
    def __init__(self):
        self.device = torch.device("cpu")  # User specified CPU in snippet
        if torch.cuda.is_available():
            print(
                "Warning: CUDA available but using CPU as per user snippet request (params_batched['device']='cpu')"
            )

        self.game_config = TicTacToeConfig()

        # Base Params from User Snippet
        self.base_params = {
            # "num_workers": 4, # Will bet set in loop
            "search_batch_size": 1,
            "use_virtual_mean": True,
            "use_mixed_precision": True,
            "compile": True,  # Mapped from 'use_torch_compile'
            "use_quantization": True,
            "qat": True,
            "transfer_interval": 100,
            # Likely needed based on MuZeroConfig requirements
            "world_model_cls": MuzeroWorldModel,
            # (Note: MuZeroConfig expects the class object or string?
            # Looking at muzero_config.py: self.world_model_cls = self.parse_field("world_model_cls", None, required=True)
            # It usually comes from a dict where it might be a class or string.
            # Let's import the class to be safe.
            "minibatch_size": 8,  # Default/Guess
            "training_steps": 1000,  # User asked for ~1000 steps
            "min_replay_buffer_size": 100,
            "replay_buffer_size": 500,
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
            "lr_ratio": 10,  # Ensure learning triggers
            # MuZero specific
            "unroll_steps": 5,
            "reanalyze_ratio": 0.0,  # Disable reanalyze for pure train flux benchmark? Or keep default
            # Recurrent/MCTS
            "projector_hidden_dim": 16,
            "predictor_hidden_dim": 16,
            "projector_output_dim": 16,
            "predictor_output_dim": 16,
            "num_simulations": 10,  # Kept low for speed benchmark?
            "root_dirichlet_alpha": 0.25,
            "root_exploration_fraction": 0.25,
            # Architecture (Small for TicTacToe)
            "residual_layers": [(16, 3, 1)] * 2,
            "conv_layers": [(16, 3, 1)],
            "dense_layer_widths": [],
        }

    def run_benchmark(self, agent_cls, name, num_steps=1000):
        print(f"\n--- Benchmarking {name} ---")

        # 1. Setup Config
        params = self.base_params.copy()
        params["training_steps"] = num_steps
        # Increase lr_ratio to ensure we actually learn every step if possible,
        # or set it such that we learn reasonably often.
        # User snippet has standard behavior.

        # Ensure 'multi_process' is handled.
        # Ray agent expects this true usually to spawn workers.
        # TorchMP agent also uses it.
        params["multi_process"] = True
        params["num_workers"] = 4

        # Create Config Object
        config = MuZeroConfig(params, self.game_config)

        # Create Environment
        env = self.game_config.make_env()

        # Instantiate Agent
        test_agents = [RandomAgent(), TicTacToeBestAgent()]
        try:
            agent = agent_cls(
                env=env,
                config=config,
                name=f"bench_{name.lower().replace(' ', '_')}",
                device=torch.device("cpu"),
                test_agents=test_agents,
            )

            # Override testing to avoid slowdowns during benchmark
            agent.test_interval = 100000
            agent.checkpoint_interval = 100000

            # Run Training
            print(f"  Starting training for {num_steps} steps...")
            start_time = time.time()

            # The 'train()' method runs until agent.training_step >= config.training_steps
            # Ensure agent.training_step starts at 0
            agent.training_step = 0
            agent.train()

            end_time = time.time()
            duration = end_time - start_time
            print(f"  Finished. Time: {duration:.2f}s")
            return duration

        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback

            traceback.print_exc()
            return None


def main():
    bench = Benchmark()
    results = []

    # 1. Ray Benchmark
    # ray_time = bench.run_benchmark(MuZeroRay, "MuZero (Ray)", num_steps=1000)
    # results.append({"Agent": "MuZero (Ray)", "Time (s)": ray_time})

    # 2. TorchMP Benchmark
    mp_time = bench.run_benchmark(MuZeroTorchMP, "MuZero (TorchMP)", num_steps=1000)
    results.append({"Agent": "MuZero (TorchMP)", "Time (s)": mp_time})

    # 3. Ray Benchmark (Running second to avoid Ray init issues impacting TorchMP if any, though Ray usually robust)
    # Actually, Ray init might persist. Let's run Ray second or handle shutdown.
    # MuZeroRay.train() handles ray.init() check.

    ray_time = bench.run_benchmark(MuZeroRay, "MuZero (Ray)", num_steps=1000)
    results.append({"Agent": "MuZero (Ray)", "Time (s)": ray_time})

    print("\n=== RESULTS ===")
    df = tabulate(results, headers="keys", tablefmt="pretty", floatfmt=".2f")
    print(df)

    if ray_time and mp_time:
        speedup = ray_time / mp_time
        print(f"\nTime Ratio (Ray / TorchMP): {speedup:.2f}x")
        if speedup > 1.0:
            print(f"TorchMP is {speedup:.2f}x FASTER than Ray")
        else:
            print(f"Ray is {1/speedup:.2f}x FASTER than TorchMP")


if __name__ == "__main__":
    main()
