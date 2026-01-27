import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import torch
import numpy as np
from agents.ppo import PPOAgent
from agent_configs.ppo_config import PPOConfig
from agent_configs.actor_config import ActorConfig
from agent_configs.critic_config import CriticConfig
from game_configs.cartpole_config import CartPoleConfig


def run_benchmark(mixed_precision: bool, iterations: int = 20):
    print(f"Benchmarking with Mixed Precision: {mixed_precision}")

    # Setup Configs with larger network to make compute bound
    game_config = CartPoleConfig()
    env = game_config.make_env()

    # Larger dense layers to simulate load
    actor_config = ActorConfig(
        {
            "learning_rate": 0.0003,
            "dense_layer_widths": [256, 256, 256],
            "clipnorm": 0.5,
        }
    )
    critic_config = CriticConfig(
        {
            "learning_rate": 0.0003,
            "dense_layer_widths": [256, 256, 256],
            "clipnorm": 0.5,
        }
    )

    config_dict = {
        "use_mixed_precision": mixed_precision,
        "training_steps": iterations,
        "steps_per_epoch": 100,  # Large batch to utilize heavy math
        "train_policy_iterations": 5,  # Multiple updates per step
        "train_value_iterations": 5,
        "multi_process": False,
        "num_minibatches": 1,
    }

    config = PPOConfig(config_dict, game_config, actor_config, critic_config)

    # Force CPU (since user is on Mac and likely testing CPU/MPS flow, but let's stick to what PPOAgent selects or force it for consistency)
    # The user request mentioned Mac. PPOAgent defaults to CUDA -> MPS -> CPU.
    # Let's check what device is being used.
    # We will initialize and check.

    agent = PPOAgent(env, config)
    print(f"Device: {agent.device}")

    # Pre-fill buffer to avoid measuring environment step overhead too much
    # We want to measure LEARNING speed.
    # However, PPO alternates.
    # Let's just run the train loop. The overhead of environment should be constant.

    start_time = time.time()

    # We need to manually drive the loop to focus on learn() if possible, or just run agent.train()
    # agent.train() runs the full loop. Let's strictly measure agent.learn() calls.

    # Generate Dummy Batch
    obs_shape = env.observation_space.shape
    num_actions = env.action_space.n

    # Store data
    observations = torch.randn(100, *obs_shape)
    actions = torch.randint(0, num_actions, (100,))
    log_probs = torch.randn(100)
    rewards = torch.randn(100)
    values = torch.randn(100)

    # Fill buffer
    for i in range(100):
        agent.replay_buffer.store(
            observations=observations[i],
            info={"legal_moves": list(range(num_actions))},
            actions=actions[i],
            values=values[i],
            log_probabilities=log_probs[i],
            rewards=rewards[i],
        )
    agent.replay_buffer.finalize_trajectory(0.0)

    # Benchmark learn()
    start_learn = time.time()
    for _ in range(iterations):
        agent.learn()
        # Mock re-filling/re-using buffer isn't perfect because PPO clears it.
        # But we modified PPO to clear it!
        # So we need to refill it.
        # Refilling takes time.

        # Fast refill
        # We can just manually reset pointers?
        # agent.replay_buffer.writer.pointer = 200
        # agent.replay_buffer.writer.size = 200
        # agent.replay_buffer.writer.path_start_idx = 0
        # But finalize_trajectory might be needed.

        # Simpler approach: Just re-store quickly.
        for i in range(100):
            agent.replay_buffer.store(
                observations=observations[i],
                info={"legal_moves": list(range(num_actions))},
                actions=actions[i],
                values=values[i],
                log_probabilities=log_probs[i],
                rewards=rewards[i],
            )
        agent.replay_buffer.finalize_trajectory(0.0)

    end_time = time.time()
    duration = end_time - start_learn

    print(f"Time for {iterations} iterations: {duration:.4f}s")
    print(f"Steps/sec: {iterations / duration:.2f}")

    return duration


if __name__ == "__main__":
    t_fp32 = run_benchmark(mixed_precision=False, iterations=10)
    t_amp = run_benchmark(mixed_precision=True, iterations=10)

    print("\nXXX RESULTS XXX")
    print(f"FP32 Time: {t_fp32:.4f}s")
    print(f"AMP Time : {t_amp:.4f}s")
    print(f"Speedup  : {t_fp32 / t_amp:.2f}x")
