import torch
import torch.multiprocessing as mp
import time
import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.muzero import MuZeroAgent
from agent_configs.muzero_config import MuZeroConfig
from game_configs.cartpole_config import CartPoleConfig
from modules.world_models.muzero_world_model import MuzeroWorldModel
from losses.basic_losses import CategoricalCrossentropyLoss


def mocked_run_worker(worker_id, config, model_queue, *args, **kwargs):
    print(f"[Worker {worker_id}] Mocked worker started.")
    # Extract received_counts from kwargs or args if we pass it that way
    # In this test, we'll use a hack: access it via config or a shared dict
    received_counts = config.received_counts
    # Wait for models
    model_count = 0
    while True:
        try:
            # Use a timeout so we don't hang forever if broadcasting fails
            model = model_queue.get(timeout=5)
            model_count += 1
            received_counts[worker_id] = model_count
            print(f"[Worker {worker_id}] Received model {model_count}")

            # Check if we should stop (we expect 3 models in this test)
            if model_count >= 3:
                break
        except Exception as e:
            print(f"[Worker {worker_id}] Error or timeout: {e}")
            break
    print(f"[Worker {worker_id}] Mocked worker finishing.")


def action_as_onehot(action, num_actions):
    one_hot = torch.zeros(num_actions)
    one_hot[action] = 1.0
    return one_hot


def verify_broadcasting():
    print("Starting Model Broadcasting Verification...")
    manager = mp.Manager()
    received_counts = manager.list([0, 0])

    game_config = CartPoleConfig()
    env = game_config.make_env()

    config_dict = {
        "world_model_cls": MuzeroWorldModel,
        "num_simulations": 1,
        "minibatch_size": 1,
        "min_replay_buffer_size": 1,
        "replay_buffer_size": 10,
        "unroll_steps": 1,
        "n_step": 1,
        "multi_process": True,
        "num_workers": 2,
        "training_steps": 1,
        "games_per_generation": 1,
        "learning_rate": 0.001,
        "action_function": action_as_onehot,
        "value_loss_function": CategoricalCrossentropyLoss(),
        "reward_loss_function": CategoricalCrossentropyLoss(),
        "policy_loss_function": CategoricalCrossentropyLoss(),
        "support_range": 31,
        "residual_layers": [],
        "conv_layers": [],
        "representation_residual_layers": [],
        "representation_conv_layers": [],
        "dynamics_residual_layers": [],
        "dynamics_conv_layers": [],
        "reward_conv_layers": [],
        "to_play_conv_layers": [],
        "critic_conv_layers": [],
        "actor_conv_layers": [],
        "dense_layer_widths": [64],
        "representation_dense_layer_widths": [64],
        "dynamics_dense_layer_widths": [64],
    }

    config = MuZeroConfig(config_dict, game_config)
    config.received_counts = received_counts

    # We need to monkey patch run_worker to verify it receives multiple models
    original_run_worker = MuZeroAgent.run_worker

    MuZeroAgent.run_worker = staticmethod(mocked_run_worker)

    agent = MuZeroAgent(env, config, name="verify_broadcasting", device="cpu")

    # Start training (which starts workers)
    # We want to trigger update_target_model multiple times.
    # MuZeroAgent.train starts workers and then calls update_target_model once initially.

    # To test broadcasting, we'll manually trigger it after workers are started.
    # However, train() is a blocking call. We might need to run it in a thread or just test the pieces.

    print("Manually triggering updates...")
    # Workers are started in train(), but we can simulate it here.

    # Start workers manually for more control
    stats_client = agent.stats.get_client()
    error_queue = mp.Queue()
    workers = [
        mp.Process(
            target=MuZeroAgent.run_worker,
            args=(
                i,
                agent.config,
                agent.update_queues[i],
                agent.replay_buffer,
                agent.device,
                agent.stop_flag,
                stats_client,
                error_queue,
                agent.model_name,
                agent.checkpoint_interval,
                agent.num_actions,
                agent._training_step,
            ),
        )
        for i in range(agent.config.num_workers)
    ]
    for w in workers:
        w.start()

    time.sleep(1)  # Wait for workers to be ready

    # Send 3 updates
    for i in range(3):
        print(f"Broadcasting update {i+1}...")
        agent.update_target_model()
        time.sleep(0.5)

    # Wait for workers to finish
    for w in workers:
        w.join(timeout=10)
        if w.is_alive():
            print(f"Worker {w.pid} timed out, terminating.")
            w.terminate()

    print(f"Final received counts: {list(received_counts)}")

    assert list(received_counts) == [
        3,
        3,
    ], f"Expected [3, 3], got {list(received_counts)}"
    print("Verification Successful!")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    verify_broadcasting()
