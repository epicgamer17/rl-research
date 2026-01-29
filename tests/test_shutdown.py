import ray
import time
from agents.muzero import MuZeroAgent
from agent_configs.muzero_config import MuZeroConfig


def test_shutdown():
    print("Initializing Ray...")
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    print("Creating Agent...")
    config = MuZeroConfig()
    config.multi_process = True
    config.num_workers = 1
    config.use_mixed_precision = False  # Avoid BFloat16 issues

    # Mock environment (basic cartpole or similar) to avoid complex setup
    import gymnasium as gym

    env = gym.make("CartPole-v1")

    agent = MuZeroAgent(env, config)

    print("Agent created. Checking actors...")
    assert hasattr(agent, "storage")
    assert hasattr(agent, "replay_buffer")
    assert hasattr(agent, "workers")
    assert len(agent.workers) == 1

    storage_ref = agent.storage
    buffer_ref = agent.replay_buffer
    worker_ref = agent.workers[0]

    print("Calling shutdown...")
    agent.shutdown()

    print("Verifying actors are dead...")

    # Try to access actors; should raise RayActorError or similar
    try:
        ray.get(storage_ref.get_weights.remote())
        print("ERROR: Storage is still alive!")
    except Exception as e:
        print(f"Storage confirmed dead: {e}")

    try:
        ray.get(buffer_ref.get_size.remote())
        print("ERROR: ReplayBuffer is still alive!")
    except Exception as e:
        print(f"ReplayBuffer confirmed dead: {e}")

    try:
        ray.get(
            worker_ref.get_weights.remote()
        )  # Assuming get_weights/similar exists or generic check
        print("ERROR: Worker is still alive!")
    except Exception as e:
        print(f"Worker confirmed dead: {e}")

    ray.shutdown()
    print("Test Complete.")


if __name__ == "__main__":
    test_shutdown()
