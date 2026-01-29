import pytest
import ray
import torch
import sys
import os
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.muzero import SharedStorage
from replay_buffers.modular_buffer import ModularReplayBuffer
from replay_buffers.buffer_factories import create_muzero_buffer
from replay_buffers.game import Game


def test_shared_storage():
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    initial_weights = {"layer1": torch.randn(10, 10)}
    storage = SharedStorage.remote(step=0, weights=initial_weights)

    # Test get_weights
    step, weights = ray.get(storage.get_weights.remote())
    assert step == 0
    assert torch.equal(weights["layer1"], initial_weights["layer1"])

    # Test set_weights
    new_weights = {"layer1": torch.randn(10, 10)}
    ray.get(storage.set_weights.remote(step=10, weights=new_weights))

    step, weights = ray.get(storage.get_weights.remote())
    assert step == 10
    assert torch.equal(weights["layer1"], new_weights["layer1"])

    ray.shutdown()


def test_modular_replay_buffer_ray():
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    # Define remote class
    RemoteBuffer = ray.remote(ModularReplayBuffer)

    obs_dim = (3, 64, 64)

    # Use the updated factory to create the remote actor!
    # passing class_fn=RemoteBuffer.remote will call RemoteBuffer.remote(...) inside the factory
    buffer = create_muzero_buffer(
        observation_dimensions=obs_dim,
        max_size=100,
        num_actions=5,
        num_players=1,
        unroll_steps=5,
        n_step=10,
        gamma=0.99,
        class_fn=RemoteBuffer.remote,
    )

    # Check initial size
    size = ray.get(buffer.get_size.remote())
    assert size == 0

    # Create a dummy game object
    game = Game(num_players=1)

    # Add some dummy data to the game
    num_steps = 10

    for _ in range(num_steps):
        game.observation_history.append(np.random.randn(*obs_dim).astype(np.float32))
        game.action_history.append(0)
        game.rewards.append(0.0)
        game.policy_history.append(torch.ones(5) / 5)
        game.value_history.append(0.0)
        game.info_history.append(
            {"player": 0, "chance": 0, "legal_moves": [0, 1, 2, 3, 4], "done": False}
        )

    # Add terminal observation/state
    game.observation_history.append(np.random.randn(*obs_dim).astype(np.float32))
    game.info_history.append(
        {"player": 0, "chance": 0, "legal_moves": [], "done": True}
    )

    # Test store_aggregate (replaces save_game)
    ray.get(buffer.store_aggregate.remote(game))

    size = ray.get(buffer.get_size.remote())
    assert size == num_steps + 1

    for _ in range(35):
        ray.get(buffer.store_aggregate.remote(game))

    size = ray.get(buffer.get_size.remote())
    # Size is capped at max_size
    assert size == 100
    # Try sampling
    batch = ray.get(buffer.sample.remote())
    assert batch is not None
    assert "observations" in batch
    assert "game_ids" in batch
    assert "ids" in batch

    # Verify IDs are populated (not all zeros)
    # Since we added 36 games (1 initial + 35 loop), we should see game_ids > 0
    # And ids should be > 0 (total steps = 11 per game * 36 games = 396 steps pushed)
    # Buffer size capped at 100, so ids will be in range [296, 396] roughly (depending on circular writes)
    assert batch["game_ids"].max().item() > 0
    assert batch["ids"].max().item() > 0

    # Test beta access
    # Get initial beta
    beta = ray.get(buffer.get_beta.remote())
    assert isinstance(beta, float)

    # Test setting beta (if method exists, otherwise skip as ModularReplayBuffer usually has set_beta but we need to check if exposed)
    # create_muzero_buffer creates ModularReplayBuffer which has set_beta.
    # We exposed get_beta. set_beta is inherited/present.
    new_beta = 0.8
    ray.get(buffer.set_beta.remote(new_beta))
    updated_beta = ray.get(buffer.get_beta.remote())
    # Allow for small float diffs or exact match
    assert abs(updated_beta - new_beta) < 1e-6

    ray.shutdown()


if __name__ == "__main__":
    test_shared_storage()
    test_modular_replay_buffer_ray()
    print("All tests passed!")
