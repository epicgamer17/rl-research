import pytest
import torch

from atomic_rl.buffers.rollout import init_rollout_buffer, RolloutBufferState
from atomic_rl.buffers import slice_unrolls, init_rollout_buffer as init_rollout_buffer_top

pytestmark = pytest.mark.unit


def test_init_rollout_buffer():
    steps = 10
    num_envs = 4
    shapes = {"obs": (8,), "action": ()}
    device = "cpu"

    buffer = init_rollout_buffer(steps, num_envs, shapes, device=device)

    assert buffer.data["obs"].shape == (num_envs, steps, 8)
    assert buffer.data["action"].shape == (num_envs, steps)
    assert buffer.data["action"].dtype == torch.long
    assert len(buffer.truncation_records) == 0
    assert buffer.storage_shape == (num_envs, steps)


def test_init_rollout_buffer_capacity_tuple():
    shapes = {"obs": (4,), "action": ()}
    buffer = init_rollout_buffer_top((8, 20), shapes, device="cpu")

    assert buffer.data["obs"].shape == (8, 20, 4)
    assert buffer.data["action"].shape == (8, 20)
    assert buffer.storage_shape == (8, 20)


def test_rollout_buffer_slice_unrolls():
    shapes = {"obs": (2,), "reward": ()}
    buffer = init_rollout_buffer((4, 10), shapes, device="cpu")

    # Populate some dummy data
    for env in range(4):
        for step in range(10):
            buffer.data["obs"][env, step] = torch.tensor([float(env), float(step)])
            buffer.data["reward"][env, step] = float(env * 10 + step)

    # Sample coords: env 1 at step 2, and env 3 at step 5
    coords = (torch.tensor([1, 3]), torch.tensor([2, 5]))
    valid_lengths = torch.full((4,), 10, dtype=torch.long)
    unrolls, mask = slice_unrolls(buffer.data, coords, unroll_steps=3, valid_lengths=valid_lengths)

    assert unrolls.batch_size == torch.Size([2, 4])
    assert mask.all()
    # For sample 0 (env 1, step 2): steps 2, 3, 4, 5
    torch.testing.assert_close(unrolls["reward"][0], torch.tensor([12.0, 13.0, 14.0, 15.0]))
    # For sample 1 (env 3, step 5): steps 5, 6, 7, 8
    torch.testing.assert_close(unrolls["reward"][1], torch.tensor([35.0, 36.0, 37.0, 38.0]))

