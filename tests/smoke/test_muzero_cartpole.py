"""Smoke tests for MuZero CartPole network components, MCTS search, and loss computation."""

import pytest
import torch

from atomic_rl.losses import compute_cross_entropy_loss
from atomic_rl.transforms import (
    compute_contractive_transform,
    compute_inverse_contractive_transform,
    compute_two_hot_distribution,
)
from examples.muzero.muzero_cartpole import (
    CartPoleMuZeroNet,
    EPSILON_TRANSFORM,
    LATENT_DIM,
    NUM_ACTIONS,
    OBS_DIM,
    SUPPORT_MAX,
    SUPPORT_MIN,
    create_recurrent_fn,
    run_batched_mcts,
)

pytestmark = pytest.mark.smoke


def test_cartpole_muzero_net_shapes():
    net = CartPoleMuZeroNet(
        obs_dim=OBS_DIM,
        num_actions=NUM_ACTIONS,
        latent_dim=LATENT_DIM,
        support_min=SUPPORT_MIN,
        support_max=SUPPORT_MAX,
    )
    batch_size = 4
    num_atoms = SUPPORT_MAX - SUPPORT_MIN + 1
    obs = torch.randn((batch_size, OBS_DIM), dtype=torch.float32)

    # 1. Initial inference
    s0, policy_logits, scalar_value, value_logits = net.initial_inference(obs)
    assert s0.shape == (batch_size, LATENT_DIM)
    assert policy_logits.shape == (batch_size, NUM_ACTIONS)
    assert scalar_value.shape == (batch_size,)
    assert value_logits.shape == (batch_size, num_atoms)

    # Min-max normalization checks on hidden state: in [0, 1]
    assert s0.min() >= 0.0 - 1e-6
    assert s0.max() <= 1.0 + 1e-6

    # 2. Recurrent inference
    actions = torch.tensor([0, 1, 1, 0], dtype=torch.long)
    (
        next_s,
        scalar_reward,
        r_policy_logits,
        r_scalar_value,
        reward_logits,
        r_value_logits,
    ) = net.recurrent_inference(s0, actions)

    assert next_s.shape == (batch_size, LATENT_DIM)
    assert scalar_reward.shape == (batch_size,)
    assert r_policy_logits.shape == (batch_size, NUM_ACTIONS)
    assert r_scalar_value.shape == (batch_size,)
    assert reward_logits.shape == (batch_size, num_atoms)
    assert r_value_logits.shape == (batch_size, num_atoms)


def test_cartpole_muzero_categorical_decoding():
    net = CartPoleMuZeroNet(
        obs_dim=OBS_DIM,
        num_actions=NUM_ACTIONS,
        latent_dim=LATENT_DIM,
        support_min=SUPPORT_MIN,
        support_max=SUPPORT_MAX,
    )
    num_atoms = SUPPORT_MAX - SUPPORT_MIN + 1

    # Zero return test: atom index where support == 0
    zero_idx = 0 - SUPPORT_MIN
    logits_zero = torch.full((1, num_atoms), -100.0)
    logits_zero[0, zero_idx] = 100.0
    decoded_zero = net.decode_support_to_scalar(logits_zero)
    assert torch.isclose(decoded_zero, torch.tensor([0.0]), atol=1e-4)

    # Positive return test: support == 5
    target_atom = 5.0
    atom_idx = int(target_atom - SUPPORT_MIN)
    logits_five = torch.full((1, num_atoms), -100.0)
    logits_five[0, atom_idx] = 100.0
    decoded_val = net.decode_support_to_scalar(logits_five)
    expected_scalar = compute_inverse_contractive_transform(
        torch.tensor([target_atom]), epsilon=EPSILON_TRANSFORM
    )
    assert torch.isclose(decoded_val, expected_scalar, atol=1e-3)


def test_cartpole_muzero_batched_mcts():
    net = CartPoleMuZeroNet(
        obs_dim=OBS_DIM,
        num_actions=NUM_ACTIONS,
        latent_dim=LATENT_DIM,
        support_min=SUPPORT_MIN,
        support_max=SUPPORT_MAX,
    )
    net.eval()
    recurrent_fn = create_recurrent_fn(net)

    batch_size = 3
    root_embed = torch.randn((batch_size, OBS_DIM), dtype=torch.float32)

    action_probs, root_values = run_batched_mcts(
        model=net,
        recurrent_fn=recurrent_fn,
        root_embed=root_embed,
        num_simulations=5,
        temperature=1.0,
        explore_dirichlet=False,
    )

    assert action_probs.shape == (batch_size, NUM_ACTIONS)
    assert root_values.shape == (batch_size,)
    assert torch.allclose(action_probs.sum(dim=-1), torch.ones(batch_size), atol=1e-5)
    assert (action_probs >= 0.0).all()


def test_cartpole_muzero_loss_backward():
    net = CartPoleMuZeroNet(
        obs_dim=OBS_DIM,
        num_actions=NUM_ACTIONS,
        latent_dim=LATENT_DIM,
        support_min=SUPPORT_MIN,
        support_max=SUPPORT_MAX,
    )
    net.train()

    batch_size = 2
    obs = torch.randn((batch_size, OBS_DIM), dtype=torch.float32)
    s0, p0, _, v0_logits = net.initial_inference(obs)

    target_policy = torch.tensor([[0.7, 0.3], [0.1, 0.9]], dtype=torch.float32)
    target_value = torch.tensor([10.0, 25.0], dtype=torch.float32)
    target_reward = torch.tensor([1.0, 1.0], dtype=torch.float32)

    # 1. Step 0 losses
    raw_p_loss_0, _ = compute_cross_entropy_loss(p0, target_policy)
    z0_scaled = compute_contractive_transform(target_value, epsilon=EPSILON_TRANSFORM)
    z0_two_hot = compute_two_hot_distribution(z0_scaled, SUPPORT_MIN, SUPPORT_MAX)
    raw_v_loss_0, _ = compute_cross_entropy_loss(v0_logits, z0_two_hot)

    loss = raw_p_loss_0.mean() + 0.25 * raw_v_loss_0.mean()

    # 2. Recurrent step
    actions = torch.tensor([0, 1], dtype=torch.long)
    next_s, _, p1, _, r1_logits, v1_logits = net.recurrent_inference(s0, actions)

    raw_p_loss_1, _ = compute_cross_entropy_loss(p1, target_policy)
    raw_v_loss_1, _ = compute_cross_entropy_loss(v1_logits, z0_two_hot)
    u_scaled = compute_contractive_transform(target_reward, epsilon=EPSILON_TRANSFORM)
    u_two_hot = compute_two_hot_distribution(u_scaled, SUPPORT_MIN, SUPPORT_MAX)
    raw_r_loss_1, _ = compute_cross_entropy_loss(r1_logits, u_two_hot)

    loss = loss + raw_p_loss_1.mean() + 0.25 * raw_v_loss_1.mean() + raw_r_loss_1.mean()
    loss.backward()

    # Verify gradients computed across all subnetworks
    for name, param in net.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient."
