import pytest
import torch
from atomic_rl.networks import AutoregressivePolicyHead
from atomic_rl.networks.architectures import AutoregressivePolicyHead as ArchAutoregressivePolicyHead


def test_autoregressive_policy_head_export():
    assert AutoregressivePolicyHead is ArchAutoregressivePolicyHead


def test_autoregressive_policy_head_stub_init():
    head = AutoregressivePolicyHead(
        latent_dim=64,
        action_dims=[6, 6, 6, 6],
        hidden_dim=64,
        embed_dim=32,
    )
    assert head.latent_dim == 64
    assert head.action_dims == [6, 6, 6, 6]
    assert head.num_steps == 4

    state = torch.randn(2, 64)
    actions = torch.zeros(2, 4, dtype=torch.long)

    with pytest.raises(NotImplementedError):
        head(state, actions)

    with pytest.raises(NotImplementedError):
        head.sample(state)
