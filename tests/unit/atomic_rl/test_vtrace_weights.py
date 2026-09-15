import pytest
import torch
from atomic_rl.losses import compute_vtrace_weights
from atomic_rl.returns import compute_vtrace_returns

pytestmark = pytest.mark.unit


def test_compute_vtrace_weights_stub():
    target_lp = torch.tensor([[0.0, -1.0]])
    behavior_lp = torch.tensor([[-0.5, -1.0]])

    with pytest.raises(NotImplementedError, match="pending"):
        compute_vtrace_weights(target_lp, behavior_lp)


def test_compute_vtrace_weights_shape_mismatch():
    with pytest.raises(AssertionError, match="Exact shape mismatch"):
        compute_vtrace_weights(torch.zeros(2, 3), torch.zeros(2, 4))


def test_compute_vtrace_returns_stub():
    zeros = torch.zeros(1, 2)
    with pytest.raises(NotImplementedError, match="pending"):
        compute_vtrace_returns(
            zeros, zeros, zeros, zeros, zeros, zeros, zeros, gamma=0.99
        )
