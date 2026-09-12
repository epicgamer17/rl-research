import pytest
import torch
from atomic_rl.evaluation import compute_ope_weights

pytestmark = pytest.mark.unit


def test_compute_ope_weights_stub():
    target_lp = torch.tensor([[0.0, 0.0, 0.0]])
    behavior_lp = torch.tensor([[-0.5, -0.5, -0.5]])

    with pytest.raises(NotImplementedError, match="pending"):
        compute_ope_weights(target_lp, behavior_lp)


def test_compute_ope_weights_shape_mismatch():
    with pytest.raises(AssertionError, match="Exact shape mismatch"):
        compute_ope_weights(torch.zeros(1, 2), torch.zeros(1, 3))
