import pytest
import torch
import torch.nn.functional as F
from atomic_rl.transforms.quantization import compute_straight_through_one_hot

pytestmark = pytest.mark.unit


def test_straight_through_one_hot_forward():
    """Verify forward pass matches hard one-hot of the argmax."""
    logits = torch.tensor([
        [0.1, 2.5, -1.0, 0.4],
        [1.2, -0.5, 3.8, 0.0],
    ])
    st, indices = compute_straight_through_one_hot(logits, dim=-1)

    expected_indices = torch.tensor([1, 2])
    torch.testing.assert_close(indices, expected_indices)

    expected_one_hot = torch.tensor([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ])
    torch.testing.assert_close(st, expected_one_hot)


def test_straight_through_one_hot_gradient():
    """Verify gradient flows straight through unperturbed (identity Jacobian)."""
    logits = torch.randn(4, 5, requires_grad=True)
    st, indices = compute_straight_through_one_hot(logits, dim=-1)

    target = torch.randn_like(st)
    loss = (st * target).sum()
    loss.backward()

    # dLoss/d(logits) == dLoss/d(st) * d(st)/d(logits) == target * 1.0
    torch.testing.assert_close(logits.grad, target)


def test_straight_through_one_hot_arbitrary_dim():
    """Verify compute_straight_through_one_hot works when dim is not -1."""
    # Shape: [Batch=2, NumClasses=4, Channel=3]
    logits = torch.randn(2, 4, 3, requires_grad=True)
    st, indices = compute_straight_through_one_hot(logits, dim=1)

    assert st.shape == logits.shape
    assert indices.shape == (2, 3)

    loss = st.sum()
    loss.backward()
    assert logits.grad.shape == logits.shape
