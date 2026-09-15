import pytest
import torch
from atomic_rl.td import compute_td_error

pytestmark = pytest.mark.unit


def test_compute_td_error_1d():
    predictions = torch.tensor([1.0, 2.0, 3.0])
    targets = torch.tensor([1.5, 2.0, 2.5])

    # delta = targets - predictions
    td_error = compute_td_error(predictions, targets)
    expected = torch.tensor([0.5, 0.0, -0.5])
    torch.testing.assert_close(td_error, expected)

    # Abs error for PER priorities
    priorities = torch.abs(td_error)
    expected_priorities = torch.tensor([0.5, 0.0, 0.5])
    torch.testing.assert_close(priorities, expected_priorities)


def test_compute_td_error_multi_dim():
    predictions = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    targets = torch.tensor([[2.0, 2.0], [1.0, 5.0]])

    td_error = compute_td_error(predictions, targets)
    expected = torch.tensor([[1.0, 0.0], [-2.0, 1.0]])
    torch.testing.assert_close(td_error, expected)


def test_compute_td_error_shape_mismatch():
    predictions = torch.tensor([1.0, 2.0])
    targets = torch.tensor([[1.0, 2.0]])

    with pytest.raises(AssertionError, match="Exact shape mismatch"):
        compute_td_error(predictions, targets)
