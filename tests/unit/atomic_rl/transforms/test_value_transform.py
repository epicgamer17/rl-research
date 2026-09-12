"""Unit tests for contractive value transformations."""

import pytest
import torch

from atomic_rl.transforms.value_transform import (
    compute_contractive_transform,
    compute_inverse_contractive_transform,
    compute_two_hot_distribution,
    compute_expected_value,
)


def test_contractive_transform_zero():
    """Zero must map exactly to zero in both forward and inverse transforms."""
    x = torch.tensor([0.0], dtype=torch.float32)
    y = compute_contractive_transform(x)
    assert y.item() == 0.0

    x_rec = compute_inverse_contractive_transform(y)
    assert x_rec.item() == 0.0


def test_contractive_transform_symmetry():
    """h(x) and h^{-1}(y) must be strictly odd functions: f(-x) == -f(x)."""
    x = torch.tensor([0.01, 0.5, 1.0, 5.0, 42.0, 300.0], dtype=torch.float32)
    y_pos = compute_contractive_transform(x)
    y_neg = compute_contractive_transform(-x)
    assert torch.allclose(y_pos, -y_neg, atol=1e-6)

    x_rec_pos = compute_inverse_contractive_transform(y_pos)
    x_rec_neg = compute_inverse_contractive_transform(y_neg)
    assert torch.allclose(x_rec_pos, -x_rec_neg, atol=1e-6)


def test_contractive_roundtrip_precision():
    """Roundtrip h^{-1}(h(x)) == x across orders of magnitude in float32 and float64."""
    # float32 test
    x32 = torch.tensor(
        [-500.0, -100.0, -10.0, -1.0, -0.1, 0.0, 0.1, 1.0, 10.0, 100.0, 500.0],
        dtype=torch.float32,
    )
    y32 = compute_contractive_transform(x32)
    x32_rec = compute_inverse_contractive_transform(y32)
    assert torch.allclose(x32, x32_rec, rtol=1e-4, atol=1e-4)

    # float64 high-precision test
    x64 = x32.double()
    y64 = compute_contractive_transform(x64)
    x64_rec = compute_inverse_contractive_transform(y64)
    assert torch.allclose(x64, x64_rec, rtol=1e-10, atol=1e-10)


def test_contractive_multidimensional_shapes():
    """Supports arbitrary batch and trailing channel dimensions."""
    x = torch.randn(4, 5, 2, dtype=torch.float32) * 50.0
    y = compute_contractive_transform(x)
    assert y.shape == (4, 5, 2)
    assert y.dtype == torch.float32

    x_rec = compute_inverse_contractive_transform(y)
    assert x_rec.shape == (4, 5, 2)
    assert torch.allclose(x, x_rec, rtol=1e-4, atol=1e-4)


def test_contractive_gradients():
    """Ensures gradients propagate smoothly through both transforms."""
    x = torch.tensor([-10.0, 0.0, 10.0], dtype=torch.float32, requires_grad=True)
    y = compute_contractive_transform(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    assert (x.grad > 0).all()  # h(x) is strictly monotonically increasing


def test_contractive_fail_fast():
    """Ensures non-tensor, non-floating, or negative epsilon raise AssertionError."""
    with pytest.raises(AssertionError):
        compute_contractive_transform([1.0, 2.0])  # type: ignore[arg-type]

    with pytest.raises(AssertionError):
        compute_contractive_transform(torch.tensor([1, 2, 3], dtype=torch.long))

    with pytest.raises(AssertionError):
        compute_contractive_transform(torch.tensor([1.0]), epsilon=-0.01)


def test_two_hot_distribution_fractional():
    """Verify linear interpolation on fractional values (e.g. 3.7 -> 0.3 on 3, 0.7 on 4)."""
    from atomic_rl.transforms.value_transform import compute_two_hot_distribution

    scalar = torch.tensor([3.7], dtype=torch.float32)
    dist = compute_two_hot_distribution(scalar, support_min=-300, support_max=300)

    assert dist.shape == (1, 601)
    assert torch.allclose(dist.sum(dim=-1), torch.tensor([1.0]))

    # Support -300..300: value 3 is index 303, value 4 is index 304
    assert torch.isclose(dist[0, 303], torch.tensor(0.3))
    assert torch.isclose(dist[0, 304], torch.tensor(0.7))
    assert dist[0].count_nonzero().item() == 2

    # Expectation equals original scalar
    support = torch.arange(-300, 301, dtype=torch.float32)
    expected_val = (dist[0] * support).sum()
    assert torch.isclose(expected_val, torch.tensor(3.7))


def test_two_hot_distribution_integer():
    """Exact integers must place 100% of probability mass on that single atom."""
    from atomic_rl.transforms.value_transform import compute_two_hot_distribution

    scalar = torch.tensor([3.0, -5.0, 0.0], dtype=torch.float32)
    dist = compute_two_hot_distribution(scalar, support_min=-300, support_max=300)

    assert torch.allclose(dist.sum(dim=-1), torch.ones(3))
    assert torch.isclose(dist[0, 303], torch.tensor(1.0))
    assert torch.isclose(dist[1, 295], torch.tensor(1.0))
    assert torch.isclose(dist[2, 300], torch.tensor(1.0))


def test_two_hot_distribution_clamping():
    """Values exceeding support bounds are clamped to min/max atom."""
    from atomic_rl.transforms.value_transform import compute_two_hot_distribution

    scalar = torch.tensor([400.0, -400.0], dtype=torch.float32)
    dist = compute_two_hot_distribution(scalar, support_min=-10, support_max=10)

    assert dist.shape == (2, 21)
    assert torch.isclose(dist[0, 20], torch.tensor(1.0))  # Max atom index (10)
    assert torch.isclose(dist[1, 0], torch.tensor(1.0))   # Min atom index (-10)


def test_two_hot_distribution_multidimensional():
    """Supports arbitrary batch dimensions [*Batch, *Channels, num_atoms]."""
    from atomic_rl.transforms.value_transform import compute_two_hot_distribution

    scalar = torch.randn(8, 4, 1, dtype=torch.float32) * 20.0
    dist = compute_two_hot_distribution(scalar, support_min=-50, support_max=50)

    assert dist.shape == (8, 4, 1, 101)
    assert torch.allclose(dist.sum(dim=-1), torch.ones(8, 4, 1))

    # Verify expected values match clamped inputs
    support = torch.arange(-50, 51, dtype=torch.float32)
    expected_vals = (dist * support).sum(dim=-1)
    clamped_scalar = torch.clamp(scalar, -50.0, 50.0)
    assert torch.allclose(expected_vals, clamped_scalar, atol=1e-5)


def test_expected_value_logits():
    """Test compute_expected_value from unnormalized logits."""
    predictions = torch.tensor([[[0.0, 100.0], [100.0, 0.0]]])  # [1, 2, 2]
    support = torch.tensor([[[-1.0, 1.0], [-1.0, 1.0]]])

    values = compute_expected_value(predictions, support)
    assert values.shape == (1, 2)
    torch.testing.assert_close(values, torch.tensor([[1.0, -1.0]]))

    # 1D support broadcast
    support_1d = torch.tensor([-1.0, 1.0])
    values_1d = compute_expected_value(predictions, support_1d)
    torch.testing.assert_close(values_1d, torch.tensor([[1.0, -1.0]]))


def test_expected_value_probs():
    """Test compute_expected_value directly with normalized probabilities (probs=True)."""
    probs = torch.tensor([[0.2, 0.8], [0.5, 0.5]])
    support = torch.tensor([-1.0, 1.0])

    # [0.2*-1 + 0.8*1, 0.5*-1 + 0.5*1] = [0.6, 0.0]
    values = compute_expected_value(probs, support, probs=True)
    torch.testing.assert_close(values, torch.tensor([0.6, 0.0]))


def test_expected_value_assertions():
    """Verify shape assertions in compute_expected_value."""
    with pytest.raises(AssertionError, match="Atoms dimension mismatch"):
        compute_expected_value(torch.randn(2, 3), torch.randn(2, 4))
    with pytest.raises(AssertionError, match="If support is not 1D"):
        compute_expected_value(torch.randn(1, 2, 3), torch.randn(2, 3))

