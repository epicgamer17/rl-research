import pytest
import torch
import torch.nn as nn
import numpy as np

from atomic_rl.metrics import (
    compute_dead_units_proportion,
    compute_average_weight_magnitude,
    compute_average_gradient_magnitude,
    compute_stable_rank,
    compute_expected_value,
    compute_iqm,
    create_wandb_plasticity_metrics,
    create_wandb_explained_variance,
    create_wandb_episode_metrics,
)

pytestmark = pytest.mark.unit


# ==========================================
# Tests for compute_dead_units_proportion
# ==========================================


def test_dead_units_none_dead():
    # All activations are strictly positive -> 0% dead units
    activations = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    assert compute_dead_units_proportion(activations) == 0.0


def test_dead_units_all_dead():
    # All elements are 0 or negative -> 100% dead units
    activations = torch.tensor([[0.0, -1.0], [0.0, 0.0]])
    assert compute_dead_units_proportion(activations) == 1.0


def test_dead_units_partial():
    # Column 0 is dead (all <= 0), Column 1 is alive (has at least one > 0)
    activations = torch.tensor([[0.0, 2.0], [-0.5, 0.0]])
    # 1 out of 2 features is dead -> 0.5
    assert compute_dead_units_proportion(activations) == 0.5


# ==========================================
# Tests for compute_average_weight_magnitude
# ==========================================


def test_weight_magnitude_empty():
    # Empty iterable should safely return 0.0
    assert compute_average_weight_magnitude([]) == 0.0


def test_weight_magnitude_standard():
    p1 = nn.Parameter(torch.tensor([1.0, -2.0]))
    p2 = nn.Parameter(torch.tensor([3.0]))
    # Total absolute sum: 1.0 + 2.0 + 3.0 = 6.0
    # Total elements: 3
    # Expected: 6.0 / 3 = 2.0
    assert compute_average_weight_magnitude([p1, p2]) == pytest.approx(2.0)


# ==========================================
# Tests for compute_average_gradient_magnitude
# ==========================================


def test_gradient_magnitude_empty_or_none():
    # Case 1: Empty list
    assert compute_average_gradient_magnitude([]) == 0.0

    # Case 2: Parameters exist but their .grad fields are None
    p1 = nn.Parameter(torch.tensor([1.0, 2.0]))
    assert compute_average_gradient_magnitude([p1]) == 0.0


def test_gradient_magnitude_mixed():
    p1 = nn.Parameter(torch.tensor([1.0, 2.0]))
    p1.grad = torch.tensor([1.5, -2.5])

    p2 = nn.Parameter(torch.tensor([3.0]))  # No gradient populated

    # Only p1 should be counted. Total absolute grad sum = 1.5 + 2.5 = 4.0
    # Total graded elements = 2
    # Expected: 4.0 / 2 = 2.0
    assert compute_average_gradient_magnitude([p1, p2]) == pytest.approx(2.0)


# ==========================================
# Tests for compute_stable_rank
# ==========================================


def test_stable_rank_zero_variance():
    # When all activations are completely identical across the batch,
    # the centered matrix becomes zero, triggering the < 1e-8 safety check.
    activations = torch.tensor([[3.0, 3.0], [3.0, 3.0]])
    assert compute_stable_rank(activations) == 0.0


def test_stable_rank_rank_one():
    # Hand-calculating a controlled matrix:
    # Let activations be a 2x2 matrix where centering leaves a clear Rank-1 structure.
    activations = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    # Mean along dim 0 is [0.0, 0.0], so centered == activations.
    # Singular values of this matrix are sqrt(2) and 0.
    # Squared singular values = 2.0 and 0.0.
    # Sum = 2.0, Max = 2.0 -> Stable Rank = 2.0 / 2.0 = 1.0
    assert compute_stable_rank(activations) == pytest.approx(1.0)


def test_stable_rank_orthogonal():
    # For isotropic/orthogonal configurations, the dimensions should express higher rank
    activations = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
    # This yields well-distributed singular values across its dimensions.
    rank = compute_stable_rank(activations)
    assert rank > 1.0


# ==========================================
# Tests for compute_expected_value
# ==========================================


def test_compute_expected_value():
    probs = torch.tensor([[0.2, 0.5, 0.3], [0.1, 0.1, 0.8]])
    support = torch.tensor([1.0, 2.0, 3.0])

    # Batch 0: 0.2*1 + 0.5*2 + 0.3*3 = 0.2 + 1.0 + 0.9 = 2.1
    # Batch 1: 0.1*1 + 0.1*2 + 0.8*3 = 0.1 + 0.2 + 2.4 = 2.7
    expected = compute_expected_value(probs, support, probs=True)
    torch.testing.assert_close(expected, torch.tensor([2.1, 2.7]))

    with pytest.raises(AssertionError, match="Atoms dimension mismatch"):
        compute_expected_value(probs, torch.tensor([1.0, 2.0]))


# ==========================================
# Tests for compute_iqm
# ==========================================


def test_compute_iqm():
    # Array with clear outliers: [1, 2, 3, 4, 5, 6, 7, 8, 100]
    # IQM should discard bottom 25% and top 25%
    scores = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0])
    iqm = compute_iqm(scores)
    # 25th percentile is 30.0, 75th percentile is 70.0
    # Trimmed: [30, 40, 50, 60, 70] -> Mean = 50.0
    assert iqm == pytest.approx(50.0)

    # Tensor compatibility
    iqm_torch = compute_iqm(torch.tensor(scores))
    assert iqm_torch == pytest.approx(50.0)


# ==========================================
# Tests for W&B metric helpers
# ==========================================


def test_wandb_metric_helpers():
    p_dict = create_wandb_plasticity_metrics(10, dead_units_proportion=0.25, stable_rank=3.5)
    assert p_dict["step"] == 10
    assert p_dict["plasticity/dead_units_proportion"] == 0.25
    assert p_dict["plasticity/stable_rank"] == 3.5

    ev_dict = create_wandb_explained_variance(20, 0.88)
    assert ev_dict["step"] == 20
    assert ev_dict["metrics/explained_variance"] == 0.88

    ep_dict = create_wandb_episode_metrics(100, episode_return=250.5, episode_length=500, success_rate=1.0)
    assert ep_dict["step"] == 100
    assert ep_dict["charts/episodic_return"] == 250.5
    assert ep_dict["charts/episodic_length"] == 500
    assert ep_dict["charts/success_rate"] == 1.0


# ==========================================
# Tests for compute_codebook_entropy
# ==========================================


def test_compute_codebook_entropy():
    from atomic_rl.metrics import compute_codebook_entropy

    codebook_size = 4

    # 1. Complete codebook collapse: all samples use code 0
    collapsed_indices = torch.zeros(100, dtype=torch.long)
    h_collapsed = compute_codebook_entropy(collapsed_indices, codebook_size=codebook_size)
    assert h_collapsed.item() == pytest.approx(0.0, abs=1e-4)

    # 2. Perfectly uniform codebook usage: 25 samples of each code (0, 1, 2, 3)
    uniform_indices = torch.tensor([0, 1, 2, 3] * 25, dtype=torch.long)
    h_uniform = compute_codebook_entropy(uniform_indices, codebook_size=codebook_size)
    # Expected Shannon entropy = log(4) = 1.38629...
    assert h_uniform.item() == pytest.approx(np.log(4), rel=1e-4)

    # 3. Floating-point one-hot tensor input
    one_hot = torch.eye(codebook_size).repeat(10, 1)  # [40, 4]
    h_one_hot = compute_codebook_entropy(one_hot, codebook_size=codebook_size)
    assert h_one_hot.item() == pytest.approx(np.log(4), rel=1e-4)


def test_compute_codebook_perplexity_and_usage():
    from atomic_rl.metrics import compute_codebook_perplexity, compute_codebook_usage

    codebook_size = 4

    # 1. Complete collapse: 1 active code, perplexity = 1.0
    collapsed_indices = torch.zeros(50, dtype=torch.long)
    perp_collapsed = compute_codebook_perplexity(collapsed_indices, codebook_size)
    probs_col, active_col = compute_codebook_usage(collapsed_indices, codebook_size)
    assert perp_collapsed.item() == pytest.approx(1.0, abs=1e-3)
    assert active_col.item() == 1
    assert probs_col[0].item() == 1.0
    assert probs_col[1:].sum().item() == 0.0

    # 2. Uniform usage: all 4 active, perplexity = 4.0
    uniform_indices = torch.tensor([0, 1, 2, 3] * 20, dtype=torch.long)
    perp_uniform = compute_codebook_perplexity(uniform_indices, codebook_size)
    probs_uni, active_uni = compute_codebook_usage(uniform_indices, codebook_size)
    assert perp_uniform.item() == pytest.approx(4.0, rel=1e-3)
    assert active_uni.item() == 4
    for k in range(4):
        assert probs_uni[k].item() == pytest.approx(0.25, rel=1e-4)



