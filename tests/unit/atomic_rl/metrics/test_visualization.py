import os
import pytest
import torch
import numpy as np
import math
import matplotlib.pyplot as plt
from unittest.mock import MagicMock, patch

from atomic_rl.metrics import (
    create_wandb_lr_plot,
    plot_learning_rate_traces,
    plot_continual_learning_performance,
    plot_plasticity_correlates,
    compute_explained_variance,
    log_distributional_metrics,
    plot_categorical_distribution,
    plot_value_prediction_scatter,
    plot_episode_returns,
    plot_loss_curves,
    plot_eval_vs_train,
)

pytestmark = pytest.mark.unit


def test_compute_explained_variance():
    y_true = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

    # 1. Perfect prediction
    y_pred_perfect = y_true.clone()
    torch.testing.assert_close(
        compute_explained_variance(y_true, y_pred_perfect), torch.tensor(1.0)
    )

    # 2. Constant mean prediction (explained variance = 0)
    y_pred_mean = torch.full_like(y_true, torch.mean(y_true))
    torch.testing.assert_close(
        compute_explained_variance(y_true, y_pred_mean), torch.tensor(0.0), atol=1e-6, rtol=1e-6
    )

    # 3. Bad prediction
    y_pred_bad = y_true * -1
    assert compute_explained_variance(y_true, y_pred_bad).item() < 0.0

    # 4. Zero variance case
    y_true_zero = torch.tensor([1.0, 1.0, 1.0])
    assert torch.isnan(compute_explained_variance(y_true_zero, y_true_zero))


@patch("wandb.Table")
@patch("wandb.plot.line")
def test_log_distributional_metrics(mock_line, mock_table):
    support = torch.tensor([0.0, 1.0, 2.0])
    # Batch size 1, 3 atoms. Logits make atom 1 certain.
    info_dict = {"predictions": torch.tensor([[0.0, 100.0, 0.0]])}
    step = 10

    # 1. Test basic metrics without chart
    metrics = log_distributional_metrics(info_dict, support, step, log_chart=False)

    # Expected Q = sum([0, 1, 0] * [0, 1, 2]) = 1.0
    assert math.isclose(metrics["metrics/expected_q_value"], 1.0, rel_tol=1e-5)
    assert "charts/distribution_curve" not in metrics

    # 2. Test with chart logging
    metrics_with_chart = log_distributional_metrics(
        info_dict, support, step, log_chart=True
    )

    assert mock_table.called
    assert mock_line.called
    assert "charts/distribution_curve" in metrics_with_chart


# ==========================================
# Tests for W&B Telemetry Payloads
# ==========================================


def test_create_wandb_lr_plot():
    """Verify standard mapping layout of the payload returned for learning rates."""
    payload = create_wandb_lr_plot(step=50, relevant_lr=0.25, irrelevant_lr=0.02)

    assert payload == {
        "step": 50,
        "metrics/relevant_lr_mean": 0.25,
        "metrics/irrelevant_lr_mean": 0.02,
    }


# ==========================================
# Tests for Matplotlib Graphic Renderers
# ==========================================


def test_plot_learning_rate_traces(tmp_path):
    """Verify that learning rate trends are processed and exported correctly."""
    # 10 execution steps, 4 individual features
    alphas_history = np.ones((10, 4))
    alphas_history[:, :2] = 0.5  # Relevant partitions
    alphas_history[:, 2:] = 0.05  # Irrelevant partitions

    save_file = tmp_path / "lr_traces.png"

    fig = plot_learning_rate_traces(
        alphas_history=alphas_history, num_relevant=2, save_path=str(save_file)
    )

    assert isinstance(fig, plt.Figure)
    assert os.path.exists(save_file)

    # Close explicitly to avoid background memory accumulation leaks
    plt.close(fig)


def test_plot_continual_learning_performance(tmp_path):
    """Verify parsing and plot structure for cross-task comparison metrics."""
    results_dict = {
        "Base Engine": [0.85, 0.70, 0.55],
        "SWR Enhanced": [0.85, 0.84, 0.86],
    }
    save_file = tmp_path / "continual_learning_perf.png"

    fig = plot_continual_learning_performance(
        results_dict=results_dict, save_path=str(save_file)
    )

    assert isinstance(fig, plt.Figure)
    assert os.path.exists(save_file)

    plt.close(fig)


def test_plot_plasticity_correlates(tmp_path):
    """Verify grid generation side-effects, including feature name layout filtering checks."""
    metrics_dict = {
        "Base System": {
            "Dead Units": [0.4, 0.5, 0.6],
            "Weight Magnitude": [1.2, 1.5, 1.9],
        },
        "SWR": {"Dead Units": [0.1, 0.1, 0.1], "Weight Magnitude": [0.8, 0.8, 0.8]},
    }
    save_file = tmp_path / "plasticity_correlates.png"

    # Runs the plotting code entirely (internal context calls plt.close() at termination)
    plot_plasticity_correlates(metrics_dict=metrics_dict, save_path=str(save_file))

    assert os.path.exists(save_file)


def test_plot_categorical_distribution(tmp_path):
    support = np.linspace(-10, 10, 21)
    probs = np.ones(21) / 21
    save_file = tmp_path / "categorical_dist.png"

    fig = plot_categorical_distribution(support, probs, expected_q=0.0, save_path=str(save_file))
    assert isinstance(fig, plt.Figure)
    assert os.path.exists(save_file)
    plt.close(fig)


def test_plot_value_prediction_scatter(tmp_path):
    y_true = torch.randn(50)
    y_pred = y_true + torch.randn(50) * 0.1
    save_file = tmp_path / "value_scatter.png"

    fig = plot_value_prediction_scatter(y_true, y_pred, save_path=str(save_file))
    assert isinstance(fig, plt.Figure)
    assert os.path.exists(save_file)
    plt.close(fig)


def test_plot_episode_returns(tmp_path):
    returns_by_system = {
        "PPO": np.random.randn(3, 50).cumsum(axis=1),
        "DQN": np.random.randn(3, 50).cumsum(axis=1),
    }
    save_file = tmp_path / "returns_ci.png"

    fig = plot_episode_returns(returns_by_system, smooth_window=5, save_path=str(save_file))
    assert isinstance(fig, plt.Figure)
    assert os.path.exists(save_file)
    plt.close(fig)


def test_plot_loss_curves(tmp_path):
    loss_history = {
        "Policy Loss": np.random.rand(30),
        "Value Loss": np.random.rand(30),
        "Entropy": np.random.rand(30),
    }
    save_file = tmp_path / "loss_curves.png"

    fig = plot_loss_curves(loss_history, save_path=str(save_file))
    assert isinstance(fig, plt.Figure)
    assert os.path.exists(save_file)
    plt.close(fig)


def test_plot_eval_vs_train(tmp_path):
    train_steps = np.arange(100)
    train_returns = np.random.randn(100).cumsum()
    eval_steps = np.arange(0, 100, 10)
    eval_returns = np.random.randn(10).cumsum()
    save_file = tmp_path / "eval_vs_train.png"

    fig = plot_eval_vs_train(train_steps, train_returns, eval_steps, eval_returns, save_path=str(save_file))
    assert isinstance(fig, plt.Figure)
    assert os.path.exists(save_file)
    plt.close(fig)

