"""
Metrics, Evaluators, and Visualizations for Reinforcement Learning.

Organized by domain:
- curves: Training and evaluation returns across seeds, multi-seed confidence intervals, loss curves, IQM.
- plasticity: Dead units, stable rank, weight/gradient norms, continual learning performance.
- explained_variance: Value function regression quality, scatter plots, and variance tracking.
- distributional: Categorical support projection, expectation calculation, and distributional Q-value plots.
- traces: Meta-optimization learning rate traces (IDBD / AutoStep).
- saliency: Gradient-based interpretability and Dueling DQN value vs. advantage saliency maps.
"""

from .curves import (
    compute_iqm,
    compute_confidence_interval,
    plot_episode_returns,
    plot_loss_curves,
    plot_eval_vs_train,
    create_wandb_episode_metrics,
)
from .plasticity import (
    compute_dead_units_proportion,
    compute_stable_rank,
    compute_average_weight_magnitude,
    compute_average_gradient_magnitude,
    plot_plasticity_correlates,
    plot_continual_learning_performance,
    create_wandb_plasticity_metrics,
)
from .explained_variance import (
    compute_explained_variance,
    plot_value_prediction_scatter,
    create_wandb_explained_variance,
)
from .distributional import (
    compute_expected_value,
    plot_categorical_distribution,
    log_distributional_metrics,
)
from .traces import (
    plot_learning_rate_traces,
    create_wandb_lr_plot,
)
from .saliency import (
    compute_saliency_map,
    compute_dueling_saliency,
    plot_dueling_saliency,
    create_wandb_saliency_image,
)
from .codebook import (
    compute_codebook_entropy,
    compute_codebook_perplexity,
    compute_codebook_usage,
    plot_codebook_distribution,
    create_wandb_codebook_metrics,
)

__all__ = [
    # Training & Evaluation Curves
    "compute_iqm",
    "compute_confidence_interval",
    "plot_episode_returns",
    "plot_loss_curves",
    "plot_eval_vs_train",
    "create_wandb_episode_metrics",
    # Plasticity & Continual Learning
    "compute_dead_units_proportion",
    "compute_stable_rank",
    "compute_average_weight_magnitude",
    "compute_average_gradient_magnitude",
    "plot_plasticity_correlates",
    "plot_continual_learning_performance",
    "create_wandb_plasticity_metrics",
    # Explained Variance
    "compute_explained_variance",
    "plot_value_prediction_scatter",
    "create_wandb_explained_variance",
    # Distributional RL
    "compute_expected_value",
    "plot_categorical_distribution",
    "log_distributional_metrics",
    # Optimization Traces
    "plot_learning_rate_traces",
    "create_wandb_lr_plot",
    # Saliency & Interpretability
    "compute_saliency_map",
    "compute_dueling_saliency",
    "plot_dueling_saliency",
    "create_wandb_saliency_image",
    # Codebook Diagnostics
    "compute_codebook_entropy",
    "compute_codebook_perplexity",
    "compute_codebook_usage",
    "plot_codebook_distribution",
    "create_wandb_codebook_metrics",
]
