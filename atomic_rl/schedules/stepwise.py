from typing import Sequence, Union
import torch


def get_stepwise_schedule(
    step: Union[int, float, torch.Tensor],
    thresholds: Sequence[Union[int, float]],
    values: Sequence[float],
) -> Union[float, torch.Tensor]:
    """Returns scheduled value for `step` based on step thresholds and values.

    Args:
        step (Union[int, float, torch.Tensor]): Current step or count.
        thresholds (Sequence[Union[int, float]]): Sequence of step boundaries where transitions occur (length N).
        values (Sequence[float]): Sequence of scheduled values for each interval (length N + 1).

    Returns:
        Union[float, torch.Tensor]: Scheduled value at the current step.

    Example:
        get_stepwise_schedule(step=3, thresholds=[5, 10], values=[1.0, 0.5, 0.0])
        # step <= 5        -> returns 1.0
        # 5 < step <= 10   -> returns 0.5
        # step > 10        -> returns 0.0
    """
    if len(values) != len(thresholds) + 1:
        raise ValueError(
            f"Expected len(values) ({len(values)}) to equal len(thresholds) + 1 ({len(thresholds) + 1})."
        )

    step_t = torch.as_tensor(step)
    step_float = step_t.float() if not step_t.is_floating_point() else step_t
    boundaries = torch.tensor(thresholds, device=step_t.device, dtype=step_float.dtype)
    vals = torch.tensor(values, device=step_t.device, dtype=torch.float32)

    indices = torch.bucketize(step_float, boundaries, right=False)
    out = vals[indices]
    return out.item() if step_t.ndim == 0 else out
