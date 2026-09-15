import pytest
import torch
from atomic_rl.schedules.stepwise import get_stepwise_schedule

pytestmark = pytest.mark.unit


def test_stepwise_schedule_single_threshold_scalar():
    thresholds = [3]
    values = [1.0, 0.0]

    # At or below threshold -> values[0]
    assert get_stepwise_schedule(0, thresholds, values) == 1.0
    assert get_stepwise_schedule(1, thresholds, values) == 1.0
    assert get_stepwise_schedule(3, thresholds, values) == 1.0

    # Above threshold -> values[1]
    assert get_stepwise_schedule(4, thresholds, values) == 0.0
    assert get_stepwise_schedule(10, thresholds, values) == 0.0


def test_stepwise_schedule_single_threshold_tensor():
    thresholds = [3]
    values = [1.0, 0.0]
    steps = torch.tensor([0, 1, 3, 4, 10])
    expected = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0])

    result = get_stepwise_schedule(steps, thresholds, values)
    torch.testing.assert_close(result, expected)


def test_stepwise_schedule_multi_threshold_scalar():
    thresholds = [5, 10]
    values = [1.0, 0.5, 0.1]

    assert get_stepwise_schedule(0, thresholds, values) == 1.0
    assert get_stepwise_schedule(5, thresholds, values) == 1.0
    assert get_stepwise_schedule(6, thresholds, values) == 0.5
    assert get_stepwise_schedule(10, thresholds, values) == 0.5
    assert get_stepwise_schedule(11, thresholds, values) == pytest.approx(0.1)
    assert get_stepwise_schedule(100, thresholds, values) == pytest.approx(0.1)


def test_stepwise_schedule_multi_threshold_tensor():
    thresholds = [5, 10]
    values = [1.0, 0.5, 0.1]
    steps = torch.tensor([0, 5, 6, 10, 11, 20])
    expected = torch.tensor([1.0, 1.0, 0.5, 0.5, 0.1, 0.1])

    result = get_stepwise_schedule(steps, thresholds, values)
    torch.testing.assert_close(result, expected)


def test_stepwise_schedule_invalid_lengths():
    with pytest.raises(ValueError):
        get_stepwise_schedule(0, thresholds=[5, 10], values=[1.0, 0.0])
