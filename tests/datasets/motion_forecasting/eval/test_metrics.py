# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Unit tests for motion forecasting metrics."""

from typing import Final

import numpy as np
import pytest

import av2.datasets.motion_forecasting.eval.metrics as metrics
from av2.utils.typing import NDArrayFloat, NDArrayNumber

# Build stationary GT trajectory at (0, 0)
test_N: Final[int] = 10
_STATIONARY_GT_TRAJ = np.zeros((test_N, 2))

# Case 1: K=1 forecast stationary at (1, 1)
forecasted_trajectories_stationary_k1 = np.ones((1, test_N, 2))
expected_ade_stationary_k1 = np.full((1,), np.sqrt(2))
expected_fde_stationary_k1 = np.full((1,), np.sqrt(2))

# Case 2: K=6 forecasts stationary at (1, 1)
forecasted_trajectories_stationary_k6 = np.ones((6, test_N, 2))
expected_ade_stationary_k6 = np.full((6,), np.sqrt(2))
expected_fde_stationary_k6 = np.full((6,), np.sqrt(2))

# Case 3: K=1 forecast in straight line on X axis
forecasted_trajectories_straight_k1 = np.stack([np.arange(test_N), np.zeros(test_N)], axis=1)[np.newaxis, ...]  # 1xNx2
expected_ade_straight_k1 = np.full((1,), np.arange(test_N).mean())
expected_fde_straight_k1 = np.full((1,), test_N - 1)

# Case 4: K=6 forecasts in straight line on X axis
forecasted_trajectories_straight_k6: NDArrayFloat = np.concatenate(  # type: ignore
    [forecasted_trajectories_straight_k1] * 6, axis=0
)  # 6xNx2
expected_ade_straight_k6 = np.full((6,), np.arange(test_N).mean())
expected_fde_straight_k6 = np.full((6,), test_N - 1)

# Case 5: K=1 forecast in diagonal line
forecasted_trajectories_diagonal_k1 = np.stack([np.arange(test_N), np.arange(test_N)], axis=1)[np.newaxis, ...]  # 1xNx2
expected_ade_diagonal_k1 = np.full((1,), 6.36396103)
expected_fde_diagonal_k1 = np.full((1,), np.hypot(test_N - 1, test_N - 1))


@pytest.mark.parametrize(
    "forecasted_trajectories, expected_ade",
    [
        (forecasted_trajectories_stationary_k1, expected_ade_stationary_k1),
        (forecasted_trajectories_stationary_k6, expected_ade_stationary_k6),
        (forecasted_trajectories_straight_k1, expected_ade_straight_k1),
        (forecasted_trajectories_straight_k6, expected_ade_straight_k6),
        (forecasted_trajectories_diagonal_k1, expected_ade_diagonal_k1),
    ],
    ids=["stationary_k1", "stationary_k6", "straight_k1", "straight_k6", "diagonal_k1"],
)
def test_compute_ade(forecasted_trajectories: NDArrayNumber, expected_ade: NDArrayFloat) -> None:
    """Test that compute_ade returns the correct output with valid inputs.

    Args:
        forecasted_trajectories: Forecasted trajectories for test case.
        expected_ade: Expected average displacement error.
    """
    ade = metrics.compute_ade(forecasted_trajectories, _STATIONARY_GT_TRAJ)
    np.testing.assert_allclose(ade, expected_ade)  # type: ignore


@pytest.mark.parametrize(
    "forecasted_trajectories, expected_fde",
    [
        (forecasted_trajectories_stationary_k1, expected_fde_stationary_k1),
        (forecasted_trajectories_stationary_k6, expected_fde_stationary_k6),
        (forecasted_trajectories_straight_k1, expected_fde_straight_k1),
        (forecasted_trajectories_straight_k6, expected_fde_straight_k6),
        (forecasted_trajectories_diagonal_k1, expected_fde_diagonal_k1),
    ],
    ids=["stationary_k1", "stationary_k6", "straight_k1", "straight_k6", "diagonal_k1"],
)
def test_compute_fde(forecasted_trajectories: NDArrayNumber, expected_fde: NDArrayFloat) -> None:
    """Test that compute_fde returns the correct output with valid inputs.

    Args:
        forecasted_trajectories: Forecasted trajectories for test case.
        expected_fde: Expected final displacement error.
    """
    fde = metrics.compute_fde(forecasted_trajectories, _STATIONARY_GT_TRAJ)
    assert np.array_equal(fde, expected_fde)


@pytest.mark.parametrize(
    "forecasted_trajectories, miss_threshold_m, expected_is_missed_label",
    [
        (forecasted_trajectories_stationary_k1, 2.0, False),
        (forecasted_trajectories_stationary_k6, 2.0, False),
        (forecasted_trajectories_straight_k1, expected_fde_straight_k1[0] + 0.01, False),
        (forecasted_trajectories_straight_k1, expected_fde_straight_k1[0] - 0.01, True),
        (forecasted_trajectories_diagonal_k1, 2.0, True),
    ],
    ids=["stationary_k1", "stationary_k6", "straight_below_threshold", "straight_above_threshold", "diagonal"],
)
def test_compute_is_missed_prediction(
    forecasted_trajectories: NDArrayNumber, miss_threshold_m: float, expected_is_missed_label: bool
) -> None:
    """Test that compute_is_missed_prediction returns the correct output with valid inputs.

    Args:
        forecasted_trajectories: Forecasted trajectories for test case.
        miss_threshold_m: Minimum distance threshold for final displacement to be considered a miss.
        expected_is_missed_label: Expected is_missed label for test case.
    """
    is_missed_prediction = metrics.compute_is_missed_prediction(
        forecasted_trajectories, _STATIONARY_GT_TRAJ, miss_threshold_m
    )

    # Check that is_missed labels are of the correct shape and have the correct value
    assert is_missed_prediction.shape == forecasted_trajectories.shape[:1]
    assert np.all(is_missed_prediction == expected_is_missed_label)
