# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>
"""Utilities to evaluate motion forecasting predictions and compute metrics."""

import numpy as np

from av2.utils.typing import NDArrayBool, NDArrayFloat, NDArrayNumber


def compute_ade(forecasted_trajectories: NDArrayNumber, gt_trajectory: NDArrayNumber) -> NDArrayFloat:
    """Compute the average displacement error for a set of K predicted trajectories (for the same actor).

    Args:
        forecasted_trajectories: (K, N, 2) predicted trajectories, each N timestamps in length.
        gt_trajectory: (N, 2) ground truth trajectory.

    Returns:
        (K,) Average displacement error for each of the predicted trajectories.
    """
    # (K,N)
    displacement_errors = np.linalg.norm(forecasted_trajectories - gt_trajectory, axis=2)  # type: ignore
    ade: NDArrayFloat = np.mean(displacement_errors, axis=1)
    return ade


def compute_fde(forecasted_trajectories: NDArrayNumber, gt_trajectory: NDArrayNumber) -> NDArrayFloat:
    """Compute the final displacement error for a set of K predicted trajectories (for the same actor).

    Args:
        forecasted_trajectories: (K, N, 2) predicted trajectories, each N timestamps in length.
        gt_trajectory: (N, 2) ground truth trajectory, FDE will be evaluated against true position at index `N-1`.

    Returns:
        (K,) Final displacement error for each of the predicted trajectories.
    """
    # Compute final displacement error for all K trajectories
    fde_vector = (forecasted_trajectories - gt_trajectory)[:, -1]  # type: ignore
    fde: NDArrayFloat = np.linalg.norm(fde_vector, axis=-1)  # type: ignore
    return fde


def compute_is_missed_prediction(
    forecasted_trajectories: NDArrayNumber,
    gt_trajectory: NDArrayNumber,
    miss_threshold_m: float = 2.0,
) -> NDArrayBool:
    """Compute whether each of K predicted trajectories (for the same actor) missed by more than a distance threshold.

    Args:
        forecasted_trajectories: (K, N, 2) predicted trajectories, each N timestamps in length.
        gt_trajectory: (N, 2) ground truth trajectory, miss will be evaluated against true position at index `N-1`.
        miss_threshold_m: Minimum distance threshold for final displacement to be considered a miss.

    Returns:
        (K,) Bools indicating whether prediction missed by more than specified threshold.
    """
    fde = compute_fde(forecasted_trajectories, gt_trajectory)
    is_missed_prediction = fde > miss_threshold_m  # type: ignore
    return is_missed_prediction
