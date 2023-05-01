# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>
"""Utilities to evaluate motion forecasting predictions and compute metrics."""

from typing import List

import numpy as np

from av2.utils.typing import NDArrayBool, NDArrayFloat


def compute_ade(
    forecasted_trajectories: NDArrayFloat, gt_trajectory: NDArrayFloat
) -> NDArrayFloat:
    """Compute the average displacement error for a set of K predicted trajectories (for the same actor).

    Args:
        forecasted_trajectories: (K, N, 2) predicted trajectories, each N timestamps in length.
        gt_trajectory: (N, 2) ground truth trajectory.

    Returns:
        (K,) Average displacement error for each of the predicted trajectories.
    """
    displacement_errors = np.linalg.norm(
        forecasted_trajectories - gt_trajectory, axis=2
    )
    ade: NDArrayFloat = np.mean(displacement_errors, axis=1)
    return ade


def compute_fde(
    forecasted_trajectories: NDArrayFloat, gt_trajectory: NDArrayFloat
) -> NDArrayFloat:
    """Compute the final displacement error for a set of K predicted trajectories (for the same actor).

    Args:
        forecasted_trajectories: (K, N, 2) predicted trajectories, each N timestamps in length.
        gt_trajectory: (N, 2) ground truth trajectory, FDE will be evaluated against true position at index `N-1`.

    Returns:
        (K,) Final displacement error for each of the predicted trajectories.
    """
    # Compute final displacement error for all K trajectories
    error_vector: NDArrayFloat = forecasted_trajectories - gt_trajectory
    fde_vector = error_vector[:, -1]
    fde: NDArrayFloat = np.linalg.norm(fde_vector, axis=-1)
    return fde


def compute_is_missed_prediction(
    forecasted_trajectories: NDArrayFloat,
    gt_trajectory: NDArrayFloat,
    miss_threshold_m: float = 2.0,
) -> NDArrayBool:
    """Compute whether each of K predicted trajectories (for the same actor) missed by more than a distance threshold.

    Args:
        forecasted_trajectories: (K, N, 2) predicted trajectories, each N timestamps in length.
        gt_trajectory: (N, 2) ground truth trajectory, miss will be evaluated against true position at index `N-1`.
        miss_threshold_m: Minimum distance threshold for final displacement to be considered a miss.

    Returns:
        (K,) bools indicating whether prediction missed by more than specified threshold.
    """
    fde = compute_fde(forecasted_trajectories, gt_trajectory)
    is_missed_prediction: NDArrayBool = fde > miss_threshold_m
    return is_missed_prediction


def compute_brier_ade(
    forecasted_trajectories: NDArrayFloat,
    gt_trajectory: NDArrayFloat,
    forecast_probabilities: NDArrayFloat,
    normalize: bool = False,
) -> NDArrayFloat:
    """Compute a probability-weighted (using Brier score) ADE for K predicted trajectories (for the same actor).

    Args:
        forecasted_trajectories: (K, N, 2) predicted trajectories, each N timestamps in length.
        gt_trajectory: (N, 2) ground truth trajectory.
        forecast_probabilities: (K,) probabilities associated with each prediction.
        normalize: Normalizes `forecast_probabilities` to sum to 1 when set to True.

    Returns:
        (K,) Probability-weighted average displacement error for each predicted trajectory.
    """
    # Compute ADE with Brier score component
    brier_score = _compute_brier_score(
        forecasted_trajectories, forecast_probabilities, normalize
    )
    ade_vector = compute_ade(forecasted_trajectories, gt_trajectory)
    brier_ade: NDArrayFloat = ade_vector + brier_score
    return brier_ade


def compute_brier_fde(
    forecasted_trajectories: NDArrayFloat,
    gt_trajectory: NDArrayFloat,
    forecast_probabilities: NDArrayFloat,
    normalize: bool = False,
) -> NDArrayFloat:
    """Compute a probability-weighted (using Brier score) FDE for K predicted trajectories (for the same actor).

    Args:
        forecasted_trajectories: (K, N, 2) predicted trajectories, each N timestamps in length.
        gt_trajectory: (N, 2) ground truth trajectory.
        forecast_probabilities: (K,) probabilities associated with each prediction.
        normalize: Normalizes `forecast_probabilities` to sum to 1 when set to True.

    Returns:
        (K,) Probability-weighted final displacement error for each predicted trajectory.
    """
    # Compute FDE with Brier score component
    brier_score = _compute_brier_score(
        forecasted_trajectories, forecast_probabilities, normalize
    )
    fde_vector = compute_fde(forecasted_trajectories, gt_trajectory)
    brier_fde: NDArrayFloat = fde_vector + brier_score
    return brier_fde


def _compute_brier_score(
    forecasted_trajectories: NDArrayFloat,
    forecast_probabilities: NDArrayFloat,
    normalize: bool = False,
) -> NDArrayFloat:
    """Compute Brier score for K predicted trajectories.

    Note: This function computes Brier score under the assumption that each trajectory is the true "best" prediction
    (i.e. each predicted trajectory has a ground truth probability of 1.0).

    Args:
        forecasted_trajectories: (K, N, 2) predicted trajectories, each N timestamps in length.
        forecast_probabilities: (K,) probabilities associated with each prediction.
        normalize: Normalizes `forecast_probabilities` to sum to 1 when set to True.

    Raises:
        ValueError: If the number of forecasted trajectories and probabilities don't match.
        ValueError: If normalize=False and `forecast_probabilities` contains values outside of the range [0, 1].

    Returns:
        (K,) Brier score for each predicted trajectory.
    """
    # Validate that # of forecast probabilities matches forecasted trajectories
    if len(forecasted_trajectories) != len(forecast_probabilities):
        raise ValueError()

    # Validate that all forecast probabilities are in the range [0, 1]
    if np.logical_or(forecast_probabilities < 0.0, forecast_probabilities > 1.0).any():
        raise ValueError(
            "At least one forecast probability falls outside the range [0, 1]."
        )

    # If enabled, normalize forecast probabilities to sum to 1
    if normalize:
        forecast_probabilities = forecast_probabilities / np.sum(forecast_probabilities)

    brier_score: NDArrayFloat = np.square((1 - forecast_probabilities))
    return brier_score


def compute_world_fde(
    forecasted_world_trajectories: NDArrayFloat, gt_world_trajectories: NDArrayFloat
) -> NDArrayFloat:
    """Compute the mean final displacement error for each of K predicted worlds.

    Args:
        forecasted_world_trajectories: (M, K, N, 2) K predicted trajectories of length N, for each of M actors.
        gt_world_trajectories: (M, N, 2) ground truth trajectories of length N, for each of M actors.

    Returns:
        (K,) Mean final displacement error for each of the predicted worlds.
    """
    actor_fdes = [
        compute_fde(forecasted_actor_trajectories, gt_actor_trajectory)
        for forecasted_actor_trajectories, gt_actor_trajectory in zip(
            forecasted_world_trajectories, gt_world_trajectories
        )
    ]

    world_fdes: NDArrayFloat = np.stack(actor_fdes).mean(axis=0)
    return world_fdes


def compute_world_ade(
    forecasted_world_trajectories: NDArrayFloat, gt_world_trajectories: NDArrayFloat
) -> NDArrayFloat:
    """Compute the mean average displacement error for each of K predicted worlds.

    Args:
        forecasted_world_trajectories: (M, K, N, 2) K predicted trajectories of length N, for each of M actors.
        gt_world_trajectories: (M, N, 2) ground truth trajectories of length N, for each of M actors.

    Returns:
        (K,) Mean average displacement error for each of the predicted worlds.
    """
    actor_ades = [
        compute_ade(forecasted_actor_trajectories, gt_actor_trajectory)
        for forecasted_actor_trajectories, gt_actor_trajectory in zip(
            forecasted_world_trajectories, gt_world_trajectories
        )
    ]

    world_ades: NDArrayFloat = np.stack(actor_ades).mean(axis=0)
    return world_ades


def compute_world_misses(
    forecasted_world_trajectories: NDArrayFloat,
    gt_world_trajectories: NDArrayFloat,
    miss_threshold_m: float = 2.0,
) -> NDArrayBool:
    """For each world, compute whether predictions for each actor misssed by more than a distance threshold.

    Args:
        forecasted_world_trajectories: (M, K, N, 2) K predicted trajectories of length N, for each of M actors.
        gt_world_trajectories: (M, N, 2) ground truth trajectories of length N, for each of M actors.
        miss_threshold_m: Minimum distance threshold for final displacement to be considered a miss.

    Returns:
        (M, K) bools indicating whether prediction missed for actor in each world.
    """
    actor_fdes = [
        compute_fde(forecasted_actor_trajectories, gt_actor_trajectory)
        for forecasted_actor_trajectories, gt_actor_trajectory in zip(
            forecasted_world_trajectories, gt_world_trajectories
        )
    ]

    world_actor_missed: NDArrayBool = np.stack(actor_fdes) > miss_threshold_m
    return world_actor_missed


def compute_world_brier_fde(
    forecasted_world_trajectories: NDArrayFloat,
    gt_world_trajectories: NDArrayFloat,
    forecasted_world_probabilities: NDArrayFloat,
    normalize: bool = False,
) -> NDArrayFloat:
    """Compute the mean final displacement error for each of K predicted worlds.

    Args:
        forecasted_world_trajectories: (M, K, N, 2) K predicted trajectories of length N, for each of M actors.
        gt_world_trajectories: (M, N, 2) ground truth trajectories of length N, for each of M actors.
        forecasted_world_probabilities: (K,) normalized probabilities associated with each world.
        normalize: Normalizes `forecasted_world_probabilities` to sum to 1 when set to True.

    Returns:
        (K,) Mean probability-weighted final displacement error for each of the predicted worlds.
    """
    actor_brier_fdes = [
        compute_brier_fde(
            forecasted_actor_trajectories,
            gt_actor_trajectory,
            forecasted_world_probabilities,
            normalize,
        )
        for forecasted_actor_trajectories, gt_actor_trajectory in zip(
            forecasted_world_trajectories, gt_world_trajectories
        )
    ]

    world_brier_fdes: NDArrayFloat = np.stack(actor_brier_fdes).mean(axis=0)
    return world_brier_fdes


def compute_world_collisions(
    forecasted_world_trajectories: NDArrayFloat, collision_threshold_m: float = 1.0
) -> NDArrayBool:
    """Compute whether any of the forecasted trajectories collide with each other.

    Args:
        forecasted_world_trajectories: (M, K, N, 2) K predicted trajectories of length N, for each of M actors.
        collision_threshold_m: Distance threshold at which point a collision is considered to have occured.

    Returns:
        (M, K) bools indicating if a collision was present for an actor in each predicted world.
    """
    actor_collisions: List[NDArrayBool] = []
    for actor_idx in range(len(forecasted_world_trajectories)):
        # Compute distance from current actor to all other predicted actors at each timestep
        forecasted_actor_trajectories = forecasted_world_trajectories[actor_idx]
        scenario_actor_dists = np.linalg.norm(
            forecasted_world_trajectories - forecasted_actor_trajectories, axis=-1
        )

        # For each world, find the closest distance to any other predicted actor, at any time
        scenario_actor_dists[actor_idx, :, :] = np.inf
        closest_dist_to_other_actor_m = scenario_actor_dists.min(axis=-1).min(axis=0)
        actor_collided_in_world = closest_dist_to_other_actor_m < collision_threshold_m
        actor_collisions.append(actor_collided_in_world)

    return np.stack(actor_collisions)
