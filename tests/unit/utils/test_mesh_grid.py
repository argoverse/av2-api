# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Unit tests for mesh grid related utilities."""

from typing import List

import numpy as np

import av2.geometry.mesh_grid as mesh_grid_utils
from av2.utils.typing import NDArrayFloat


def test_get_mesh_grid_as_point_cloud_3x3square() -> None:
    """Ensure a sampled regular grid returns 9 grid points from 1 meter resolution on 2x2 meter area."""
    min_x = -3  # integer, minimum x-coordinate of 2D grid
    max_x = -1  # integer, maximum x-coordinate of 2D grid
    min_y = 2  # integer, minimum y-coordinate of 2D grid
    max_y = 4  # integer, maximum y-coordinate of 2D grid

    # return pts, a Numpy array of shape (N,2)
    pts = mesh_grid_utils.get_mesh_grid_as_point_cloud(
        min_x, max_x, min_y, max_y, downsample_factor=1.0
    )

    assert pts.shape == (9, 2)
    gt_pts: NDArrayFloat = np.array(
        [
            [-3.0, 2.0],
            [-2.0, 2.0],
            [-1.0, 2.0],
            [-3.0, 3.0],
            [-2.0, 3.0],
            [-1.0, 3.0],
            [-3.0, 4.0],
            [-2.0, 4.0],
            [-1.0, 4.0],
        ]
    )

    assert np.allclose(gt_pts, pts)


def test_get_mesh_grid_as_point_cloud_3x2rect() -> None:
    """Ensure a sampled regular grid returns 6 grid points from 1 meter resolution on 1x2 meter area."""
    min_x = -3  # integer, minimum x-coordinate of 2D grid
    max_x = -1  # integer, maximum x-coordinate of 2D grid
    min_y = 2  # integer, minimum y-coordinate of 2D grid
    max_y = 3  # integer, maximum y-coordinate of 2D grid

    # return pts, a Numpy array of shape (N,2)
    pts = mesh_grid_utils.get_mesh_grid_as_point_cloud(
        min_x, max_x, min_y, max_y, downsample_factor=1.0
    )

    assert pts.shape == (6, 2)
    # fmt: off
    gt_pts: NDArrayFloat = np.array(
        [
            [-3.0, 2.0],
            [-2.0, 2.0],
            [-1.0, 2.0],
            [-3.0, 3.0],
            [-2.0, 3.0],
            [-1.0, 3.0]
        ])
    # fmt: on
    assert np.allclose(gt_pts, pts)


def test_get_mesh_grid_as_point_cloud_single_pt() -> None:
    """Ensure a sampled regular grid returns only 1 point for a range of 0 meters in x and 0 meters in y."""
    min_x = -3  # integer, minimum x-coordinate of 2D grid
    max_x = -3  # integer, maximum x-coordinate of 2D grid
    min_y = 2  # integer, minimum y-coordinate of 2D grid
    max_y = 2  # integer, maximum y-coordinate of 2D grid

    # return pts, a Numpy array of shape (N,2)
    pts = mesh_grid_utils.get_mesh_grid_as_point_cloud(
        min_x, max_x, min_y, max_y, downsample_factor=1.0
    )

    assert pts.shape == (1, 2)
    gt_pts: NDArrayFloat = np.array([[-3.0, 2.0]])

    assert np.allclose(gt_pts, pts)


def test_get_mesh_grid_as_point_cloud_downsample() -> None:
    """Given 3x3 area, ensure a sampled regular grid returns coordinates at 4 corners only."""
    min_x = -3  # integer, minimum x-coordinate of 2D grid
    max_x = 0  # integer, maximum x-coordinate of 2D grid
    min_y = 2  # integer, minimum y-coordinate of 2D grid
    max_y = 5  # integer, maximum y-coordinate of 2D grid

    # return pts, a Numpy array of shape (N,2)
    pts = mesh_grid_utils.get_mesh_grid_as_point_cloud(
        min_x, max_x, min_y, max_y, downsample_factor=3.0
    )

    assert pts.shape == (4, 2)

    # fmt: off
    gt_pts: List[List[float]] = [
        [-3.0, 2.0],
        [0.0, 2.0],
        [-3.0, 5.0],
        [0.0, 5.0]
    ]
    # fmt: on
    assert np.allclose(gt_pts, pts)
