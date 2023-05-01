"""Tests for the rasterize sub-module."""

from typing import Tuple

import numpy as np

from av2.rendering.rasterize import xyz_to_bev
from av2.utils.typing import NDArrayFloat


def _build_dummy_raster_inputs(
    n: int, d: int
) -> Tuple[
    NDArrayFloat, Tuple[float, float, float], Tuple[float, float, float], NDArrayFloat
]:
    """Build dummy inputs for the rasterize function.

    Args:
        n: Number of points.
        d: Number of dimensions.

    Returns:
        (n,d) points, (3,) voxel resolution, (3,) grid size, (n,) cmap values.
    """
    xyz = np.ones((n, d))
    voxel_resolution = (0.1, 0.1, 0.1)
    grid_size_m = (50.0, 50.0, 10.0)
    cmap = np.ones_like(xyz[:, 0:1])
    return xyz, voxel_resolution, grid_size_m, cmap


def test_rasterize_Nx3() -> None:
    """Test the rasterize function with (N,3) input."""
    n, d = 1000, 3
    xyz, voxel_resolution, grid_size_m, cmap = _build_dummy_raster_inputs(n, d)
    xyz_to_bev(xyz, voxel_resolution, grid_size_m, cmap)


def test_rasterize_Nx4() -> None:
    """Test the rasterize function with (N,4) input."""
    n, d = 1000, 4
    xyz, voxel_resolution, grid_size_m, cmap = _build_dummy_raster_inputs(n, d)
    xyz_to_bev(xyz, voxel_resolution, grid_size_m, cmap)
