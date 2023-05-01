# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Unit tests for N-dimensional grids."""

import numpy as np

from av2.rendering.color import GRAY_BGR
from av2.structures.ndgrid import BEVGrid, NDGrid
from av2.utils.typing import NDArrayFloat, NDArrayInt


def test_ndgrid_3d() -> None:
    """Unit tests for the NDGrid class."""
    min_range_m = (-5.0, -5.0, -5.0)
    max_range_m = (+5.0, +5.0, +5.0)
    resolution_m_per_cell = (+0.1, +0.1, +0.1)

    ndgrid = NDGrid(min_range_m, max_range_m, resolution_m_per_cell)

    dims_expected = (100.0, 100.0, 100.0)
    assert ndgrid.dims == dims_expected

    range_m_expected = (10.0, 10.0, 10.0)
    assert ndgrid.range_m == range_m_expected

    points: NDArrayFloat = np.array([[1.11, 2.22, 3.33], [4.44, 5.55, 6.66]])
    scaled_points = ndgrid.scale_points(points)
    scaled_points_expected = [[11.1, 22.2, 33.3], [44.4, 55.5, 66.6]]
    assert np.allclose(scaled_points, scaled_points_expected)

    quantized_points = ndgrid.quantize_points(points)
    quantized_points_expected = [[1, 2, 3], [4, 6, 7]]
    assert np.allclose(quantized_points, quantized_points_expected)

    grid_coordinates = ndgrid.transform_to_grid_coordinates(points)
    grid_coordinates_expected = [[61, 72, 83], [94, 106, 117]]
    assert np.allclose(grid_coordinates, grid_coordinates_expected)


def test_bev_grid() -> None:
    """Unit tests for the BEVGrid class."""
    min_range_m = (-5.0, -5.0)
    max_range_m = (+5.0, +5.0)
    resolution_m_per_cell = (+0.1, +0.1)

    bev_grid = BEVGrid(min_range_m, max_range_m, resolution_m_per_cell)

    points: NDArrayFloat = np.array([[1.11, 2.22, 3.33], [4.44, 5.55, 6.66]])

    color = GRAY_BGR
    bev_img = bev_grid.points_to_bev_img(points, color=color)

    grid_coordinates_expected: NDArrayInt = np.array([61, 72], dtype=int)
    values_expected = GRAY_BGR
    assert np.array_equal(
        bev_img[grid_coordinates_expected[1], grid_coordinates_expected[0]],
        values_expected,
    )


def test_BEVGrid_non_integer_multiple() -> None:
    """Unit test with non-integer-multiple resolution."""
    min_range_m = (-5.0, -5.0)
    max_range_m = (+5.0, +5.0)
    resolution_m_per_cell = (+0.3, +0.3)

    bev_grid = BEVGrid(min_range_m, max_range_m, resolution_m_per_cell)

    dims_expected = (33, 33)
    assert bev_grid.dims == dims_expected

    range_m_expected = (10.0, 10.0)
    assert bev_grid.range_m == range_m_expected

    points: NDArrayFloat = np.array([[1.11, 2.22], [4.44, 5.55]])
    scaled_points = bev_grid.scale_points(points)
    scaled_points_expected = [[3.7, 7.4], [14.8, 18.5]]
    assert np.allclose(scaled_points, scaled_points_expected)

    quantized_points = bev_grid.quantize_points(points)
    quantized_points_expected = [[1, 2], [4, 6]]
    assert np.allclose(quantized_points, quantized_points_expected)

    grid_coordinates = bev_grid.transform_to_grid_coordinates(points)
    grid_coordinates_expected = [[20, 24], [31, 35]]
    assert np.allclose(grid_coordinates, grid_coordinates_expected)

    color = GRAY_BGR

    points2: NDArrayFloat = np.array([[4.99, 4.99]])
    bev_img = bev_grid.points_to_bev_img(points2, color=color)
    assert bev_img.sum() == 0  # round(4.99 / 0.3) is out of bounds, so it won't be set.
