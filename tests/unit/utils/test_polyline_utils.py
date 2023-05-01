# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Unit tests for polyline related utilities."""

import numpy as np

import av2.geometry.polyline_utils as polyline_utils
from av2.utils.typing import NDArrayFloat


def test_convert_lane_boundaries_to_polygon_3d() -> None:
    """Ensure 3d lane polygon is created correctly from 3d left and right lane boundaries."""
    # fmt: off
    right_ln_bnd: NDArrayFloat = np.array(
        [
            [7, 3, 5],
            [11, -1, 4]
        ])
    left_ln_bnd: NDArrayFloat = np.array(
        [
            [10, 3, 7],
            [14, -1, 8]
        ])
    # fmt: on
    polygon = polyline_utils.convert_lane_boundaries_to_polygon(
        right_ln_bnd, left_ln_bnd
    )

    # fmt: off
    gt_polygon: NDArrayFloat = np.array(
        [
            [7, 3, 5],
            [11, -1, 4],
            [14, -1, 8],
            [10, 3, 7],
            [7, 3, 5]
        ])
    # fmt: on
    assert np.allclose(polygon, gt_polygon)


def test_straight_centerline_to_polygon() -> None:
    """Try converting a simple straight polyline into a polygon.

    Represents the conversion from a centerline to a lane segment polygon.

    Note that the returned polygon will ba a Numpy array of
    shape (2N+1,2), with duplicate first and last vertices.
    Dots below signify the centerline coordinates.

            |   .   |
            |   .   |
            |   .   |
            |   .   |
    """
    # create centerline: Numpy array of shape (N,2)
    # fmt: off
    centerline: NDArrayFloat = np.array(
        [
            [0, 2.0],
            [0.0, 0.0],
            [0.0, -2.0]
        ])
    # fmt: on

    polygon = polyline_utils.centerline_to_polygon(centerline, width_scaling_factor=2)

    # assert np.array_equal(polygon, gt_polygon)
    assert polygon.shape == (7, 2)


def test_centerline_to_polygon() -> None:
    """Ensure correctness of conversion of centerline->two parallel lines for degenerate case.

    Ensure we can extract two parallel lines (connected to form a polygon) from a central polyline,
    when some central waypoints have zero slope in dy direction.
    """
    # fmt: off
    centerline: NDArrayFloat = np.array(
        [
            [22.87, 6.56],
            [29.93, 6.82],
            [30.0, 6.82]
        ],
        dtype=float
    )
    # fmt: on
    polygon = polyline_utils.centerline_to_polygon(centerline, width_scaling_factor=0.1)
    assert polygon.shape == (7, 2)
