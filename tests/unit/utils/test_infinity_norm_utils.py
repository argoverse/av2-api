# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Unit tests for infinity norm utilities."""

import numpy as np

import av2.geometry.infinity_norm_utils as infinity_norm_utils
from av2.utils.typing import NDArrayFloat


def test_has_pts_in_infinity_norm_radius1() -> None:
    """No points within radius."""
    # fmt: off
    pts: NDArrayFloat = np.array(
        [
            [5.1, 0],
            [0, -5.1],
            [5.1, 5.1]
        ])
    # fmt: on
    within = infinity_norm_utils.has_pts_in_infinity_norm_radius(
        pts, window_center=np.zeros(2), window_sz=5
    )
    assert not within


def test_has_pts_in_infinity_norm_radius2() -> None:
    """1 point within radius."""
    # fmt: off
    pts: NDArrayFloat = np.array(
        [
            [4.9, 0],
            [0, -5.1],
            [5.1, 5.1]
        ])
    # fmt: on
    within = infinity_norm_utils.has_pts_in_infinity_norm_radius(
        pts, window_center=np.zeros(2), window_sz=5
    )
    assert within


def test_has_pts_in_infinity_norm_radius3() -> None:
    """All pts within radius."""
    # fmt: off
    pts: NDArrayFloat = np.array(
        [
            [4.9, 0],
            [0, -4.9],
            [4.9, 4.9]
        ])
    # fmt: on
    within = infinity_norm_utils.has_pts_in_infinity_norm_radius(
        pts, window_center=np.zeros(2), window_sz=5
    )
    assert within


def test_has_pts_in_infinity_norm_radius4() -> None:
    """All pts within radius."""
    pts: NDArrayFloat = np.array([[4.9, 4.9]])
    within = infinity_norm_utils.has_pts_in_infinity_norm_radius(
        pts, window_center=np.zeros(2), window_sz=5
    )
    assert within
