# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Unit tests on utilities that perform dense 2d grid interpolation from sparse values."""
from typing import Final, Tuple

import numpy as np

import av2.utils.dense_grid_interpolation as dense_grid_interpolation
from av2.utils.typing import NDArrayByte, NDArrayFloat, NDArrayInt

RED_RGB: Final[Tuple[int, int, int]] = (255, 0, 0)
GREEN_RGB: Final[Tuple[int, int, int]] = (0, 255, 0)
BLUE_RGB: Final[Tuple[int, int, int]] = (0, 0, 255)


def test_interp_dense_grid_from_sparse_insufficient_points_simplex() -> None:
    """Try to interpolate a dense grid using an insufficient number of samples.

    Attempts interpolation with 2 data points (need at least 4).
    """
    bev_img: NDArrayByte = np.zeros((10, 10, 3), dtype=np.uint8)
    points: NDArrayInt = np.array([[1, 1], [5, 5]])
    rgb_values: NDArrayByte = np.array([RED_RGB, GREEN_RGB], dtype=np.uint8)
    grid_h = 10
    grid_w = 10

    bev_img_interp = dense_grid_interpolation.interp_dense_grid_from_sparse(
        grid_img=bev_img,
        points=points,
        values=rgb_values,
        grid_h=grid_h,
        grid_w=grid_w,
        interp_method="linear",
    )
    assert np.allclose(bev_img_interp, np.zeros((10, 10, 3), dtype=np.uint8))


def test_interp_dense_grid_from_sparse_byte() -> None:
    """Interpolate *uint8* vals using linear interpolation, with color data points provided in 4 corners of a square.

     ___       __
    |Red     Red |

    |Green  Blue |
    |___      ___|

    """
    bev_img: NDArrayByte = np.zeros((4, 4, 3), dtype=np.uint8)

    # provided as (x,y) tuples
    points: NDArrayInt = np.array([[0, 0], [0, 3], [3, 3], [3, 0]])
    rgb_values: NDArrayByte = np.array(
        [RED_RGB, GREEN_RGB, BLUE_RGB, RED_RGB], dtype=np.uint8
    )
    grid_h = 4
    grid_w = 4

    bev_img_interp = dense_grid_interpolation.interp_dense_grid_from_sparse(
        grid_img=bev_img,
        points=points,
        values=rgb_values,
        grid_h=grid_h,
        grid_w=grid_w,
        interp_method="linear",
    )

    assert bev_img_interp.dtype == np.dtype(np.uint8)
    assert isinstance(bev_img_interp, np.ndarray)
    assert bev_img_interp.shape == (4, 4, 3)

    # now, index in at (y,x)
    assert np.allclose(bev_img_interp[0, 0], RED_RGB)
    assert np.allclose(bev_img_interp[3, 0], GREEN_RGB)
    assert np.allclose(bev_img_interp[3, 3], BLUE_RGB)
    assert np.allclose(bev_img_interp[0, 3], RED_RGB)


def test_interp_dense_grid_from_sparse_float() -> None:
    """Interpolate *float* vals using linear interpolation, with color data points provided in 4 corners of a square.

     ___       __
    |Red     Red |

    |Green  Blue |
    |___      ___|

    """
    bev_img: NDArrayFloat = np.zeros((4, 4, 3), dtype=float)

    # provided as (x,y) tuples
    points: NDArrayInt = np.array([[0, 0], [0, 3], [3, 3], [3, 0]], dtype=int)
    rgb_values: NDArrayFloat = np.array(
        [RED_RGB, GREEN_RGB, BLUE_RGB, RED_RGB], dtype=float
    )
    grid_h = 4
    grid_w = 4

    bev_img_interp = dense_grid_interpolation.interp_dense_grid_from_sparse(
        grid_img=bev_img,
        points=points,
        values=rgb_values,
        grid_h=grid_h,
        grid_w=grid_w,
        interp_method="linear",
    )

    assert bev_img_interp.dtype == np.dtype(np.float64)
    assert isinstance(bev_img_interp, np.ndarray)
    assert bev_img_interp.shape == (4, 4, 3)

    # now, index in at (y,x)
    assert np.allclose(bev_img_interp[0, 0], RED_RGB)
    assert np.allclose(bev_img_interp[3, 0], GREEN_RGB)
    assert np.allclose(bev_img_interp[3, 3], BLUE_RGB)
    assert np.allclose(bev_img_interp[0, 3], RED_RGB)
