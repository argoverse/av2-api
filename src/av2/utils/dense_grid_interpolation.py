# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Utility for interpolating a dense 2d grid from sparse values."""

from typing import Final, Union

import numpy as np
import scipy.interpolate

import av2.geometry.mesh_grid as mesh_grid
from av2.utils.typing import NDArrayByte, NDArrayFloat, NDArrayInt

# minimum number of points required by QHull to construct an initial simplex, for interpolation.
MIN_REQUIRED_POINTS_SIMPLEX: Final[int] = 4


def interp_dense_grid_from_sparse(
    grid_img: Union[NDArrayByte, NDArrayFloat],
    points: NDArrayInt,
    values: Union[NDArrayByte, NDArrayFloat],
    grid_h: int,
    grid_w: int,
    interp_method: str,
) -> Union[NDArrayByte, NDArrayFloat]:
    """Interpolate a dense 2d grid from sparse values.

    We interpolate (y,x) points instead of (x,y). Then we get
    not quite the same as `matplotlib.mlab.griddata

    Args:
        grid_img: array of shape (grid_h, grid_w, 3) representing empty image to populate
        points: array of shape (K,2) representing 2d coordinates.
        values: array of shape (K,D), representing values at the K input points. For example,
            D=3 for interpolation of RGB values in the BEV, or D=1 for depth values in an ego-view depth map.
        grid_h: height of the dense grid, in pixels.
        grid_w: width of the dense grid, in pixels.
        interp_method: interpolation method, either "linear" or "nearest"

    Returns:
        grid_img: array of shape (grid_h, grid_w, 3) representing densely interpolated image.
            Data type of output array is determined by the dtype of `values`, e.g. uint8 for RGB values.

    Raises:
        ValueError: If requested interpolation method is invalid.
    """
    if interp_method not in ["linear", "nearest"]:
        raise ValueError("Unknown interpolation method.")
    if grid_img.dtype != values.dtype:
        raise ValueError("Grid and values should be the same datatype.")

    if points.shape[0] < MIN_REQUIRED_POINTS_SIMPLEX:
        # return the empty grid, since we can't interpolate.
        return grid_img

    # get (x,y) tuples back
    grid_coords = mesh_grid.get_mesh_grid_as_point_cloud(
        min_x=0, max_x=grid_w - 1, min_y=0, max_y=grid_h - 1
    )
    # make RGB a function of (dim0=x,dim1=y)
    interp_vals = scipy.interpolate.griddata(
        points, values, grid_coords, method=interp_method
    )

    u = grid_coords[:, 0].astype(np.int32)
    v = grid_coords[:, 1].astype(np.int32)
    # Now index in at (y,x) locations
    grid_img[v, u] = interp_vals
    return grid_img
