# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Intersection over Union (IoU) methods."""

import numpy as np

from av2.utils.typing import NDArrayFloat


def iou_3d_axis_aligned(src_dims_m: NDArrayFloat, target_dims_m: NDArrayFloat) -> NDArrayFloat:
    """Compute 3d, axis-aligned (vertical axis alignment) intersection-over-union (IoU) between two sets of cuboids.

    Both objects are aligned to their +x axis and their centroids are placed at the origin
    before computation of the IoU.

    Args:
        src_dims_m: (N,3) Source dimensions (length,width,height).
        target_dims_m: (N,3) Target dimensions (length,width,height).

    Returns:
        (N,) Intersection-over-union between the source and target dimensions.
    """
    inter: NDArrayFloat = np.minimum(src_dims_m, target_dims_m).prod(axis=1)
    union: NDArrayFloat = np.maximum(src_dims_m, target_dims_m).prod(axis=1)
    iou: NDArrayFloat = np.divide(inter, union)
    return iou
