# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Utility functions for dilation of a binary mask."""


import cv2
import numpy as np

from av2.utils.typing import NDArrayByte, NDArrayFloat


def dilate_by_l2(img: NDArrayByte, dilation_thresh: float = 5.0) -> NDArrayByte:
    """Dilate a mask using the L2 distance from a zero pixel.

    OpenCV's distance transform calculates the DISTANCE TO THE CLOSEST ZERO PIXEL for each
    pixel of the source image. Although the distance type could be L1, L2, etc, we use L2.

    We specify the "maskSize", which represents the size of the distance transform mask. It can
    be 3, 5, or CV_DIST_MASK_PRECISE (the latter option is only supported by the first function).

    For us, foreground values are 1 and background values are 0.

    Args:
        img: Array of shape (M, N) representing an 8-bit single-channel (binary) source image
        dilation_thresh: Threshold for how far away a zero pixel can be

    Returns:
        An image with the same size with the dilated mask

    Raises:
        ValueError: If distance transform isn't a uint8 array.
    """
    if img.dtype != np.dtype(np.uint8):
        raise ValueError("Input to distance transform must be a uint8 array.")

    mask_diff: NDArrayByte = np.ones_like(img, dtype=np.uint8) - img
    distance_mask: NDArrayFloat = cv2.distanceTransform(
        mask_diff, distanceType=cv2.DIST_L2, maskSize=cv2.DIST_MASK_PRECISE
    )
    dilated_img: NDArrayByte = np.less_equal(distance_mask.astype(np.float32), dilation_thresh).astype(np.uint8)
    return dilated_img
