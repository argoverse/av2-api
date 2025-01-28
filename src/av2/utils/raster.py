# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Rasterization related utilities."""

from typing import List

import cv2
import numpy as np
from cv2.typing import MatLike
from PIL import Image, ImageDraw

from av2.utils.typing import NDArrayByte, NDArrayFloat, NDArrayNumber


def get_mask_from_polygons(
    polygons: List[NDArrayFloat], img_h: int, img_w: int
) -> NDArrayByte:
    """Rasterize multiple polygons onto a single 2d grid/array.

    NOTE: Pillow can gracefully handle the scenario when a polygon has coordinates outside of the grid,
    and also when the first and last vertex of a polygon's coordinates is repeated.

    Args:
        polygons: list of (N,2) numpy float arrays, where N is variable per polygon.
        img_h: height of the image grid to generate, in pixels
        img_w: width of the image grid to generate, in pixels

    Returns:
        2d array with 0/1 values representing a binary segmentation mask
    """
    mask_img = Image.new("L", size=(img_w, img_h), color=0)
    for polygon in polygons:
        vert_list = [(x, y) for x, y in polygon]
        ImageDraw.Draw(mask_img).polygon(vert_list, outline=1, fill=1)

    mask: NDArrayByte = np.array(mask_img)
    return mask


def blend_images(
    img0: NDArrayNumber, img1: NDArrayNumber, alpha: float = 0.7
) -> NDArrayNumber:
    """Alpha-blend two images together.

    Args:
        img0: Array of shape (H,W,3).
        img1: Array of shape (H,W,3).
        alpha: Alpha blending coefficient.

    Returns:
        Array of shape (H,W,3).
    """
    blended = cv2.addWeighted(img0, alpha, img1, (1 - alpha), gamma=0)
    return blended
