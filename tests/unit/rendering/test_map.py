# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Unit tests for HD maps."""

import matplotlib.pyplot as plt
import numpy as np

import av2.rendering.map as map_rendering_utils
from av2.rendering.color import RED_BGR
from av2.utils.typing import NDArrayBool, NDArrayByte, NDArrayInt


def test_draw_visible_polyline_segments_cv2_some_visible() -> None:
    """Test rendering when one vertex is marked as occluded, and so two line segments are dropped out."""
    visualize = False
    # 6 vertices in the polyline.
    line_segments_arr: NDArrayInt = np.array(
        [[50, 0], [50, 20], [50, 40], [50, 60], [50, 80], [60, 120]]
    )

    valid_pts_bool: NDArrayBool = np.array([True, True, True, False, True, True])
    img_bgr: NDArrayByte = np.zeros((100, 100, 3), dtype=np.uint8)
    map_rendering_utils.draw_visible_polyline_segments_cv2(
        line_segments_arr=line_segments_arr,
        valid_pts_bool=valid_pts_bool,
        image=img_bgr,
        color=RED_BGR,
        thickness_px=2,
    )
    if visualize:
        plt.imshow(img_bgr[:, :, ::-1])
        plt.show()


def test_draw_visible_polyline_segments_cv2_all_visible() -> None:
    """Test rendering when all vertices are visible (and thus all line segments are visible)."""
    visualize = False
    # 6 vertices in the polyline.
    line_segments_arr: NDArrayInt = np.array(
        [[50, 0], [50, 20], [50, 40], [50, 60], [50, 80], [60, 120]]
    )

    valid_pts_bool: NDArrayBool = np.array([True, True, True, True, True, True])
    img_bgr: NDArrayByte = np.zeros((100, 100, 3), dtype=np.uint8)
    map_rendering_utils.draw_visible_polyline_segments_cv2(
        line_segments_arr=line_segments_arr,
        valid_pts_bool=valid_pts_bool,
        image=img_bgr,
        color=RED_BGR,
        thickness_px=2,
    )
    if visualize:
        plt.imshow(img_bgr[:, :, ::-1])
        plt.show()
