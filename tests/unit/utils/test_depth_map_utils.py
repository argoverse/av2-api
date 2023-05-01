# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Unit tests for depth map utilities."""

import matplotlib.pyplot as plt
import numpy as np

import av2.utils.depth_map_utils as depth_map_utils
from av2.utils.typing import NDArrayByte, NDArrayFloat


def test_vis_depth_map() -> None:
    """Ensure a dummy RGB image and a dummy depth map can be plotted side by side."""
    visualize = False

    H = 1000
    W = 2000
    img_rgb: NDArrayByte = np.zeros((H, W, 3), dtype=np.uint8)
    img_rgb[:, :, 0] = 255  # channels will be (255,0,0) for red.

    depth_map: NDArrayFloat = (
        np.arange(H * W).reshape(H, W).astype(np.float32) / (H * W) * 255
    )

    depth_map_utils.vis_depth_map(
        img_rgb=img_rgb, depth_map=depth_map, interp_depth_map=True
    )
    if visualize:
        plt.show()
        plt.close("all")

    depth_map_utils.vis_depth_map(
        img_rgb=img_rgb, depth_map=depth_map, interp_depth_map=False
    )
    if visualize:
        plt.show()
        plt.close("all")
