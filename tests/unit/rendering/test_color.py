# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Unit test on colormap utilities."""

import numpy as np

import av2.rendering.color as color_utils
from av2.rendering.color import GREEN_HEX, RED_HEX


def test_create_colormap() -> None:
    """Ensure we can create a red-to-green RGB colormap with values in [0,1]."""
    colors_arr_rgb = color_utils.create_colormap(
        color_list=[RED_HEX, GREEN_HEX], n_colors=10
    )
    assert np.logical_and(0 <= colors_arr_rgb, colors_arr_rgb <= 1).all()

    assert colors_arr_rgb.shape == (10, 3)
