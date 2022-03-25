# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Defines a helper class for generating bird's eye view (BEV) images."""

import math
from typing import Tuple

import numpy as np

from av2.geometry.sim2 import Sim2
from av2.utils.typing import NDArrayFloat


class BEVParams:
    def __init__(self, range_m: float, res_meters_per_px: float = 0.1, accumulate_sweeps: bool = True) -> None:
        """Determine BEV grid sizes, based on desired range and resolution.

        The image dimensions are a function of the rendering range and the rendering resolution.

        Args:
            range_m: max distance to points to include in rendering, by infinity norm.
            res_meters_per_px: resolution, expressed in meters/pixel.
            accumulate_sweeps: whether to accumulate LiDAR sweeps.
        """
        self.range_m = range_m
        # To find image dimensions, we use: (#meters) * (px/meter) = (#pixels)
        self.img_h: int = int(math.ceil(2 * range_m / res_meters_per_px))
        self.img_w: int = self.img_h

        self.res_meters_per_px = res_meters_per_px
        self.accumulate_sweeps = accumulate_sweeps

        # units are in meters
        # get grid boundaries in meters (and we dont allow indexing beyond img_h - 1, img_w - 1)
        # change limit so it cannot round to above.
        self.xmin_m = -range_m
        self.xmax_m = range_m - self.res_meters_per_px
        self.ymin_m = -range_m
        self.ymax_m = range_m - self.res_meters_per_px

        self.xlims = [self.xmin_m, self.xmax_m]
        self.ylims = [self.ymin_m, self.ymax_m]

    @property
    def img_Sim2_ego(self) -> Sim2:
        """Return the Similarity(2) object that maps ego-vehicle coordinates to image coordinates."""
        # scale measured in px/m = 1 / (m/px).
        # this allows us to do (#meters) * (px/m) = (#pixels)
        return Sim2(R=np.eye(2), t=np.array([-self.xmin_m, -self.ymin_m]), s=1 / self.res_meters_per_px)

    def get_image_dims(self) -> Tuple[int, int]:
        """Return the image dimensions."""
        return self.img_h, self.img_w

    def __str__(self) -> str:
        """Return a string representation of the class for stdout."""
        return f"BEVParams_range_{self.range_m}_res_m_px_{self.res_meters_per_px}_img_h_w_{self.img_h}_{self.img_w}"
