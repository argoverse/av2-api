# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Defines a container to hold multiple aggregated LiDAR sweeps."""
from __future__ import annotations

from typing import Final, Optional

import numpy as np

from av2.geometry.se3 import SE3
from av2.structs.sweep import Sweep
from av2.utils.typing import NDArrayByte, NDArrayFloat, NDArrayInt

INVALID_TIMESTAMP: Final[int] = 0


class AggregatedSweep(Sweep):
    """Mutable container representing multiple aggregated LiDAR sweeps.

    Note: unlike Sweep, the AggregatedSweep class has no 1:1 range view representation.
    """

    def __init__(self) -> None:
        """Initialize an empty container."""
        self.xyz: NDArrayFloat = np.zeros((0, 3), dtype=np.float32)
        self.intensity: NDArrayByte = np.zeros((0), dtype=np.uint8)
        self.laser_number: NDArrayByte = np.zeros((0), dtype=np.uint8)
        self.offset_ns: NDArrayInt = np.zeros((0), dtype=np.int32)

        self.timestamp_ns: int = INVALID_TIMESTAMP
        self.ego_SE3_up_lidar: Optional[SE3] = None
        self.ego_SE3_down_lidar: Optional[SE3] = None

    def add_sweep(self, sweep: Sweep) -> None:
        """Add the LiDAR returns associated w/ a single sweep to the aggregate/container."""
        self.xyz = np.vstack([self.xyz, sweep.xyz])
        self.intensity = np.concatenate([self.intensity, sweep.intensity])
        self.laser_number = np.concatenate([self.laser_number, sweep.laser_number])

        # self.offset_ns = sweep.tov_ns - self.tov_ns