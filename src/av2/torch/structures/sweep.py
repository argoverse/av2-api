"""PyTorch sweep sub-module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from kornia.geometry.liegroup import Se3

import av2._r as rust
from av2.torch.structures.lidar import Lidar

from .cuboids import Cuboids
from .utils import SE3_from_frame


@dataclass(frozen=True)
class Sweep:
    """Stores the annotations and lidar for one sweep.

    Notation:
        N: Number of lidar points.
        K: Number of lidar features.

    Args:
        city_SE3_ego: Rigid transformation describing the city pose of the ego-vehicle.
        lidar: Lidar sensor data.
        sweep_uuid: Log id and nanosecond timestamp (unique identifier).
        cuboids: Cuboids representing objects in the scene.
    """

    city_SE3_ego: Se3
    lidar: Lidar
    sweep_uuid: Tuple[str, int]
    cuboids: Optional[Cuboids]

    @classmethod
    def from_rust(cls, sweep: rust.Sweep) -> Sweep:
        """Build a sweep from the Rust backend.

        Args:
            sweep: Sweep object from the Rust backend data-loader.

        Returns:
            Sweep object.
        """
        cuboids = Cuboids(_frame=sweep.cuboids.to_pandas())
        city_SE3_ego = SE3_from_frame(frame=sweep.city_pose.to_pandas())
        lidar = Lidar(sweep.lidar.to_pandas())
        return cls(city_SE3_ego=city_SE3_ego, lidar=lidar, sweep_uuid=sweep.sweep_uuid, cuboids=cuboids)
