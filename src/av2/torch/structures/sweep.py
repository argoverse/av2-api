"""PyTorch sweep sub-module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from kornia.geometry.liegroup import Se3
from kornia.geometry.linalg import transform_points

import av2._r as rust
from av2.map.map_api import ArgoverseStaticMap
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
        is_ground: Tensor of boolean values indicating which points belong to the ground.
    """

    city_SE3_ego: Se3
    lidar: Lidar
    sweep_uuid: Tuple[str, int]
    cuboids: Optional[Cuboids]
    is_ground: Optional[torch.Tensor] = None

    @classmethod
    def from_rust(
        cls, sweep: rust.Sweep, avm: Optional[ArgoverseStaticMap] = None
    ) -> Sweep:
        """Build a sweep from the Rust backend.

        Args:
            sweep: Sweep object from the Rust backend data-loader.
            avm: Map object for computing ground labels

        Returns:
            Sweep object.
        """
        cuboids: Optional[Cuboids] = None
        if sweep.cuboids is not None:
            cuboids = Cuboids(_frame=sweep.cuboids.to_pandas())
        city_SE3_ego = SE3_from_frame(frame=sweep.city_pose.to_pandas())
        lidar = Lidar(sweep.lidar.to_pandas())

        is_ground = None
        if avm is not None:
            pcl_ego = lidar.as_tensor()[:, :3]
            pcl_city_1 = transform_points(city_SE3_ego.matrix(), pcl_ego[None])[0]
            is_ground = torch.from_numpy(
                avm.get_ground_points_boolean(pcl_city_1.numpy()).astype(bool)
            )

        return cls(
            city_SE3_ego=city_SE3_ego,
            lidar=lidar,
            sweep_uuid=sweep.sweep_uuid,
            is_ground=is_ground,
            cuboids=cuboids,
        )
