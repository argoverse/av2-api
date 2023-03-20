"""Pytorch sweep module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from kornia.geometry.liegroup import Se3
from kornia.geometry.linalg import transform_points

import av2._r as rust
from av2.map.map_api import ArgoverseStaticMap

from .. import LIDAR_COLUMNS
from .cuboids import Cuboids
from .utils import SE3_from_frame, tensor_from_frame


@dataclass(frozen=True)
class Sweep:
    """Stores the annotations and lidar for one sweep.

    Notation:
        N: Number of lidar points.

    Args:
        city_SE3_ego: Rigid transformation describing the city pose of the ego-vehicle.
        lidar_xyzi: (N,4) torch.Tensor of lidar points containing (x,y,z) in meters and intensity (i).
        sweep_uuid: Log id and nanosecond timestamp (unique identifier).
        cuboids: Cuboids representing objects in the scene.
        is_ground: Tensor of boolean values indicatind which points belong to the ground
    """

    city_SE3_ego: Se3
    lidar_xyzi: torch.Tensor
    sweep_uuid: Tuple[str, int]
    cuboids: Optional[Cuboids]
    is_ground: Optional[torch.Tensor] = None

    @classmethod
    def from_rust(cls, sweep: rust.Sweep, avm: Optional[ArgoverseStaticMap] = None) -> Sweep:
        """Build a sweep from the Rust backend.

        Args:
            sweep: Sweep object from the Rust backend dataloader.
            avm: Map object for computing ground labels

        Returns:
            Sweep object.
        """
        if sweep.annotations is not None:
            cuboids = Cuboids(_frame=sweep.annotations.to_pandas())
        else:
            cuboids = None

        city_SE3_ego = SE3_from_frame(frame=sweep.city_pose.to_pandas())
        lidar_xyzi = tensor_from_frame(sweep.lidar.to_pandas(), list(LIDAR_COLUMNS))
        if avm is not None:
            pcl_ego = lidar_xyzi[:, :3]
            pcl_city_1 = transform_points(city_SE3_ego.matrix(), pcl_ego[None])[0]
            is_ground = torch.from_numpy(avm.get_ground_points_boolean(pcl_city_1.numpy()).astype(bool))
        else:
            is_ground = None

        return cls(
            city_SE3_ego=city_SE3_ego,
            lidar_xyzi=lidar_xyzi,
            sweep_uuid=sweep.sweep_uuid,
            is_ground=is_ground,
            cuboids=cuboids,
        )
