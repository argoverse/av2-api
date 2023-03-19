"""Pytorch sweep module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from kornia.geometry.liegroup import Se3

import av2._r as rust

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
    """

    city_SE3_ego: Se3
    lidar_xyzi: torch.Tensor
    sweep_uuid: Tuple[str, int]
    cuboids: Optional[Cuboids]

    @classmethod
    def from_rust(cls, sweep: rust.Sweep) -> Sweep:
        """Build a sweep from the Rust backend.

        Args:
            sweep: Sweep object from the Rust backend dataloader.

        Returns:
            Sweep object.
        """
        cuboids = Cuboids(_frame=sweep.annotations.to_pandas())
        city_SE3_ego = SE3_from_frame(frame=sweep.city_pose.to_pandas())
        lidar_xyzi = tensor_from_frame(sweep.lidar.to_pandas(), list(LIDAR_COLUMNS))
        return cls(city_SE3_ego=city_SE3_ego, lidar_xyzi=lidar_xyzi, sweep_uuid=sweep.sweep_uuid, cuboids=cuboids)
