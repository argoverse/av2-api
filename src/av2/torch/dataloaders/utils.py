"""Pytorch dataloader utilities."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Final, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from kornia.geometry.conversions import euler_from_quaternion
from kornia.geometry.liegroup import Se3, So3
from kornia.geometry.quaternion import Quaternion
from torch import Tensor

import av2._r as rust

DEFAULT_ANNOTATIONS_TENSOR_FIELDS: Final = (
    "tx_m",
    "ty_m",
    "tz_m",
    "length_m",
    "width_m",
    "height_m",
    "qw",
    "qx",
    "qy",
    "qz",
)
DEFAULT_LIDAR_TENSOR_FIELDS: Final = ("x", "y", "z", "intensity")
QUAT_WXYZ_FIELDS: Final = ("qw", "qx", "qy", "qz")
TRANSLATION_FIELDS: Final = ("tx_m", "ty_m", "tz_m")


@dataclass(frozen=True)
class Cuboids:
    """Object containing metadata for cuboids.

    Args:
        _frame: Dataframe containing the annotations and their attributes.
    """

    _frame: pd.DataFrame

    @cached_property
    def as_tensor(self) -> Tensor:
        """Return cuboids as a (N,7) tensor.

        Cuboid parameterization:
            tx_m: Translation in the x-axis in meters.
            ty_m: Translation in the y-axis in meters.
            tz_m: Translation in the z-axis in meters.
            length_m: Length in meters.
            width_m: Width in meters.
            height_m: Height in meters.
            yaw_radians: Counter-clock rotation measured from the x-axis.

        Returns:
            (N,7) Center-based (in meters) cuboid parameterization with extents + yaw (in radians).
        """
        cuboids_qwxyz = frame_to_tensor(self._frame, list(DEFAULT_ANNOTATIONS_TENSOR_FIELDS))
        quat_wxyz = cuboids_qwxyz[:, 6:10]
        w, x, y, z = quat_wxyz[:, 0], quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3]
        _, _, yaw = euler_from_quaternion(w, x, y, z)
        return torch.concat([cuboids_qwxyz[:, :6], yaw[:, None]], dim=-1)

    @cached_property
    def categories(self) -> List[str]:
        """Return the category names."""
        category_names: List[str] = self._frame["category"].to_list()
        return category_names

    @cached_property
    def track_uuids(self) -> List[str]:
        """Return the unique track identifiers."""
        category_names: List[str] = self._frame["track_uuid"].to_list()
        return category_names


@dataclass(frozen=True)
class Sweep:
    """Stores the annotations and lidar for one sweep.

    Notation:
        N: Number of lidar points.

    Args:
        city_SE3_ego: Rigid transformation describing the city pose of the ego-vehicle.
        lidar_xyzi: (N,4) Tensor of lidar points containing (x,y,z) in meters and intensity (i).
        sweep_uuid: Log id and nanosecond timestamp (unique identifier).
        cuboids: Cuboids representing objects in the scene.
    """

    city_SE3_ego: Se3
    lidar_xyzi: Tensor
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
        city_SE3_ego = frame_to_SE3(frame=sweep.city_pose.to_pandas())
        lidar_xyzi = frame_to_tensor(sweep.lidar.to_pandas(), list(DEFAULT_LIDAR_TENSOR_FIELDS))
        return cls(city_SE3_ego=city_SE3_ego, lidar_xyzi=lidar_xyzi, sweep_uuid=sweep.sweep_uuid, cuboids=cuboids)


def frame_to_tensor(frame: pd.DataFrame, columns: List[str]) -> Tensor:
    """Build lidar `torch` tensor from `pandas` dataframe.

    Notation:
        N: Number of rows.
        K: Number of columns.

    Args:
        frame: (N,K) Pandas DataFrame containing N rows with K columns.
        fields: List of DataFrame columns.

    Returns:
        (N,K) tensor containing the frame data.
    """
    frame_npy = frame.loc[:, columns].to_numpy().astype(np.float32)
    return torch.as_tensor(frame_npy)


def frame_to_SE3(frame: pd.DataFrame) -> Se3:
    """Build SE(3) object from `pandas` DataFrame.

    Notation:
        N: Number of rigid transformations.

    Args:
        frame: (N,4) Pandas DataFrame containing quaternion coefficients.

    Returns:
        Kornia Se3 object representing a (N,4,4) tensor of homogeneous transformations.
    """
    quaternion_npy = frame.loc[0, list(QUAT_WXYZ_FIELDS)].to_numpy().astype(float)
    quat_wxyz = Quaternion(torch.as_tensor(quaternion_npy, dtype=torch.float32))
    rotation = So3(quat_wxyz)

    translation_npy = frame.loc[0, list(TRANSLATION_FIELDS)].to_numpy().astype(np.float32)
    translation = torch.as_tensor(translation_npy, dtype=torch.float32)
    dst_SE3_src = Se3(rotation[None], translation[None])
    return dst_SE3_src
