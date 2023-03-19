"""Pytorch dataloader utilities."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
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

# Cuboids represented as (x,y,z) in meters, (l,w,h) in meters, and theta (in radians).
CUBOID_XYZLWHT_COLUMN_NAMES: Final = (
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


class CuboidMode(str, Enum):
    """Cuboid parameterization modes."""

    # (x,y,z) translation (meters),
    # (l,w,h) extents (meters),
    # (t,) theta counter-clockwise rotation from the x-axis (in radians).
    XYZLWHT = "XYZLWHT"


@dataclass(frozen=True)
class Cuboids:
    """Cuboid representation for objects in the scene.

    Args:
        _frame: Dataframe containing the annotations and their attributes.
    """

    _frame: pd.DataFrame

    @cached_property
    def as_tensor(self, cuboid_mode: CuboidMode = CuboidMode.XYZLWHT) -> Tensor:
        """Return object cuboids as an (N,K) tensor.

        Notation:
            N: Number of objects.
            K: Length of the cuboid mode parameterization.

        Args:
            cuboid_mode: Cuboid parameterization mode. Defaults to (N,7) tensor.

        Returns:
            (N,K) tensor of cuboids with the specified cuboid_mode parameterization.
        """
        if cuboid_mode == CuboidMode.XYZLWHT:
            cuboids_qwxyz = tensor_from_frame(self._frame, list(CUBOID_XYZLWHT_COLUMN_NAMES))
            quat_wxyz = cuboids_qwxyz[:, 6:10]
            w, x, y, z = quat_wxyz[:, 0], quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3]
            _, _, yaw = euler_from_quaternion(w, x, y, z)
            return torch.concat([cuboids_qwxyz[:, :6], yaw[:, None]], dim=-1)
        else:
            raise NotImplementedError("{orientation_mode} orientation mode is not implemented.")

    @cached_property
    def category_names(self) -> List[str]:
        """Return the object category names."""
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
        city_SE3_ego = SE3_from_frame(frame=sweep.city_pose.to_pandas())
        lidar_xyzi = tensor_from_frame(sweep.lidar.to_pandas(), list(DEFAULT_LIDAR_TENSOR_FIELDS))
        return cls(city_SE3_ego=city_SE3_ego, lidar_xyzi=lidar_xyzi, sweep_uuid=sweep.sweep_uuid, cuboids=cuboids)


def tensor_from_frame(frame: pd.DataFrame, columns: List[str]) -> Tensor:
    """Build lidar `torch` tensor from `pandas` dataframe.

    Notation:
        N: Number of rows.
        K: Number of columns.

    Args:
        frame: (N,K) Pandas DataFrame containing N rows with K columns.
        columns: List of DataFrame columns.

    Returns:
        (N,K) tensor containing the frame data.
    """
    frame_npy = frame.loc[:, columns].to_numpy().astype(np.float32)
    return torch.as_tensor(frame_npy)


def SE3_from_frame(frame: pd.DataFrame) -> Se3:
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
