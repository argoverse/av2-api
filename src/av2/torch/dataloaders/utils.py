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

import av2._r as rust  # Rust extension.

XYZLWH_QWXYZ_COLUMNS: Final = (
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
LIDAR_COLUMNS: Final = ("x", "y", "z", "intensity")
QWXYZ_COLUMNS: Final = ("qw", "qx", "qy", "qz")
TRANSLATION_COLUMNS: Final = ("tx_m", "ty_m", "tz_m")


class CuboidMode(str, Enum):
    """Cuboid parameterization modes."""

    XYZLWH_YAW = "XYZLWH_YAW"  # 1-DOF orientation.
    XYZLWH_QWXYZ = "XYZLWH_QWXYZ"  # 3-DOF orientation.


@dataclass(frozen=True)
class Cuboids:
    """Cuboid representation for objects in the scene.

    Args:
        _frame: Dataframe containing the annotations and their attributes.
    """

    _frame: pd.DataFrame

    @cached_property
    def category(self) -> List[str]:
        """Return the object category names."""
        category_names: List[str] = self._frame["category"].to_list()
        return category_names

    @cached_property
    def track_uuid(self) -> List[str]:
        """Return the unique track identifiers."""
        category_names: List[str] = self._frame["track_uuid"].to_list()
        return category_names

    def as_tensor(self, cuboid_mode: CuboidMode = CuboidMode.XYZLWH_YAW) -> torch.Tensor:
        """Return object cuboids as an (N,K) tensor.

        Notation:
            N: Number of objects.
            K: Length of the cuboid mode parameterization.

        Args:
            cuboid_mode: Cuboid parameterization mode. Defaults to (N,7) tensor.

        Returns:
            (N,K) torch.Tensor of cuboids with the specified cuboid_mode parameterization.

        Raises:
            NotImplementedError: Raised if the cuboid mode is not supported.
        """
        xyzlwh_qwxyz = tensor_from_frame(self._frame, list(XYZLWH_QWXYZ_COLUMNS))
        if cuboid_mode == CuboidMode.XYZLWH_YAW:
            quat_wxyz = xyzlwh_qwxyz[:, 6:10]
            w, x, y, z = quat_wxyz[:, 0], quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3]
            _, _, yaw = euler_from_quaternion(w, x, y, z)
            return torch.concat([xyzlwh_qwxyz[:, :6], yaw[:, None]], dim=-1)
        elif cuboid_mode == CuboidMode.XYZLWH_QWXYZ:
            return xyzlwh_qwxyz
        else:
            raise NotImplementedError(f"{cuboid_mode} orientation mode is not implemented.")


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


def tensor_from_frame(frame: pd.DataFrame, columns: List[str]) -> torch.Tensor:
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
        SE(3) object representing a (N,4,4) tensor of homogeneous transformations.
    """
    quaternion_npy = frame.loc[0, list(QWXYZ_COLUMNS)].to_numpy().astype(float)
    quat_wxyz = Quaternion(torch.as_tensor(quaternion_npy, dtype=torch.float32))
    rotation = So3(quat_wxyz)

    translation_npy = frame.loc[0, list(TRANSLATION_COLUMNS)].to_numpy().astype(np.float32)
    translation = torch.as_tensor(translation_npy, dtype=torch.float32)
    dst_SE3_src = Se3(rotation[None], translation[None])
    return dst_SE3_src
