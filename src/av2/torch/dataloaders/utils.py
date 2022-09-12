"""Pytorch sensor dataloader utilities."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, unique
from typing import Final, Tuple

import pandas as pd
import torch
from torch import Tensor

from av2.torch.dataloaders.conversions import quat_to_yaw

DEFAULT_ANNOTATIONS_COLS: Final[Tuple[str, ...]] = (
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
    "vx_m",
    "vy_m",
    "vz_m",
    "category",
)

LIDAR_GLOB_PATTERN: Final[str] = "*/sensors/lidar/*"
MAX_STR_LEN: Final[int] = 32

DEFAULT_ANNOTATIONS_TENSOR_FIELDS: Final[Tuple[str, ...]] = (
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
    "vx_m",
    "vy_m",
    "vz_m",
)
DEFAULT_LIDAR_TENSOR_FIELDS: Final[Tuple[str, ...]] = ("x", "y", "z", "intensity")


@unique
class OrientationMode(str, Enum):
    """Orientation (pose) modes for the ground truth annotations."""

    QUATERNION_WXYZ = "QUATERNION_WXYZ"
    YAW = "YAW"


@dataclass
class Annotations:
    """Dataclass for ground truth annotations."""

    # timestamp_ns: Tensor
    tx_m: Tensor
    ty_m: Tensor
    tz_m: Tensor
    length_m: Tensor
    width_m: Tensor
    height_m: Tensor
    qw: Tensor
    qx: Tensor
    qy: Tensor
    qz: Tensor
    vx_m: Tensor
    vy_m: Tensor
    vz_m: Tensor
    # track_uuid: Tuple[str, ...]
    category: Tuple[str, ...]

    @classmethod
    def from_dataframe(cls, dataframe: pd.DataFrame) -> Annotations:
        """Build an annotations object from a Pandas DataFrame.

        Args:
            dataframe: Pandas DataFrame of annotations fields.

        Returns:
            The annotations object.
        """
        columns = dataframe.to_dict("list")
        for key, value in columns.items():
            elem = value[0]
            if isinstance(elem, (int, float)):
                columns[key] = torch.as_tensor(value)
        return cls(**columns)

    def as_tensor(
        self,
        field_ordering: Tuple[str, ...] = DEFAULT_ANNOTATIONS_TENSOR_FIELDS,
        orientation_mode: OrientationMode = OrientationMode.YAW,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        """Return the lidar sweep as a dense tensor.

        Args:
            field_ordering: Feature ordering for the tensor.
            orientation_mode: Orientation (pose) representation for the annotations.
            dtype: Target datatype for casting.

        Returns:
            (N,K) tensor where N is the number of lidar points and K
                is the number of features.
        """
        if orientation_mode == OrientationMode.YAW:
            augmented_ordering = list(
                filter(lambda field_name: field_name not in ("qw", "qx", "qy", "qz"), field_ordering)
            )
            augmented_ordering.insert(6, "yaw")
            field_ordering = tuple(augmented_ordering)
        fields = [
            getattr(self, field_name) if field_name != "yaw" else self.yaw_radians for field_name in field_ordering
        ]
        return torch.stack(fields, dim=-1).type(dtype)

    @property
    def quaternion(self) -> Tensor:
        """Quaternion in scalar first order (w, x, y, z)."""
        return torch.stack((self.qw, self.qx, self.qy, self.qz), dim=-1)

    @property
    def yaw_radians(self) -> torch.Tensor:
        """Rotation about the gravity-aligned axis (z) in radians."""
        return quat_to_yaw(self.quaternion)


@dataclass
class Lidar:
    """Dataclass for lidar sweeps."""

    x: Tensor
    y: Tensor
    z: Tensor
    intensity: Tensor
    laser_number: Tensor
    offset_ns: Tensor

    @classmethod
    def from_dataframe(cls, dataframe: pd.DataFrame) -> Lidar:
        """Build a lidar object from a Pandas DataFrame.

        Args:
            dataframe: Pandas DataFrame of lidar fields.

        Returns:
            The lidar object.
        """
        columns = dataframe.to_dict("list")
        for field_name, field in columns.items():
            elem = field[0]
            if isinstance(elem, (int, float)):
                columns[field_name] = torch.as_tensor(field)
        return cls(**columns)

    def as_tensor(
        self, field_ordering: Tuple[str, ...] = DEFAULT_LIDAR_TENSOR_FIELDS, dtype: torch.dtype = torch.float32
    ) -> Tensor:
        """Return the lidar sweep as a dense tensor.

        Args:
            field_ordering: Feature ordering for the tensor.
            dtype: Target datatype for casting.

        Returns:
            (N,K) tensor where N is the number of lidar points and K
                is the number of features.
        """
        fields = [getattr(self, field_name) for field_name in field_ordering]
        return torch.stack(fields, dim=-1).type(dtype)


@dataclass
class Sweep:
    """Stores the annotations and lidar for one sweep."""

    annotations: Annotations
    lidar: Lidar
