"""Pytorch sensor dataloader utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Final, Tuple

import fsspec.asyn
import polars as pl
import torch
from torch import Tensor

from av2.geometry.geometry import mat_to_xyz, quat_to_mat
from av2.geometry.se3 import SE3

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
DEFAULT_LIDAR_TENSOR_FIELDS: Final[Tuple[str, ...]] = ("x", "y", "z")
QUAT_WXYZ_FIELDS: Final[Tuple[str, ...]] = ("qw", "qx", "qy", "qz")


@unique
class OrientationMode(str, Enum):
    """Orientation (pose) modes for the ground truth annotations."""

    QUATERNION_WXYZ = "QUATERNION_WXYZ"
    YAW = "YAW"


@dataclass
class Annotations:
    """Dataclass for ground truth annotations."""

    dataframe: pl.DataFrame

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
            quaternions = self.dataframe.select(pl.col(list(QUAT_WXYZ_FIELDS))).to_numpy()
            mat = quat_to_mat(quaternions)
            yaw = mat_to_xyz(mat)[:, -1]

            first_occurence = min(
                [
                    i if field_name in ["qw", "qx", "qy", "qz"] else math.inf
                    for (i, field_name) in enumerate(field_ordering)
                ]
            )
            field_ordering = tuple(
                filter(lambda field_name: field_name not in QUAT_WXYZ_FIELDS, field_ordering)
            )
            field_ordering = field_ordering[:first_occurence] + ("yaw",) + field_ordering[first_occurence:]
            dataframe = self.dataframe.with_columns(yaw=pl.from_numpy(yaw).to_series())

        return torch.as_tensor(dataframe.select(pl.col(list(field_ordering))).to_numpy(), dtype=dtype)


@dataclass
class Lidar:
    """Dataclass for lidar sweeps."""

    dataframe: pl.DataFrame

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
        return torch.as_tensor(self.dataframe.select(pl.col(list(field_ordering))).to_numpy())


@dataclass
class Sweep:
    """Stores the annotations and lidar for one sweep."""

    annotations: Annotations
    lidar: Lidar


def prevent_fsspec_deadlock() -> None:
    """Reset the fsspec global lock to prevent deadlocking in forked processes."""
    fsspec.asyn.reset_lock()


def query_SE3(poses: pl.DataFrame, timestamp_ns: int) -> SE3:
    """Query the SE(3) transformation as the provided timestamp in nanoseconds.

    Args:
        poses: DataFrame of quaternion and translation components.
        timestamp_ns: Timestamp of interest in nanoseconds.

    Returns:
        SE(3) at timestamp_ns.
    """
    pose = poses.filter(pl.col("timestamp_ns") == timestamp_ns)
    quat = pose.select(["qw", "qx", "qy", "qz"]).to_numpy().squeeze()
    translation = pose.select(["tx_m", "ty_m", "tz_m"]).to_numpy().squeeze()
    return SE3(
        rotation=quat_to_mat(quat),
        translation=translation,
    )
