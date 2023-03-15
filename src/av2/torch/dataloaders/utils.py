"""Pytorch dataloader utilities."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, unique
from functools import cached_property
from typing import Final, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor

import av2._r as rust
from av2.geometry.geometry import quat_to_mat

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
DEFAULT_LIDAR_TENSOR_FIELDS: Final = ("x", "y", "z")
QUAT_WXYZ_FIELDS: Final = ("qw", "qx", "qy", "qz")
TRANSLATION_FIELDS: Final = ("tx_m", "ty_m", "tz_m")


@unique
class OrientationMode(str, Enum):
    """Orientation modes for the ground truth annotations."""

    QUATERNION_WXYZ = "QUATERNION_WXYZ"
    YAW = "YAW"


@dataclass(frozen=True)
class Annotations:
    """Dataclass for ground truth annotations.

    Args:
        dataframe: Dataframe containing the annotations and their attributes.
    """

    dataframe: pd.DataFrame

    @property
    def category_names(self) -> List[str]:
        """Return the category names."""
        category_names: List[str] = self.dataframe["category"].to_list()
        return category_names

    @property
    def track_uuids(self) -> List[str]:
        """Return the unique track identifiers."""
        category_names: List[str] = self.dataframe["track_uuid"].to_list()
        return category_names

    def as_tensor(
        self,
        field_ordering: Tuple[str, ...] = DEFAULT_ANNOTATIONS_TENSOR_FIELDS,
    ) -> Tensor:
        """Return the annotations as a tensor.

        Args:
            field_ordering: Feature ordering for the tensor.

        Returns:
            (N,K) tensor where N is the number of annotations and K
                is the number of annotation fields.
        """
        dataframe_npy = self.dataframe.loc[:, list(field_ordering)].to_numpy().astype(np.float32)
        return torch.as_tensor(dataframe_npy)


@dataclass(frozen=True)
class Lidar:
    """Dataclass for lidar sweeps.

    Args:
        dataframe: Dataframe containing the lidar and its attributes.
    """

    dataframe: pd.DataFrame

    def as_tensor(
        self,
        field_ordering: Tuple[str, ...] = DEFAULT_LIDAR_TENSOR_FIELDS,
    ) -> Tensor:
        """Return the lidar sweep as a tensor.

        Args:
            field_ordering: Feature ordering for the tensor.

        Returns:
            (N,K) tensor where N is the number of lidar points and K
                is the number of features.
        """
        dataframe_npy = self.dataframe.loc[:, list(field_ordering)].to_numpy().astype(np.float32)
        return torch.as_tensor(dataframe_npy)


@dataclass(frozen=True)
class Sweep:
    """Stores the annotations and lidar for one sweep.

    Args:
        annotations: Annotations parameterization.
        city_pose: Rigid transformation describing the city pose of the ego-vehicle.
        lidar: Lidar parameters.
        sweep_uuid: Log id and nanosecond timestamp (unique identifier).
    """

    annotations: Optional[Annotations]
    city_pose: Pose
    lidar: Lidar
    sweep_uuid: Tuple[str, int]

    @classmethod
    def from_rust(cls, sweep: rust.Sweep) -> Sweep:
        """Build a sweep from the Rust backend."""
        annotations = Annotations(dataframe=sweep.annotations.to_pandas())
        city_pose = Pose(dataframe=sweep.city_pose.to_pandas())
        lidar = Lidar(dataframe=sweep.lidar.to_pandas())
        return cls(annotations=annotations, city_pose=city_pose, lidar=lidar, sweep_uuid=sweep.sweep_uuid)


@dataclass(frozen=True)
class Pose:
    """Pose class for rigid transformations."""

    dataframe: pd.DataFrame

    @cached_property
    def Rt(self) -> Tuple[Tensor, Tensor]:
        """Return a (3,3) rotation matrix and a (3,) translation vector."""
        quat_wxyz = self.dataframe.loc[0, list(QUAT_WXYZ_FIELDS)].to_numpy().astype(np.float32)
        translation = self.dataframe.loc[0, list(TRANSLATION_FIELDS)].to_numpy().astype(np.float32)

        rotation = quat_to_mat(quat_wxyz)
        return torch.as_tensor(rotation), torch.as_tensor(translation)
