"""Pytorch dataloader utilities."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, unique
from typing import Final, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
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
class Sweep:
    """Stores the annotations and lidar for one sweep.

    Notation:
        N: Number of lidar points.

    Args:
        annotations: Annotations parameterization.
        city_SE3_ego: Rigid transformation describing the city pose of the ego-vehicle.
        lidar_xyzi: (N,4) Tensor of lidar points containing (x,y,z) in meters and intensity (i).
        sweep_uuid: Log id and nanosecond timestamp (unique identifier).
    """

    annotations: Optional[Annotations]
    city_SE3_ego: Se3
    lidar_xyzi: Tensor
    sweep_uuid: Tuple[str, int]

    @classmethod
    def from_rust(cls, sweep: rust.Sweep) -> Sweep:
        """Build a sweep from the Rust backend."""
        annotations = Annotations(dataframe=sweep.annotations.to_pandas())
        city_SE3_ego = frame_to_SE3(frame=sweep.city_pose.to_pandas())
        lidar_xyzi = frame_to_tensor(sweep.lidar.to_pandas())

        return cls(
            annotations=annotations, city_SE3_ego=city_SE3_ego, lidar_xyzi=lidar_xyzi, sweep_uuid=sweep.sweep_uuid
        )


def frame_to_tensor(frame: pd.DataFrame) -> Tensor:
    """Build lidar `torch` tensor from `pandas` dataframe.

    Notation:
        N: Number of lidar points.
        K: Number of lidar attributes.

    Args:
        frame: (N,K) Pandas DataFrame containing lidar fields.

    Returns:
        (N,4) Tensor of (x,y,z) in meters and intensity (i).
    """
    lidar_npy = frame.loc[:, list(DEFAULT_LIDAR_TENSOR_FIELDS)].to_numpy().astype(np.float32)
    return torch.as_tensor(lidar_npy)


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
