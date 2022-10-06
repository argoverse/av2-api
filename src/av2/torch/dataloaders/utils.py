"""Pytorch sensor dataloader utilities."""

from __future__ import annotations

import itertools
import sys
from dataclasses import dataclass
from enum import Enum, unique
from typing import Final, Optional, Tuple

import fsspec.asyn
import numpy as np
import polars as pl
import torch
from torch import Tensor

from av2.geometry.geometry import mat_to_xyz, quat_to_mat
from av2.geometry.se3 import SE3
from av2.torch.structures.dataframe import DataFrame
from av2.utils.typing import NDArrayFloat

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
    # "vx_m",
    # "vy_m",
    # "vz_m",
)
DEFAULT_LIDAR_TENSOR_FIELDS: Final[Tuple[str, ...]] = ("x", "y", "z")
QUAT_WXYZ_FIELDS: Final[Tuple[str, ...]] = ("qw", "qx", "qy", "qz")


@unique
class OrientationMode(str, Enum):
    """Orientation (pose) modes for the ground truth annotations."""

    QUATERNION_WXYZ = "QUATERNION_WXYZ"
    YAW = "YAW"


@unique
class CuboidMode(str, Enum):
    """Box mode (parameterization) of ground truth annotations."""

    XYZLWH_THETA = "XYZLWH_THETA"
    XYZLWH_QWXYZ = "XYZLWH_QWXYZ"
    XYZ = "XYZ"

    @staticmethod
    def convert(dataframe: DataFrame, src: CuboidMode, target: CuboidMode) -> DataFrame:
        """Convert an annotations dataframe from src to target cuboid parameterization.

        Args:
            dataframe: Annotations dataframe.
            src: Cuboid parameterization of the dataframe.
            target: Desired parameterization of the dataframe.

        Returns:
            The dataframe in the new parameterization format.

        Raises:
            NotImplementedError: If the cuboid mode conversion isn't supported.
        """
        if src == target:
            return dataframe
        if src == CuboidMode.XYZLWH_QWXYZ and target == CuboidMode.XYZLWH_THETA:
            quaternions = dataframe[list(QUAT_WXYZ_FIELDS)].to_numpy().astype(np.float64)
            rotation = quat_to_mat(quaternions)
            yaw = mat_to_xyz(rotation)[:, -1]

            start = min(
                i if field_name in QUAT_WXYZ_FIELDS else sys.maxsize
                for (i, field_name) in enumerate(DEFAULT_ANNOTATIONS_TENSOR_FIELDS)
            )
            field_ordering = tuple(
                filter(lambda field_name: field_name not in QUAT_WXYZ_FIELDS, DEFAULT_ANNOTATIONS_TENSOR_FIELDS)
            )
            field_ordering = field_ordering[:start] + ("yaw",) + field_ordering[start:]
            yaw_frame = DataFrame.from_numpy(yaw, columns=["yaw"])
            dataframe = DataFrame.concat([dataframe, yaw_frame], axis=1)
            dataframe = dataframe[list(field_ordering)]
        elif src == CuboidMode.XYZLWH_QWXYZ and target == CuboidMode.XYZ:
            unit_vertices_obj_xyz_m = np.array(
                [
                    [+1, +1, +1],  # 0
                    [+1, -1, +1],  # 1
                    [+1, -1, -1],  # 2
                    [+1, +1, -1],  # 3
                    [-1, +1, +1],  # 4
                    [-1, -1, +1],  # 5
                    [-1, -1, -1],  # 6
                    [-1, +1, -1],  # 7
                ],
            )

            dims_lwh_m = dataframe[["length_m", "width_m", "height_m"]].to_numpy()

            # Transform unit polygons.
            vertices_obj_xyz_m = (dims_lwh_m[:, None] / 2.0) * unit_vertices_obj_xyz_m[None]

            quat = dataframe[list(QUAT_WXYZ_FIELDS)].to_numpy()
            rotation = quat_to_mat(quat)
            translation = dataframe[["tx_m", "ty_m", "tz_m"]].to_numpy()

            vertices = (rotation @ vertices_obj_xyz_m.transpose(0, 2, 1)).transpose(0, 2, 1) + translation[:, None]
            columns = list(
                itertools.chain.from_iterable(
                    [(f"tx_{i}", f"ty_{i}", f"tz_{i}") for i in range(len(unit_vertices_obj_xyz_m))]
                )
            )
            vertices = vertices.reshape(-1, len(unit_vertices_obj_xyz_m) * 3)
            remaining_columns = set(dataframe.columns).difference(set(["tx_m", "ty_m", "tz_m", "qw", "qx", "qy", "qz"]))
            updated_columns = list(filter(lambda x: x in remaining_columns, dataframe.columns))

            vertices_frame = DataFrame.from_numpy(vertices, columns=columns, backend=dataframe.backend)
            dataframe = DataFrame.concat(
                [dataframe[updated_columns], vertices_frame],
                axis=1,
            )
            return dataframe
        else:
            raise NotImplementedError("This conversion is not implemented!")
        return dataframe


@dataclass(frozen=True)
class Annotations:
    """Dataclass for ground truth annotations."""

    dataframe: DataFrame
    cuboid_mode: CuboidMode = CuboidMode.XYZLWH_QWXYZ

    def as_tensor(
        self,
        cuboid_mode: CuboidMode = CuboidMode.XYZLWH_THETA,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        """Return the lidar sweep as a dense tensor.

        Args:
            cuboid_mode: Target parameterization for the cuboids.
            dtype: Target datatype for casting.

        Returns:
            (N,K) tensor where N is the number of lidar points and K
                is the number of features.
        """
        dataframe = CuboidMode.convert(self.dataframe, self.cuboid_mode, cuboid_mode)
        return torch.as_tensor(dataframe.to_numpy(), dtype=dtype)

    def compute_interior_points(self, lidar: Lidar) -> Tensor:
        """Compute a pairwise interior point mask.

        Args:
            lidar: Lidar object.

        Returns:
            (num_annotations,num_points) boolean tensor indicating whether the point
                falls into the kth cuboid.
        """
        dataframe = CuboidMode.convert(self.dataframe, self.cuboid_mode, CuboidMode.XYZ)
        points_xyz = lidar.as_tensor()

        columns = list(itertools.chain.from_iterable([(f"tx_{i}", f"ty_{i}", f"tz_{i}") for i in range(8)]))
        cuboid_vertices = torch.as_tensor(dataframe[columns].to_numpy(), dtype=torch.float32).reshape(-1, 8, 3)
        pairwise_point_masks = compute_interior_points_mask(points_xyz, cuboid_vertices)
        return pairwise_point_masks


@dataclass(frozen=True)
class Lidar:
    """Dataclass for lidar sweeps."""

    dataframe: DataFrame

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
        dataframe_npy = self.dataframe[list(field_ordering)].to_numpy()
        return torch.as_tensor(dataframe_npy, dtype=dtype)


@dataclass(frozen=True)
class Sweep:
    """Stores the annotations and lidar for one sweep."""

    lidar: Lidar
    sweep_uuid: Tuple[str, int]
    annotations: Optional[Annotations] = None


def prevent_fsspec_deadlock() -> None:
    """Reset the fsspec global lock to prevent deadlocking in forked processes."""
    fsspec.asyn.reset_lock()


def query_pose(poses: DataFrame, timestamp_ns: int) -> SE3:
    """Query the SE(3) transformation as the provided timestamp in nanoseconds.

    Args:
        poses: DataFrame of quaternion and translation components.
        timestamp_ns: Timestamp of interest in nanoseconds.

    Returns:
        SE(3) at timestamp_ns.
    """
    pose = poses[poses["timestamp_ns"] == timestamp_ns][["qw", "qx", "qy", "qz", "tx_m", "ty_m", "tz_m"]]
    pose_npy = pose.to_numpy().squeeze()
    quat = pose_npy[:4].astype(np.float64)
    translation = pose_npy[4:].astype(np.float64)
    return SE3(
        rotation=quat_to_mat(quat),
        translation=translation,
    )


def compute_interior_points_mask(points_xyz: Tensor, cuboid_vertices: Tensor) -> Tensor:
    r"""Compute the interior points within a set of _axis-aligned_ cuboids.

    Reference:
        https://math.stackexchange.com/questions/1472049/check-if-a-point-is-inside-a-rectangular-shaped-area-3d
            5------4
            |\\    |\\
            | \\   | \\
            6--\\--7  \\
            \\  \\  \\ \\
        l    \\  1-------0    h
         e    \\ ||   \\ ||   e
          n    \\||    \\||   i
           g    \\2------3    g
            t      width.     h
             h.               t.

    Args:
        points_xyz: (N,3) Points in Cartesian space.
        cuboid_vertices: (K,8,3) Vertices of the cuboids.

    Returns:
        (N,) A tensor of boolean flags indicating whether the points
            are interior to the cuboid.
    """
    vertices = cuboid_vertices[:, [6, 3, 1]]
    uvw = cuboid_vertices[:, 2:3] - vertices
    reference_vertex = cuboid_vertices[:, 2:3]

    dot_uvw_reference = uvw @ reference_vertex.transpose(1, 2)
    dot_uvw_vertices = torch.diagonal(uvw @ vertices.transpose(1, 2), 0, 2)[..., None]
    dot_uvw_points = uvw @ points_xyz.T

    constraint_a = torch.logical_and(dot_uvw_reference <= dot_uvw_points, dot_uvw_points <= dot_uvw_vertices)
    constraint_b = torch.logical_and(dot_uvw_reference >= dot_uvw_points, dot_uvw_points >= dot_uvw_vertices)
    is_interior: Tensor = torch.logical_or(constraint_a, constraint_b).all(dim=1)
    return is_interior
