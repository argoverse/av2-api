"""Pytorch detection dataloader utilities."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, unique
from functools import cached_property
from typing import Dict, Final, List, Optional, Tuple

import fsspec.asyn
import numpy as np
import pandas as pd
import torch
from ab2.evaluation.scene_flow.constants import CATEGORY_MAP
from kornia.geometry.conversions import (
    convert_points_from_homogeneous,
    convert_points_to_homogeneous,
    euler_from_quaternion,
)
from kornia.geometry.liegroup import Se3, So3
from kornia.geometry.linalg import transform_points
from kornia.geometry.quaternion import Quaternion
from torch import BoolTensor, ByteTensor, FloatTensor, Tensor

import av2._r as rust
from av2.datasets.sensor.constants import AnnotationCategories
from av2.geometry.se3 import SE3
from av2.map.map_api import ArgoverseStaticMap
from av2.structures.cuboid import Cuboid, CuboidList
from av2.utils.typing import NDArrayBool, NDArrayByte, NDArrayFloat

MAX_STR_LEN: Final[int] = 32

# Cuboids represented as (x,y,z) in meters, (l,w,h) in meters, and theta (in radians).
XYZLWH_QWXYZ_COLUMN_NAMES: Final = (
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

    def as_tensor(self, cuboid_mode: CuboidMode = CuboidMode.XYZLWH_YAW) -> Tensor:
        """Return object cuboids as an (N,K) tensor.

        Notation:
            N: Number of objects.
            K: Length of the cuboid mode parameterization.

        Args:
            cuboid_mode: Cuboid parameterization mode. Defaults to (N,7) tensor.

        Returns:
            (N,K) Tensor of cuboids with the specified cuboid_mode parameterization.

        Raises:
            NotImplementedError: Raised if the cuboid mode is not supported.
        """
        xyzlwh_qwxyz = tensor_from_frame(self._frame, list(XYZLWH_QWXYZ_COLUMN_NAMES))
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
        lidar_xyzi: (N,4) Tensor of lidar points containing (x,y,z) in meters and intensity (i).
        sweep_uuid: Log id and nanosecond timestamp (unique identifier).
        is_ground: Tensor of boolean values indicatind which points belong to the ground
        cuboids: Cuboids representing objects in the scene.
    """

    city_SE3_ego: Se3
    lidar_xyzi: Tensor
    sweep_uuid: Tuple[str, int]
    cuboids: Optional[Cuboids]
    is_ground: Optional[Tensor] = None

    @classmethod
    def from_rust(cls, sweep: rust.Sweep, avm: Optional[ArgoverseStaticMap] = None) -> Sweep:
        """Build a sweep from the Rust backend."""
        if sweep.annotations is not None:
            cuboids = Cuboids(_frame=sweep.annotations.to_pandas())
        else:
            cuboids = None

        city_SE3_ego = SE3_from_frame(frame=sweep.city_pose.to_pandas())
        lidar_xyzi = tensor_from_frame(sweep.lidar.to_pandas(), list(DEFAULT_LIDAR_TENSOR_FIELDS))

        if avm is not None:
            pcl_ego = lidar_xyzi[:, :3]
            pcl_city_1 = transform_points(city_SE3_ego.matrix().squeeze(), pcl_ego)
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


@dataclass(frozen=True)
class Flow:
    """Models scene flow pseudolabels for a LiDARSweep.

       Scene Flow pseudolabels come from the relative motion of all tracked objects
       between two sweeps.


    Args:
        flow: (N,3) Motion vectors (x,y,z) in meters.
        valid: (N,1) 1 if the flow was succesfuly estimated for that point 0 otherwise
        classes: (N,1) the semantic object class of each point (0 is background)
        ego_motion: SE3 the motion of the vehicle between the two sweeps
    """

    flow: FloatTensor
    valid: BoolTensor
    classes: ByteTensor
    dynamic: BoolTensor

    def __len__(self) -> int:
        """Return the number of LiDAR returns in the aggregated sweep."""
        return int(self.flow.shape[0]) if self.flow is not None else 0

    @classmethod
    def from_sweep_pair(cls, sweeps: Tuple[Sweep, Sweep]) -> Flow:
        """Create flow object from a pair of Sweeps."""
        poses = [sweep.city_SE3_ego for sweep in sweeps]
        ego1_SE3_ego0 = poses[1].inverse() * poses[0]
        if sweeps[0].cuboids is None or sweeps[1].cuboids is None:
            raise ValueError("Can only create flow from sweeps with annotations")
        else:
            cuboids: List[Cuboids] = [sweeps[0].cuboids, sweeps[1].cuboids]

        cuboid_maps = [cuboids_to_id_cuboid_map(cubs) for cubs in cuboids]
        pcs = [sweep.lidar_xyzi[:, :3] for sweep in sweeps]

        rigid_flow = (transform_points(ego1_SE3_ego0.matrix().squeeze(), pcs[0]) - pcs[0]).float().detach()
        flow = rigid_flow.clone()

        valid = torch.ones(len(pcs[0]), dtype=torch.bool)
        classes = torch.zeros(len(pcs[0]), dtype=torch.uint8)

        for id in cuboid_maps[0]:
            c0 = cuboid_maps[0][id]
            c0.length_m += 0.2  # the bounding boxes are a little too tight and some points are missed
            c0.width_m += 0.2
            obj_pts, obj_mask = [torch.from_numpy(arr) for arr in c0.compute_interior_points(pcs[0].numpy())]
            classes[obj_mask] = CATEGORY_MAP[str(c0.category)]

            if id in cuboid_maps[1]:
                c1 = cuboid_maps[1][id]
                c1_SE3_c0 = c1.dst_SE3_object.compose(c0.dst_SE3_object.inverse())
                obj_flow = torch.from_numpy(c1_SE3_c0.transform_point_cloud(obj_pts.numpy())) - obj_pts
                flow[obj_mask] = (obj_flow.float()).detach()
            else:
                valid[obj_mask] = 0

        dynamic = ((flow - rigid_flow) ** 2).sum(-1).sqrt() >= 0.05

        return cls(
            flow=torch.FloatTensor(flow),
            valid=torch.BoolTensor(valid),
            classes=torch.ByteTensor(classes),
            dynamic=torch.BoolTensor(dynamic),
        )


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
    dst_SE3_src.rotation._q.requires_grad_(False)
    dst_SE3_src.translation.requires_grad_(False)
    return dst_SE3_src


def cuboids_to_id_cuboid_map(cuboids: Cuboids) -> Dict[str, Cuboid]:
    """Create a mapping between track UUIDs and cuboids.

    Args:
        cuboids: the cuboids to transform into a mapping

    Returns:
        A dict with the UUIDs as keys and the coresponding cuboids as values.
    """
    ids = cuboids.track_uuid
    annotations_df_with_ts = cuboids._frame.assign(timestamp_ns=None)
    cuboid_list = CuboidList.from_dataframe(annotations_df_with_ts)

    cuboids_and_ids = dict(zip(ids, cuboid_list.cuboids))
    return cuboids_and_ids
