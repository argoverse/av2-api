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
from kornia.geometry.conversions import convert_points_from_homogeneous, convert_points_to_homogeneous
from kornia.geometry.liegroup import Se3, So3
from kornia.geometry.quaternion import Quaternion
from torch import BoolTensor, ByteTensor, FloatTensor, Tensor

import av2._r as rust
from av2.geometry.se3 import SE3
from av2.map.map_api import ArgoverseStaticMap
from av2.structures.cuboid import Cuboid, CuboidList
from av2.utils.typing import NDArrayBool, NDArrayByte, NDArrayFloat

CATEGORY_MAP = {
    "ANIMAL": 0,
    "ARTICULATED_BUS": 1,
    "BICYCLE": 2,
    "BICYCLIST": 3,
    "BOLLARD": 4,
    "BOX_TRUCK": 5,
    "BUS": 6,
    "CONSTRUCTION_BARREL": 7,
    "CONSTRUCTION_CONE": 8,
    "DOG": 9,
    "LARGE_VEHICLE": 10,
    "MESSAGE_BOARD_TRAILER": 11,
    "MOBILE_PEDESTRIAN_CROSSING_SIGN": 12,
    "MOTORCYCLE": 13,
    "MOTORCYCLIST": 14,
    "OFFICIAL_SIGNALER": 15,
    "PEDESTRIAN": 16,
    "RAILED_VEHICLE": 17,
    "REGULAR_VEHICLE": 18,
    "SCHOOL_BUS": 19,
    "SIGN": 20,
    "STOP_SIGN": 21,
    "STROLLER": 22,
    "TRAFFIC_LIGHT_TRAILER": 23,
    "TRUCK": 24,
    "TRUCK_CAB": 25,
    "VEHICULAR_TRAILER": 26,
    "WHEELCHAIR": 27,
    "WHEELED_DEVICE": 28,
    "WHEELED_RIDER": 29,
}


MAX_STR_LEN: Final[int] = 32

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
    """Orientation (pose) modes for the ground truth annotations."""

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
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        """Return the annotations as a tensor.

        Args:
            field_ordering: Feature ordering for the tensor.
            dtype: Target datatype for casting.

        Returns:
            (N,K) tensor where N is the number of annotations and K
                is the number of annotation fields.
        """
        dataframe_npy = self.dataframe.loc[:, list(field_ordering)].to_numpy()
        return torch.as_tensor(dataframe_npy, dtype=dtype)


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
        is_ground: Tensor of boolean values indicatind which points belong to the ground
    """

    annotations: Optional[Annotations]
    city_SE3_ego: Se3
    lidar_xyzi: Tensor
    sweep_uuid: Tuple[str, int]
    is_ground: Optional[Tensor] = None

    @classmethod
    def from_rust(cls, sweep: rust.Sweep, avm: Optional[ArgoverseStaticMap] = None) -> Sweep:
        """Build a sweep from the Rust backend."""
        if sweep.annotations is not None:
            annotations = Annotations(dataframe=sweep.annotations.to_pandas())
        else:
            annotations = None
        city_SE3_ego = frame_to_SE3(frame=sweep.city_pose.to_pandas())
        lidar_xyzi = frame_to_tensor(sweep.lidar.to_pandas())

        if avm is not None:
            pcl_ego = lidar_xyzi[:, :3]
            pcl_city_1 = apply_se3(city_SE3_ego, pcl_ego)
            is_ground = torch.from_numpy(avm.get_ground_points_boolean(pcl_city_1.numpy()).astype(bool))
        else:
            is_ground = None

        return cls(
            annotations=annotations,
            city_SE3_ego=city_SE3_ego,
            lidar_xyzi=lidar_xyzi,
            sweep_uuid=sweep.sweep_uuid,
            is_ground=is_ground,
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
        if sweeps[0].annotations is None or sweeps[1].annotations is None:
            raise ValueError("Can only create flow from sweeps with annotations")
        else:
            annotations: List[Annotations] = [sweeps[0].annotations, sweeps[1].annotations]

        cuboids = [annotations_to_id_cudboid_map(anno) for anno in annotations]
        pcs = [sweep.lidar_xyzi[:, :3] for sweep in sweeps]

        rigid_flow = (apply_se3(ego1_SE3_ego0, pcs[0]) - pcs[0]).float().detach()
        flow = rigid_flow.clone()

        valid = torch.ones(len(pcs[0]), dtype=torch.bool)
        classes = torch.zeros(len(pcs[0]), dtype=torch.uint8)

        for id in cuboids[0]:
            c0 = cuboids[0][id]
            c0.length_m += 0.2  # the bounding boxes are a little too tight and some points are missed
            c0.width_m += 0.2
            obj_pts, obj_mask = [torch.from_numpy(arr) for arr in c0.compute_interior_points(pcs[0].numpy())]
            classes[obj_mask] = CATEGORY_MAP[str(c0.category)] + 1

            if id in cuboids[1]:
                c1 = cuboids[1][id]
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


@torch.no_grad()
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
    dst_SE3_src.rotation._q.requires_grad_(False)
    dst_SE3_src.translation.requires_grad_(False)
    return dst_SE3_src


def annotations_to_id_cuboid_map(annotations: Annotations) -> Dict[str, Cuboid]:
    """Create a mapping between track UUIDs and cuboids.

    Args:
        annotations: the annotations to transform into cuboids

    Returns:
        A dict with the UUIDs as keys and the coresponding cuboids as values.
    """
    ids = annotations.dataframe.track_uuid.to_numpy()
    annotations_df_with_ts = annotations.dataframe.assign(timestamp_ns=None)
    cuboid_list = CuboidList.from_dataframe(annotations_df_with_ts)

    cuboids_and_ids = dict(zip(ids, cuboid_list.cuboids))
    return cuboids_and_ids


def apply_se3(se3: Se3, pts: Tensor) -> Tensor:
    """Apply an Se3 transformation to a tensor of points (N x 3)."""
    mat = se3.matrix()
    if len(mat.shape) > len(pts.shape):
        mat = mat.squeeze()
    return convert_points_from_homogeneous(convert_points_to_homogeneous(pts) @ mat.T)
