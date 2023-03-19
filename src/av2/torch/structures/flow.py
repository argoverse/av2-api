"""Pytorch flow module."""

from dataclasses import dataclass
from typing import Dict, Final, List, Optional, Tuple

import torch
from kornia.geometry.linalg import transform_points
from torch import BoolTensor, ByteTensor, FloatTensor, Tensor

from av2.evaluation.scene_flow.constants import CATEGORY_MAP
from av2.structures.cuboid import Cuboid, CuboidList
from av2.torch.structures.cuboids import Cuboids
from av2.torch.structures.sweep import Sweep


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


def cuboids_to_id_cuboid_map(cuboids: Cuboids) -> Dict[str, Cuboid]:
    """Create a mapping between track UUIDs and cuboids.

    Args:
        cuboids: Cuboids to transform into a mapping.

    Returns:
        A dict with the UUIDs as keys and the coresponding cuboids as values.
    """
    ids = cuboids.track_uuid
    annotations_df_with_ts = cuboids._frame.assign(timestamp_ns=None)
    cuboid_list = CuboidList.from_dataframe(annotations_df_with_ts)

    cuboids_and_ids = dict(zip(ids, cuboid_list.cuboids))
    return cuboids_and_ids
