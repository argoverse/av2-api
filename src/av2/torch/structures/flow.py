"""PyTorch flow sub-module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Final, Tuple

import torch
from kornia.geometry.linalg import transform_points
from torch import BoolTensor, ByteTensor, FloatTensor

from av2.evaluation.scene_flow.constants import (
    CATEGORY_TO_INDEX,
    SCENE_FLOW_DYNAMIC_THRESHOLD,
)
from av2.structures.cuboid import Cuboid, CuboidList
from av2.torch.structures.cuboids import Cuboids
from av2.torch.structures.sweep import Sweep

BOUNDING_BOX_EXPANSION: Final = 0.2


@dataclass(frozen=True)
class Flow:
    """Models scene flow pseudo-labels for a lidar sweep.

       Scene Flow pseudolabels come from the relative motion of all tracked objects
       between two sweeps.


    Args:
        flow: (N,3) Motion vectors (x,y,z) in meters.
        is_valid: (N,) 1 if the flow was successfully estimated for that point 0 otherwise
        category_indices: (N,) the semantic object class of each point (0 is background)
        is_dynamic: (N,) 1 if the point is considered dynamic 0 otherwise
    """

    flow: FloatTensor
    is_valid: BoolTensor
    category_indices: ByteTensor
    is_dynamic: BoolTensor

    def __len__(self) -> int:
        """Return the number of LiDAR returns in the aggregated sweep."""
        return int(self.flow.shape[0])

    @classmethod
    def from_sweep_pair(cls, sweeps: Tuple[Sweep, Sweep]) -> Flow:
        """Create flow object from a pair of Sweeps.

        Args:
            sweeps: Pair of sweeps to compute the flow between.

        Returns:
            Flow object.

        Raises:
            ValueError: If the sweeps do not have annotations loaded.
        """
        current_sweep, next_sweep = (sweeps[0], sweeps[1])
        if current_sweep.cuboids is None or next_sweep.cuboids is None:
            raise ValueError("Can only create flow from sweeps with annotations")
        else:
            current_cuboids, next_cuboids = current_sweep.cuboids, next_sweep.cuboids
        city_SE3_ego0, city_SE3_ego1 = (
            current_sweep.city_SE3_ego,
            next_sweep.city_SE3_ego,
        )
        ego1_SE3_ego0 = city_SE3_ego1.inverse() * city_SE3_ego0

        current_cuboid_map = cuboids_to_id_cuboid_map(current_cuboids)
        next_cuboid_map = cuboids_to_id_cuboid_map(next_cuboids)

        current_pc = current_sweep.lidar.as_tensor()[:, :3]
        rigid_flow = (
            (transform_points(ego1_SE3_ego0.matrix(), current_pc[None])[0] - current_pc)
            .float()
            .detach()
        )
        flow = rigid_flow.clone()

        is_valid = torch.ones(len(current_pc), dtype=torch.bool)
        category_inds = torch.zeros(len(current_pc), dtype=torch.uint8)

        for id in current_cuboid_map:
            c0 = current_cuboid_map[id]
            c0.length_m += BOUNDING_BOX_EXPANSION  # the bounding boxes are a little too tight sometimes
            c0.width_m += BOUNDING_BOX_EXPANSION
            obj_pts_npy, obj_mask_npy = c0.compute_interior_points(current_pc.numpy())
            obj_pts, obj_mask = torch.as_tensor(
                obj_pts_npy, dtype=torch.float32
            ), torch.as_tensor(obj_mask_npy)
            category_inds[obj_mask] = CATEGORY_TO_INDEX[str(c0.category)]

            if id in next_cuboid_map:
                c1 = next_cuboid_map[id]
                c1_SE3_c0 = c1.dst_SE3_object.compose(c0.dst_SE3_object.inverse())
                flow[obj_mask] = (
                    torch.as_tensor(
                        c1_SE3_c0.transform_point_cloud(obj_pts_npy),
                        dtype=torch.float32,
                    )
                    - obj_pts
                )
            else:
                is_valid[obj_mask] = 0

        dynamic_norm = torch.linalg.vector_norm(flow - rigid_flow, dim=-1)
        is_dynamic: BoolTensor = dynamic_norm >= SCENE_FLOW_DYNAMIC_THRESHOLD

        return cls(
            flow=torch.FloatTensor(flow),
            is_valid=torch.BoolTensor(is_valid),
            category_indices=torch.ByteTensor(category_inds),
            is_dynamic=torch.BoolTensor(is_dynamic),
        )


def cuboids_to_id_cuboid_map(cuboids: Cuboids) -> Dict[str, Cuboid]:
    """Create a mapping between track UUIDs and cuboids.

    Args:
        cuboids: Cuboids to transform into a mapping.

    Returns:
        A dict with the UUIDs as keys and the corresponding cuboids as values.
    """
    ids = cuboids.track_uuid
    annotations_df_with_ts = cuboids._frame.assign(timestamp_ns=None)
    cuboid_list = CuboidList.from_dataframe(annotations_df_with_ts)

    cuboids_and_ids = dict(zip(ids, cuboid_list.cuboids))
    return cuboids_and_ids
