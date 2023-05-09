"""PyTorch Cuboids sub-module."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from typing import List, Tuple

import pandas as pd
import torch
from kornia.geometry.conversions import euler_from_quaternion

import av2._r as rust

from .. import XYZLWH_QWXYZ_COLUMNS
from .utils import ndarray_from_frame, tensor_from_frame


class CuboidMode(str, Enum):
    """Cuboid parameterization modes."""

    XYZLWH_T = "XYZLWH_T"  # 1-DOF orientation.
    XYZLWH_QWXYZ = "XYZLWH_QWXYZ"  # 3-DOF orientation.
    XYZ_VERTICES = "XYZ_VERTICES"  # Vertex representation (eight vertices).


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

    def as_tensor(self, cuboid_mode: CuboidMode = CuboidMode.XYZLWH_T) -> torch.Tensor:
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
        if cuboid_mode == CuboidMode.XYZLWH_T:
            xyzlwh_qwxyz = tensor_from_frame(self._frame, list(XYZLWH_QWXYZ_COLUMNS))
            quat_wxyz = xyzlwh_qwxyz[:, 6:10]
            w, x, y, z = (
                quat_wxyz[:, 0],
                quat_wxyz[:, 1],
                quat_wxyz[:, 2],
                quat_wxyz[:, 3],
            )
            _, _, yaw = euler_from_quaternion(w, x, y, z)
            return torch.concat([xyzlwh_qwxyz[:, :6], yaw[:, None]], dim=-1)
        elif cuboid_mode == CuboidMode.XYZLWH_QWXYZ:
            return xyzlwh_qwxyz
        elif cuboid_mode == CuboidMode.XYZ_VERTICES:
            xyzlwh_qwxyz = ndarray_from_frame(self._frame, list(XYZLWH_QWXYZ_COLUMNS))
            xyz_vertices = rust.cuboids_to_vertices(xyzlwh_qwxyz)
            return torch.as_tensor(xyz_vertices)
        else:
            raise NotImplementedError(
                f"{cuboid_mode} orientation mode is not implemented."
            )

    def compute_interior_points_mask(self, points_xyz_m: torch.Tensor) -> torch.Tensor:
        """Compute a boolean mask for each cuboid indicating whether each point is interior to the geometry.

        Notation:
            N: Number of points.
            M: Number of cuboids.

        Args:
            points_xyz_m: (N,3) Points to filter.

        Returns:
            (N,M) Boolean array indicating which points are interior.
        """
        vertices_dst_xyz = self.as_tensor(CuboidMode.XYZ_VERTICES)
        is_interior = rust.compute_interior_points_mask(
            points_xyz_m[:, :3].numpy(), vertices_dst_xyz.numpy()
        )
        return torch.as_tensor(is_interior)
