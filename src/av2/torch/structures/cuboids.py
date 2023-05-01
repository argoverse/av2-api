"""PyTorch Cuboids sub-module."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from typing import List

import pandas as pd
import torch
from kornia.geometry.conversions import euler_from_quaternion

from .. import XYZLWH_QWXYZ_COLUMNS
from .utils import tensor_from_frame


class CuboidMode(str, Enum):
    """Cuboid parameterization modes."""

    XYZLWH_T = "XYZLWH_T"  # 1-DOF orientation.
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
        xyzlwh_qwxyz = tensor_from_frame(self._frame, list(XYZLWH_QWXYZ_COLUMNS))
        if cuboid_mode == CuboidMode.XYZLWH_T:
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
        else:
            raise NotImplementedError(
                f"{cuboid_mode} orientation mode is not implemented."
            )
