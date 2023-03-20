"""PyTorch Lidar sub-module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd
import torch

from av2.torch import LIDAR_COLUMNS
from av2.torch.structures.utils import tensor_from_frame


@dataclass(frozen=True)
class Lidar:
    """Lidar sensor data structure.

    Args:
        _frame: Dataframe containing lidar coordinates and features.
    """

    _frame: pd.DataFrame

    def as_tensor(self, columns: Tuple[str, ...] = LIDAR_COLUMNS) -> torch.Tensor:
        """Return the lidar as a tensor with the specified columns.

        Args:
            columns: List of ordered column names.

        Returns:
            Tensor of lidar data.
        """
        return tensor_from_frame(self._frame, list(columns))
