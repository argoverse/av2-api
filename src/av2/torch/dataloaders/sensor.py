"""Pytorch dataloader for the Argoverse 2 dataset."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, List, Tuple, Union

import numpy as np
import pandas as pd
from pyarrow import feather
from upath import UPath

try:
    import torch
    from torch import Tensor
    from torch.utils.data import Dataset

except ImportError as e:
    logging.error(e)

DEFAULT_ANNOTATIONS_COLS: Final[Tuple[str, ...]] = (
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
    "category",
)
LIDAR_GLOB_PATTERN: Final[str] = "*/sensors/lidar/*"


@dataclass
class Sweep:
    """Stores the annotations and lidar for one sweep."""

    annotations: Tensor
    lidar: Tensor


@dataclass
class Av2(Dataset[Sweep]):
    """Pytorch dataloader for the sensor dataset."""

    dataset_dir: Union[Path, UPath]
    split: str
    ordered_annotations_cols: Tuple[str, ...] = DEFAULT_ANNOTATIONS_COLS
    annotations_pose_mode: str = "YAW"

    keys: List[Tuple[str, int]] = field(init=False)

    def __post_init__(self) -> None:
        """Initialize the key mappings."""
        self._init_key_mapping()

    @property
    def split_dir(self) -> Union[Path, UPath]:
        """Sensor dataset split directory."""
        return self.dataset_dir / self.split

    def __getitem__(self, index: int) -> Sweep:
        """Return annotations and lidar for one sweep.

        Args:
            index: Integer dataset index.

        Returns:
            Sweep object containing annotations and lidar.
        """
        key = self.keys[index]
        annotations = self.read_annotations(key)
        lidar = self.read_lidar(key)
        return Sweep(annotations, lidar)

    def _init_key_mapping(self) -> None:
        """Initialize the key to path mapping."""
        fpaths = self.split_dir.glob(LIDAR_GLOB_PATTERN)
        self.keys = [(fpath.parts[-4], int(fpath.stem)) for fpath in fpaths]

    def read_annotations(self, key: Tuple[str, int]) -> Tensor:
        """Read the sweep annotations.

        Args:
            key: Unique key (log_id, timestamp_ns).

        Returns:
            Tensor of annotations.
        """
        log_id, timestamp_ns = key
        annotations_fpath = self.split_dir / log_id / "annotations.feather"
        annotations = _read_feather(annotations_fpath)
        query = (annotations["num_interior_pts"] > 0) & (annotations["timestamp_ns"] == timestamp_ns)
        annotations = annotations.loc[query, list(self.ordered_annotations_cols)].reset_index(drop=True)

        if "category" in annotations.columns:
            category = annotations["category"]
            annotations = annotations.drop("category", axis=1)
        return torch.as_tensor(annotations.to_numpy(dtype=np.float32))

    def read_lidar(self, key: Tuple[str, int]) -> Tensor:
        """Read the lidar sweep.

        Args:
            key: Unique key (log_id, timestamp_ns).

        Returns:
            Tensor of annotations.
        """
        log_id, timestamp_ns = key
        lidar_fpath = self.split_dir / log_id / "sensors" / "lidar" / f"{timestamp_ns}.feather"
        dataframe = _read_feather(lidar_fpath)
        return torch.as_tensor(dataframe.to_numpy(dtype=np.float32))


def _read_feather(path: Union[Path, UPath]) -> pd.DataFrame:
    """Read an Apache feather file.

    Args:
        path: Path to the feather file.

    Returns:
        Pandas dataframe containing the arrow data.
    """
    with path.open("rb") as file_handle:
        dataframe: pd.DataFrame = feather.read_feather(file_handle)
    return dataframe
