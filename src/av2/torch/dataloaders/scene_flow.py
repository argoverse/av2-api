"""Pytorch dataloader for the Argoverse 2 dataset."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import cached_property
from math import inf
from typing import Optional, Tuple

import pandas as pd
from torch.utils.data import Dataset

import av2._r as r
from av2.utils.typing import PathType

from .utils import Sweep

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SceneFlowDataloader(Dataset[Tuple[Sweep, Optional[Sweep]]]):
    """Pytorch dataloader for the sensor dataset.

    Args:
        root_dir: Path to the dataset directory.
        dataset_name: Dataset name.
        split_name: Name of the dataset split.
        min_annotation_range: Min Euclidean distance between the egovehicle origin and the annotation cuboid centers.
        max_annotation_range: Max Euclidean distance between the egovehicle origin and the annotation cuboid centers.
        min_lidar_range: Min Euclidean distance between the egovehicle origin and the lidar points.
        max_lidar_range: Max Euclidean distance between the egovehicle origin and the lidar points.
        min_interior_pts: Min number of points inside each annotation.
        num_accum_sweeps: Number of temporally accumulated sweeps (accounting for egovehicle motion).
        return_annotations: Boolean flag indicating whether to return annotations.
    """

    root_dir: PathType
    dataset_name: str
    split_name: str
    min_annotation_range: float = 0.0
    max_annotation_range: float = inf
    min_lidar_range: float = 0.0
    max_lidar_range: float = inf
    min_interior_pts: int = 0
    num_accum_sweeps: int = 1
    return_annotations: bool = False
    memory_map: bool = False

    _backend: r.Dataloader = field(init=False)
    _current_idx: int = 0

    def __post_init__(self) -> None:
        """Build the file index."""
        if self.return_annotations and self.dataset_name == "lidar":
            raise RuntimeError("The lidar dataset does not have annotations!")
        if not self.return_annotations and self.return_velocity_estimates:
            raise RuntimeError("with_annotations must be enabled to return annotations' velocities.")

        self._backend = r.Dataloader(
            str(self.root_dir),
            "sensor",
            self.split_name,
            self.dataset_name,
            self.num_accum_sweeps,
            self.memory_map,
        )

    @cached_property
    def file_index(self) -> pd.DataFrame:
        return self._backend.file_index.to_pandas()

    def __getitem__(self, index: int) -> Tuple[Sweep, Optional[Sweep]]:
        sweep = self._backend.get(index)
        next_sweep = None

        next_index = index + 1
        if next_index < len(self):
            candidate_log_id: str = self.file_index.loc[next_index, ["log_id"]].item()
            current_log_id = sweep.sweep_uuid[0]
            if candidate_log_id == current_log_id:
                next_sweep = Sweep.from_rust(self._backend.get(next_index))
        return Sweep.from_rust(sweep), next_sweep

    def __len__(self) -> int:
        return self._backend.__len__()

    def __iter__(self):
        return self

    def __next__(self):
        if self._current_idx >= self.__len__():
            raise StopIteration
        datum = self.__getitem__(self._current_idx)
        self._current_idx += 1
        return datum
