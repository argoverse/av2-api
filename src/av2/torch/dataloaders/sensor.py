"""Pytorch dataloader for the Argoverse 2 dataset."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from math import inf

from torch.utils.data import Dataset

import av2._r as r
from av2.torch.dataloaders.utils import Annotations, Lidar
from av2.utils.typing import PathType

from .utils import Sweep

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Dataloader(Dataset[Sweep]):
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
        file_caching_mode: File caching mode.
        return_annotations: Boolean flag indicating whether to return annotations.
        return_velocity_estimates: Boolean flag indicating whether to return annotations' velocity estimates.
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
    return_velocity_estimates: bool = False

    _backend: r.Dataloader = field(init=False)
    _current_idx: int = 0

    def __post_init__(self) -> None:
        """Build the file index."""
        if self.return_annotations and self.dataset_name == "lidar":
            raise RuntimeError("The lidar dataset does not have annotations!")
        if not self.return_annotations and self.return_velocity_estimates:
            raise RuntimeError("with_annotations must be enabled to return annotations' velocities.")

        self._backend = r.Dataloader(
            str(self.root_dir), "sensor", self.split_name, self.dataset_name, self.num_accum_sweeps
        )

    def __getitem__(self, index: int) -> Sweep:
        sweep = self._backend.get(index)
        annotations = Annotations(dataframe=sweep.annotations.to_pandas())
        lidar = Lidar(dataframe=sweep.lidar.to_pandas())
        sweep = Sweep(annotations=annotations, lidar=lidar, sweep_uuid=sweep.sweep_uuid)

        self._current_idx += 1
        return sweep

    def __len__(self) -> int:
        return self._backend.__len__()

    def __iter__(self):
        return self

    def __next__(self):
        if self._current_idx >= self.__len__():
            raise StopIteration
        return self.__getitem__(self._current_idx)
