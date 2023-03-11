"""Pytorch dataloader for the Argoverse 2 dataset."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from math import inf
from typing import Final

import torch
from torch import Tensor
from torch.utils.data import Dataset

import av2._r as r
from av2.utils.typing import PathType
from av2.torch.dataloaders.utils import Annotations, Lidar

from .utils import Sweep

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

XYZ_FIELDS: Final = ("x", "y", "z")
LIDAR_GLOB_PATTERN: Final = "sensors/lidar/*.feather"


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

    _backend: r.SensorDataset = field(init=False)

    def __post_init__(self) -> None:
        """Build the file index."""
        if self.return_annotations and self.dataset_name == "lidar":
            raise RuntimeError("The lidar dataset does not have annotations!")
        if not self.return_annotations and self.return_velocity_estimates:
            raise RuntimeError("with_annotations must be enabled to return annotations' velocities.")

        self._backend = r.Dataloader(
            str(self.root_dir), "sensor", self.split_name, self.dataset_name, self.num_accum_sweeps
        )

    def __repr__(self) -> str:
        """Dataloader info."""
        info = "Dataloader configuration settings:\n"
        for key, value in sorted(self.items()):
            if key == "file_index":
                continue
            info += f"\t{key}: {value}\n"
        return info

    def __getitem__(self, index) -> Sweep:
        datum = self._backend.get(index)
        annotations = Annotations(dataframe=datum.annotations.to_pandas())
        lidar = Lidar(dataframe=datum.lidar.to_pandas())
        sweep = Sweep(annotations=annotations, lidar=lidar, sweep_uuid=datum.sweep_uuid)
        return sweep
