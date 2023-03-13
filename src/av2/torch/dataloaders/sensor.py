"""Pytorch dataloader for the Argoverse 2 dataset."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from math import inf

from torch.utils.data import Dataset

import av2._r as r
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
        num_accum_sweeps: Number of temporally accumulated sweeps (accounting for egovehicle motion).
        memory_mapped: Boolean flag indicating whether to memory map the dataframes.
    """

    root_dir: PathType
    dataset_name: str
    split_name: str
    num_accum_sweeps: int = 1
    memory_map: bool = False

    _backend: r.Dataloader = field(init=False)
    _current_idx: int = 0

    def __post_init__(self) -> None:
        """Initialize Rust backend."""
        self._backend = r.Dataloader(
            str(self.root_dir),
            "sensor",
            self.split_name,
            self.dataset_name,
            self.num_accum_sweeps,
            self.memory_map,
        )

    def __getitem__(self, index: int) -> Sweep:
        """Get a sweep from the sensor dataset."""
        sweep = self._backend.get(index)
        return Sweep.from_rust(sweep)

    def __len__(self) -> int:
        """Length of the sensor dataset (number of sweeps)."""
        return self._backend.__len__()

    def __iter__(self) -> Dataloader:
        """Iterate method for the dataloader."""
        return self

    def __next__(self):
        """Return the next sweep."""
        if self._current_idx >= self.__len__():
            raise StopIteration
        datum = self.__getitem__(self._current_idx)
        self._current_idx += 1
        return datum
