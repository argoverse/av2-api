"""Pytorch dataloader for the scene flow task."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import cached_property
from math import inf
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from kornia.geometry.liegroup import Se3
from torch.utils.data import Dataset

import av2._r as rust
from av2.map.map_api import ArgoverseStaticMap
from av2.utils.typing import PathType

from .utils import Flow, Sweep

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SceneFlowDataloader(Dataset[Tuple[Sweep, Sweep, Se3, Optional[Flow]]]):
    """Pytorch dataloader for the sensor dataset.

    Args:
        root_dir: Path to the dataset directory.
        dataset_name: Dataset name (e.g., "av2").
        split_name: Name of the dataset split (e.g., "train").
        num_accum_sweeps: Number of temporally accumulated sweeps (accounting for ego-vehicle motion).
        memory_mapped: Boolean flag indicating whether to memory map the dataframes.
    """

    root_dir: PathType
    dataset_name: str
    split_name: str
    num_accumulated_sweeps: int = 1
    memory_mapped: bool = False

    _backend: rust.Dataloader = field(init=False)
    _current_idx: int = 0

    def __post_init__(self) -> None:
        """Initialize Rust backend."""
        self._backend = rust.Dataloader(
            str(self.root_dir),
            self.dataset_name,
            "sensor",
            self.split_name,
            self.num_accumulated_sweeps,
            self.memory_mapped,
        )
        self.data_dir = Path(self.root_dir) / self.dataset_name / "sensor" / self.split_name

    @cached_property
    def file_index(self) -> pd.DataFrame:
        """File index dataframe composed of (log_id, timestamp_ns)."""
        return self._backend.file_index.to_pandas()

    @cached_property
    def index_map(self) -> List[int]:
        """Create a mapping between indicies in this dataloader and the underlying one."""
        inds = []
        N = self._backend.__len__()
        for i in range(N):
            if i + 1 < N:
                next_log_id = self.file_index.loc[i + 1, ["log_id"]].item()
                current_log_id = self.file_index.loc[i, ["log_id"]].item()
                if current_log_id == next_log_id:
                    inds.append(i)
        return inds

    def get_log_id(self, index: int) -> str:
        """Return the log name for a given sweep index."""
        return str(self.file_index.loc[index, ["log_id"]].item())

    def __getitem__(self, index: int) -> Tuple[Sweep, Sweep, Se3, Optional[Flow]]:
        """Get a pair of sweeps, ego motion, and flow if annotations are available."""
        backend_index = self.index_map[index]
        log = self.file_index.loc[index, ["log_id"]].item()
        log_dir_path = log_map_dirpath = self.data_dir / log
        log_map_dirpath = self.data_dir / log / "map"
        avm = ArgoverseStaticMap.from_map_dir(log_map_dirpath, build_raster=True)

        sweep = Sweep.from_rust(self._backend.get(backend_index), avm=avm)
        next_sweep = Sweep.from_rust(self._backend.get(backend_index + 1), avm=avm)

        if sweep.cuboids is not None:
            flow = Flow.from_sweep_pair((sweep, next_sweep))
        else:
            flow = None

        ego_motion = next_sweep.city_SE3_ego.inverse() * sweep.city_SE3_ego
        ego_motion.rotation._q.requires_grad_(False)
        ego_motion.translation.requires_grad_(False)

        return sweep, next_sweep, ego_motion, flow

    def __len__(self) -> int:
        """Length of the scene flow dataset (number of pairs of sweeps)."""
        return len(self.index_map)

    def __iter__(self) -> SceneFlowDataloader:
        """Iterate method for the dataloader."""
        return self

    def __next__(self) -> Tuple[Sweep, Sweep, Se3, Optional[Flow]]:
        """Return a tuple of sweeps for scene flow."""
        if self._current_idx >= self.__len__():
            raise StopIteration
        datum = self.__getitem__(self._current_idx)
        self._current_idx += 1
        return datum
