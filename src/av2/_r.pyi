"""Rust backend typing stubs."""

from dataclasses import dataclass
from typing import Tuple

import polars as pl

@dataclass
class Dataloader:
    root_dir: str
    dataset_name: str
    dataset_type: str
    split_name: str
    num_accum_sweeps: int
    memory_map: bool

    def get(self, index: int) -> Sweep: ...
    def __len__(self) -> int: ...

@dataclass
class Sweep:
    annotations: pl.DataFrame
    city_pose: pl.DataFrame
    lidar: pl.DataFrame
    sweep_uuid: Tuple[str, int]
