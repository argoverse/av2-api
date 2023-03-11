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

    def get(self, index: int) -> Sweep: ...

@dataclass
class Sweep:
    annotations: pl.DataFrame
    lidar: pl.DataFrame
    sweep_uuid: Tuple[str, int]
