"""Rust backend typing stubs."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import polars as pl

@dataclass
class DataLoader:
    root_dir: str
    dataset_name: str
    dataset_type: str
    split_name: str
    num_accumulated_sweeps: int
    memory_map: bool

    file_index: pl.DataFrame = field(init=False)

    def get(self, index: int) -> Sweep: ...
    def get_synchronized_images(self, index: int) -> List[TimeStampedImage]: ...
    def __len__(self) -> int: ...

@dataclass
class Sweep:
    city_pose: pl.DataFrame
    lidar: pl.DataFrame
    sweep_uuid: Tuple[str, int]
    cuboids: Optional[pl.DataFrame]

@dataclass
class Intrinsics:
    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float
    width_px: int
    height_px: int

@dataclass
class SE3:
    rotation: np.array
    translation: np.array

@dataclass
class PinholeCamera:
    ego_se3_cam: SE3
    intrinsics: Intrinsics
    camera_name: str

@dataclass
class TimeStampedImage:
    image: np.array
    camera_model: PinholeCamera
    timestamp_ns: int
