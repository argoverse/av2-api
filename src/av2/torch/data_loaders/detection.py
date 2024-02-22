"""PyTorch data-loader for 3D object detection task."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Tuple

import torch
from kornia.geometry.camera.pinhole import PinholeCamera, PinholeCamerasList
from torch.utils.data import Dataset

import av2._r as rust
from av2.torch.structures.time_stamped_image import TimeStampedImage
from av2.utils.typing import PathType

from ..structures.sweep import Sweep

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DetectionDataLoader(Dataset[Sweep]):  # type: ignore
    """PyTorch data-loader for the sensor dataset.

    The sensor dataset should exist somewhere such as `~/data/datasets/{dataset_name}/{dataset_type}/{split_name}`,
    where
        dataset_name = "av2",
        dataset_type = "sensor,
        split_name = "train".

    This data-loader backend is implemented in Rust for speed. Each iteration will yield a new sweep.

    Args:
        root_dir: Path to the dataset directory.
        dataset_name: Dataset name (e.g., "av2").
        split_name: Name of the dataset split (e.g., "train").
        num_accumulated_sweeps: Number of temporally accumulated sweeps (accounting for ego-vehicle motion).
        memory_mapped: Boolean flag indicating whether to memory map the dataframes.
    """

    root_dir: PathType
    dataset_name: str
    split_name: str
    num_accumulated_sweeps: int = 1
    memory_mapped: bool = False

    _backend: rust.DataLoader = field(init=False)
    _current_sweep_index: int = 0

    def __post_init__(self) -> None:
        """Initialize Rust backend."""
        self._backend = rust.DataLoader(
            str(self.root_dir),
            self.dataset_name,
            "sensor",
            self.split_name,
            self.num_accumulated_sweeps,
            self.memory_mapped,
        )

    def __getitem__(self, sweep_index: int) -> Sweep:
        """Get a sweep from the sensor dataset."""
        sweep = self._backend.get(sweep_index)
        return Sweep.from_rust(sweep)

    def __len__(self) -> int:
        """Length of the sensor dataset (number of sweeps)."""
        return self._backend.__len__()

    def __iter__(self) -> DetectionDataLoader:
        """Iterate method for the data-loader."""
        return self

    def __next__(self) -> Sweep:
        """Return the next sweep."""
        if self._current_sweep_index >= self.__len__():
            raise StopIteration
        datum = self.__getitem__(self._current_sweep_index)
        self._current_sweep_index += 1
        return datum

    def get_synchronized_images(self, sweep_index: int) -> List[TimeStampedImage]:
        """Get the synchronized ring images associated with the sweep index."""
        time_stamped_images = self._backend.get_synchronized_images(sweep_index)
        time_stamped_image_list = []
        for ts_image in time_stamped_images:
            image = torch.as_tensor(ts_image.image)
            camera_model_rs = ts_image.camera_model

            intrinsics = torch.as_tensor(camera_model_rs.intrinsics.k)
            extrinsics = torch.as_tensor(camera_model_rs.extrinsics)

            height = torch.as_tensor([image.shape[1]])
            width = torch.as_tensor([image.shape[2]])
            camera_model = PinholeCamera(
                intrinsics=intrinsics, extrinsics=extrinsics, height=height, width=width
            )

            timestamp_ns = ts_image.timestamp_ns
            time_stamped_image = TimeStampedImage(
                image=image, camera_model=camera_model, timestamp_ns=timestamp_ns
            )
            time_stamped_image_list.append(time_stamped_image)
        return time_stamped_image_list
