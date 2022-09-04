"""Pytorch dataloader for the Argoverse 2 dataset."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, List, Tuple, Union, cast

import numpy as np
import pandas as pd
from pyarrow import feather
from upath import UPath

logger = logging.Logger(__file__)

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
MAX_STR_LEN: Final[int] = 32

try:
    import torch
    from torch import Tensor
    from torch.utils.data import Dataset

    @dataclass
    class Sweep:
        """Stores the annotations and lidar for one sweep."""

        annotations_categories: Tuple[str, ...]
        annotations_cuboids: Tensor  # torch.float32
        lidar: Tensor  # torch.float32

    @dataclass
    class Av2(Dataset[Sweep]):
        """Pytorch dataloader for the sensor dataset."""

        dataset_dir: Union[Path, UPath]
        split: str
        ordered_annotations_cols: Tuple[str, ...] = DEFAULT_ANNOTATIONS_COLS
        annotations_pose_mode: str = "YAW"  # TODO: Add pose modes.
        flush_file_index: bool = False

        file_index: List[Tuple[str, int]] = field(init=False)

        def __post_init__(self) -> None:
            """Build the file index."""
            self._build_file_index()

        @property
        def file_index_path(self) -> Union[Path, UPath]:
            """Return the file index path."""
            return Path.home() / ".cache" / "av2" / "torch" / "file_index.feather"

        @property
        def split_dir(self) -> Union[Path, UPath]:
            """Sensor dataset split directory."""
            return self.dataset_dir / self.split

        def annotations_path(self, log_id: str) -> Union[Path, UPath]:
            """Return the annotations at the specified log id.

            Args:
                log_id: Unique log identifier.

            Returns:
                Annotations path for the entire log.
            """
            return self.split_dir / log_id / "annotations.feather"

        def lidar_path(self, log_id: str, timestamp_ns: int) -> Union[Path, UPath]:
            """Return the lidar path at the specified log id and timestamp.

            Args:
                log_id: Unique log identifier.
                timestamp_ns: Lidar timestamp in nanoseconds.

            Returns:
                Lidar path at the log id and timestamp.
            """
            return self.split_dir / log_id / "sensors" / "lidar" / f"{timestamp_ns}.feather"

        def key(self, index: int) -> Tuple[str, int]:
            """Return key at the given index."""
            return self.file_index[index]

        def __getitem__(self, index: int) -> Sweep:
            """Return annotations and lidar for one sweep.

            Args:
                index: Integer dataset index.

            Returns:
                Sweep object containing annotations and lidar.
            """
            annotations_categories, annotations_cuboids = self.read_annotations(index)
            lidar = self.read_lidar(index)
            return Sweep(annotations_categories, annotations_cuboids, lidar)

        def _build_file_index(self) -> None:
            """Initialize the key to path mapping."""
            if not self.flush_file_index and self.file_index_path.exists():
                dataframe = _read_feather(self.file_index_path)
                self.file_index = [(cast(str, key[0]), cast(int, key[1])) for key in dataframe.to_numpy().tolist()]
            else:
                logger.info("Initializing file index. This may take a few minutes.")
                paths = sorted(self.split_dir.glob(LIDAR_GLOB_PATTERN))
                self.file_index = [(key.parts[-4], int(key.stem)) for key in paths]

                self.file_index_path.parent.mkdir(parents=True, exist_ok=True)
                dataframe = pd.DataFrame.from_records(self.file_index, columns=["log_id", "timestamp_ns"])
                dataframe.to_feather(self.file_index_path)

        def read_annotations(self, index: int) -> Tuple[Tuple[str, ...], Tensor]:
            """Read the sweep annotations.

            Args
                key: Unique key (log_id, timestamp_ns).

            Returns:
                Tensor of annotations.
            """
            log_id, timestamp_ns = self.key(index)
            annotations_path = self.annotations_path(log_id)
            annotations = _read_feather(annotations_path)
            annotations = self._populate_annotations_velocity(index, annotations)

            query = (annotations["num_interior_pts"] > 0) & (annotations["timestamp_ns"] == timestamp_ns)
            annotations = annotations.loc[query, list(self.ordered_annotations_cols)].reset_index(drop=True)

            annotations_categories = tuple(annotations["category"].to_numpy().tolist())
            annotations = annotations.drop("category", axis=1)
            return annotations_categories, torch.as_tensor(annotations.to_numpy(dtype=np.float32))

        def _populate_annotations_velocity(self, index: int, annotations: pd.DataFrame) -> pd.DataFrame:
            """Populate the annotations with their estimated velocities.

            Args:
                index: Current file index location.
                annotations: DataFrame of annotations loaded from a feather file.

            Returns:
                DataFrame populated with velocities.
            """
            object_centers = annotations[["tx_m", "ty_m", "tz_m"]].to_numpy()

            self._build_temporal_window(index, annotations)
            return annotations

        def _build_temporal_window(self, index: int, annotations: pd.DataFrame) -> pd.DataFrame:
            # TODO: Add temporal window for velocity computation.
            # previous_frame = self.key(index - 1)
            # current_frame = self.key(index)
            # next_frame = self.key(index + 1)
            return annotations

        def read_lidar(self, index: int) -> Tensor:
            """Read the lidar sweep.

            Args:
                key: Unique key (log_id, timestamp_ns).

            Returns:
                Tensor of annotations.
            """
            log_id, timestamp_ns = self.key(index)
            lidar_path = self.lidar_path(log_id, timestamp_ns)
            dataframe = _read_feather(lidar_path)
            return torch.as_tensor(dataframe.to_numpy(dtype=np.float32))

    def _read_feather(path: Union[Path, UPath]) -> pd.DataFrame:
        """Read an Apache feather file.

        Args:
            path: Path to the feather file.

        Returns:
            Pandas dataframe containing the arrow data.
        """
        with path.open("rb") as file_handle:  # type: ignore
            dataframe: pd.DataFrame = feather.read_feather(file_handle)
        return dataframe

except ImportError as e:
    print("Please install Pytorch to use this dataloader.")
