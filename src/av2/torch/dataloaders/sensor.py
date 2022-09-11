"""Pytorch dataloader for the Argoverse 2 dataset."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Final, List, Tuple, Union, cast

import numpy as np
import pandas as pd
from pyarrow import feather
from scipy.spatial.transform import Rotation as R
from upath import UPath

from av2.utils.typing import NDArrayFloat

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
    "vx_m",
    "vy_m",
    "vz_m",
    "category",
)
LIDAR_GLOB_PATTERN: Final[str] = "*/sensors/lidar/*"
MAX_STR_LEN: Final[int] = 32

PathType = Union[Path, UPath]

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

        dataset_dir: PathType
        split: str
        ordered_annotations_cols: Tuple[str, ...] = DEFAULT_ANNOTATIONS_COLS
        annotations_pose_mode: str = "YAW"  # TODO: Add pose modes.
        flush_file_index: bool = False

        file_index: List[Tuple[str, int]] = field(init=False)

        def __post_init__(self) -> None:
            """Build the file index."""
            self._build_file_index()

        @property
        def file_index_path(self) -> PathType:
            """Return the file index path."""
            return Path.home() / ".cache" / "av2" / "torch" / "file_index.feather"

        @property
        def split_dir(self) -> PathType:
            """Sensor dataset split directory."""
            return self.dataset_dir / self.split

        def annotations_path(self, log_id: str) -> PathType:
            """Return the annotations at the specified log id.

            Args:
                log_id: Unique log identifier.

            Returns:
                Annotations path for the entire log.
            """
            return self.split_dir / log_id / "annotations.feather"

        def lidar_path(self, log_id: str, timestamp_ns: int) -> PathType:
            """Return the lidar path at the specified log id and timestamp.

            Args:
                log_id: Unique log identifier.
                timestamp_ns: Lidar timestamp in nanoseconds.

            Returns:
                Lidar path at the log id and timestamp.
            """
            return self.split_dir / log_id / "sensors" / "lidar" / f"{timestamp_ns}.feather"

        def pose_path(self, log_id: str) -> PathType:
            """Return the path to the city egopose feather file."""
            return self.split_dir / log_id / "city_SE3_egovehicle.feather"

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
            current_log_id, current_timestamp_ns = self.key(index)
            pose_path = self.pose_path(current_log_id)
            timestamp_ns_to_city_SE3_ego: Dict[int, Any] = (
                _read_feather(pose_path).set_index("timestamp_ns").to_dict("index")
            )

            velocity_metadata = []
            for current_track_uuid, annotation_track in annotations.groupby(by=["track_uuid"], sort=True):
                timestamp_ns_to_annotation: Dict[int, Any] = (
                    annotation_track.sort_values("timestamp_ns").set_index("timestamp_ns").to_dict("index")
                )
                index_to_key = dict(enumerate(timestamp_ns_to_annotation.keys()))
                for i, (current_timestamp_ns, _) in enumerate(timestamp_ns_to_annotation.items()):
                    previous_timestamp_ns = index_to_key.get(i - 1, None)
                    next_timestamp_ns = index_to_key.get(i + 1, None)

                    xyz_current_city = self._compute_city_annotation_coordinates(
                        timestamp_ns_to_annotation, timestamp_ns_to_city_SE3_ego, current_timestamp_ns
                    )

                    est_velocity_list: List[NDArrayFloat] = []
                    if previous_timestamp_ns is not None:
                        xyz_previous_city = self._compute_city_annotation_coordinates(
                            timestamp_ns_to_annotation, timestamp_ns_to_city_SE3_ego, previous_timestamp_ns
                        )

                        timedelta_ns_0 = current_timestamp_ns - previous_timestamp_ns
                        xyz_delta_0 = xyz_current_city - xyz_previous_city
                        est_velocity_list += [xyz_delta_0 / (timedelta_ns_0 * 1e-9)]
                    if next_timestamp_ns is not None:
                        xyz_next_city = self._compute_city_annotation_coordinates(
                            timestamp_ns_to_annotation, timestamp_ns_to_city_SE3_ego, next_timestamp_ns
                        )
                        timedelta_ns_1 = next_timestamp_ns - current_timestamp_ns
                        xyz_delta_1 = xyz_next_city - xyz_current_city
                        est_velocity_list += [xyz_delta_1 / (timedelta_ns_1 * 1e-9)]

                    if len(est_velocity_list) == 0:
                        vx_m, vy_m, vz_m = 0.0, 0.0, 0.0
                    else:
                        vx_m, vy_m, vz_m = np.mean(est_velocity_list, axis=0).tolist()
                    row = (current_track_uuid, current_timestamp_ns, vx_m, vy_m, vz_m)
                    velocity_metadata.append(row)

            velocity_frame = pd.DataFrame(
                velocity_metadata, columns=["track_uuid", "timestamp_ns", "vx_m", "vy_m", "vz_m"]
            )
            annotations = annotations.merge(on=["track_uuid", "timestamp_ns"], right=velocity_frame)
            return annotations

        def _compute_city_annotation_coordinates(
            self,
            annotations: Dict[int, Any],
            timestamp_ns_to_city_SE3_ego: Dict[int, Dict[str, float]],
            timestamp_ns: int,
        ) -> NDArrayFloat:
            annotation = annotations[timestamp_ns]
            city_R_ego, city_t_ego = self._construct_pose(timestamp_ns_to_city_SE3_ego[timestamp_ns])
            xyz_ego = np.array([annotation["tx_m"], annotation["ty_m"], annotation["tz_m"]])
            xyz_city = R.from_quat(city_R_ego).apply(xyz_ego) + city_t_ego
            return xyz_city

        def _construct_pose(self, city_SE3_ego: Dict[str, float]) -> Tuple[NDArrayFloat, NDArrayFloat]:
            city_R_ego: NDArrayFloat = np.array(
                [city_SE3_ego["qx"], city_SE3_ego["qy"], city_SE3_ego["qz"], city_SE3_ego["qw"]]
            )
            city_t_ego: NDArrayFloat = np.array([city_SE3_ego["tx_m"], city_SE3_ego["ty_m"], city_SE3_ego["tz_m"]])
            return city_R_ego, city_t_ego

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

    def _read_feather(path: PathType) -> pd.DataFrame:
        """Read an Apache feather file.

        Args:
            path: Path to the feather file.

        Returns:
            Pandas dataframe containing the arrow data.
        """
        with path.open("rb") as file_handle:
            dataframe: pd.DataFrame = feather.read_feather(file_handle)
        return dataframe

except ImportError as e:
    print("Please install Pytorch to use this dataloader.")
