"""Pytorch dataloader for the Argoverse 2 dataset."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from math import inf
from os import PathLike
from pathlib import Path
from typing import Any, Dict, Final, ItemsView, List, Tuple, cast

import numpy as np
import pandas as pd
import polars as pl
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset

from av2.utils.io import read_feather
from av2.utils.typing import NDArrayFloat, NDArrayFloat32, PathType

from .utils import LIDAR_GLOB_PATTERN, Annotations, Lidar, Sweep, prevent_fsspec_deadlock, query_SE3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)

ANNOTATION_UUID_FIELDS: Final[Tuple[str, str]] = ("track_uuid", "timestamp_ns")
SWEEP_UUID_FIELDS: Final[Tuple[str, str]] = ("log_id", "timestamp_ns")
POINT_COORDINATE_FIELDS: Final[Tuple[str, str, str]] = ("x", "y", "z")


@dataclass
class Av2(Dataset[Sweep]):  # type: ignore
    """Pytorch dataloader for the sensor dataset."""

    dataset_dir: PathType
    split: str
    flush_file_index: bool = False

    file_index: List[Tuple[str, int]] = field(init=False)
    max_annotation_range: float = inf
    max_lidar_range: float = inf
    num_accumulated_sweeps: int = 1

    def __post_init__(self) -> None:
        """Build the file index."""
        prevent_fsspec_deadlock()
        self._build_file_index()
        self._log_dataloader_configuration()

    def items(self) -> ItemsView[str, Any]:
        """Return the attribute_name, attribute pairs for the dataloader."""
        return self.__dict__.items()

    @property
    def file_index_path(self) -> PathType:
        """File index path."""
        return Path.home() / ".cache" / "av2" / "torch" / "file_index.feather"

    @property
    def split_dir(self) -> PathType:
        """Sensor dataset split directory."""
        return self.dataset_dir / self.split

    def _log_dataloader_configuration(self) -> None:
        """Log the dataloader configuration."""
        info = "Dataloader has been configured. Here are the settings:\n"
        for key, value in self.items():
            if key == "file_index":
                continue
            info += f"\t{key}: {value}\n"
        logger.info("%s", info)

    def annotations_path(self, log_id: str) -> PathType:
        """Get the annotations at the specified log id.

        Args:
            log_id: Unique log identifier.

        Returns:
            Annotations path for the entire log.
        """
        return self.split_dir / log_id / "annotations.feather"

    def lidar_path(self, log_id: str, timestamp_ns: int) -> PathType:
        """Get the lidar path at the specified log id and timestamp.

        Args:
            log_id: Unique log identifier.
            timestamp_ns: Lidar timestamp in nanoseconds.

        Returns:
            Lidar path at the log id and timestamp.
        """
        return self.split_dir / log_id / "sensors" / "lidar" / f"{timestamp_ns}.feather"

    def pose_path(self, log_id: str) -> PathType:
        """Get the city egopose path."""
        return self.split_dir / log_id / "city_SE3_egovehicle.feather"

    def sweep_uuid(self, index: int) -> Tuple[str, int]:
        """Get the sweep uuid at the given index.

        Args:
            index: Dataset index.

        Returns:
            The sweep uuid (log_id, timestamp_ns).
        """
        return self.file_index[index]

    def __getitem__(self, index: int) -> Sweep:
        """Get the annotations and lidar for one sweep.

        Args:
            index: Dataset index.

        Returns:
            Sweep object containing annotations and lidar.
        """
        annotations = self.read_annotations(index)
        lidar = self.read_lidar(index)
        return Sweep(annotations=annotations, lidar=lidar)

    def _build_file_index(self) -> None:
        """Build the file index consisting of (log_id, timestamp_ns) pairs."""
        if not self.flush_file_index and self.file_index_path.exists():
            logger.info("Using cached file index ...")
            dataframe = read_feather(self.file_index_path)
            self.file_index = [(cast(str, key[0]), cast(int, key[1])) for key in dataframe.to_numpy().tolist()]
        else:
            logger.info("Building file index. This may take a moment ...")
            paths = sorted(self.split_dir.glob(LIDAR_GLOB_PATTERN))
            self.file_index = [(key.parts[-4], int(key.stem)) for key in paths]

            self.file_index_path.parent.mkdir(parents=True, exist_ok=True)
            dataframe = pd.DataFrame.from_records(self.file_index, columns=SWEEP_UUID_FIELDS)
            dataframe.to_feather(self.file_index_path)

    def read_annotations(self, index: int) -> Annotations:
        """Read the sweep annotations.

        Args:
            index: Dataset index.

        Returns:
            The annotations object.
        """
        log_id, timestamp_ns = self.sweep_uuid(index)
        annotations_path = self.annotations_path(log_id)
        annotations = self._read_feather(annotations_path)
        annotations = self._populate_annotations_velocity(index, annotations.to_pandas())
        query = (pl.col("num_interior_pts") > 0) & (pl.col("timestamp_ns") == timestamp_ns)
        annotations = pl.from_pandas(annotations).filter(query)
        return Annotations.from_dataframe(annotations)

    def _populate_annotations_velocity(self, index: int, annotations: pd.DataFrame) -> pd.DataFrame:
        """Populate the annotations with their estimated velocities.

        Args:
            index: Dataset index.
            annotations: DataFrame of annotations loaded from a feather file.

        Returns:
            The dataFrame populated with velocities.
        """
        current_log_id, current_timestamp_ns = self.sweep_uuid(index)
        pose_path = self.pose_path(current_log_id)
        timestamp_ns_to_city_SE3_ego = cast(
            Dict[int, Any], read_feather(pose_path).set_index("timestamp_ns").to_dict("index")
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

        columns = ANNOTATION_UUID_FIELDS + ("vx_m", "vy_m", "vz_m")
        velocity_frame = pd.DataFrame(velocity_metadata, columns=list(columns))
        annotations = annotations.merge(on=ANNOTATION_UUID_FIELDS, right=velocity_frame)
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
        xyz_city: NDArrayFloat = R.from_quat(city_R_ego).apply(xyz_ego) + city_t_ego
        return xyz_city

    def _construct_pose(self, city_SE3_ego: Dict[str, float]) -> Tuple[NDArrayFloat, NDArrayFloat]:
        city_R_ego: NDArrayFloat = np.array(
            [city_SE3_ego["qx"], city_SE3_ego["qy"], city_SE3_ego["qz"], city_SE3_ego["qw"]]
        )
        city_t_ego: NDArrayFloat = np.array([city_SE3_ego["tx_m"], city_SE3_ego["ty_m"], city_SE3_ego["tz_m"]])
        return city_R_ego, city_t_ego

    def read_lidar(self, index: int) -> Lidar:
        """Read the lidar sweep.

        Args:
            index: Dataset index.

        Returns:
            Tensor of annotations.
        """
        log_id, timestamp_ns = self.sweep_uuid(index)
        window = self.file_index[max(index - self.num_accumulated_sweeps + 1, 0) : index]
        filtered_window: List[Tuple[str, int]] = list(filter(lambda sweep_uuid: sweep_uuid[0] == log_id, window))

        lidar_path = self.lidar_path(log_id, timestamp_ns)
        dataframe_list = [self._read_feather(lidar_path)]
        if len(window) > 0:
            poses = self._read_feather(self.pose_path(log_id))
            ego_current_SE3_city = query_SE3(poses, timestamp_ns).inverse()
            for log_id, timestamp_ns in filtered_window:
                city_SE3_ego_past = query_SE3(poses, timestamp_ns)
                ego_current_SE3_ego_past = ego_current_SE3_city.compose(city_SE3_ego_past)

                dataframe = self._read_feather(self.lidar_path(log_id, timestamp_ns))
                point_cloud: NDArrayFloat = dataframe[list(POINT_COORDINATE_FIELDS)].to_numpy().astype(np.float64)
                dataframe[list(POINT_COORDINATE_FIELDS)] = ego_current_SE3_ego_past.transform_point_cloud(
                    point_cloud
                ).astype(np.float32)
                dataframe_list.append(dataframe)
        dataframe = pl.concat(dataframe_list)
        dataframe = self._post_process_lidar(dataframe)
        return Lidar.from_dataframe(dataframe)

    def _post_process_lidar(self, dataframe: pl.DataFrame) -> pl.DataFrame:
        """Apply post-processing operations on the point cloud.

        Args:
            dataframe: Lidar dataframe.

        Returns:
            The filtered lidar dataframe.
        """
        query = pl.col(["x"]).pow(2) + pl.col(["y"]).pow(2) + pl.col(["z"]).pow(2) <= self.max_lidar_range**2
        return dataframe.filter(query)

    def _read_feather(self, path: PathType) -> pl.DataFrame:
        with path.open("rb") as file_handle:
            return pl.read_ipc(file_handle)
