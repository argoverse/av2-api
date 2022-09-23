"""Pytorch dataloader for the Argoverse 2 dataset."""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from enum import Enum, unique
from math import inf
from pathlib import Path
from typing import Any, Final, ItemsView, List, Optional, Tuple

import joblib
import numpy as np
import polars as pl
from torch.utils.data import Dataset

from av2.geometry.geometry import quat_to_mat
from av2.utils.typing import NDArrayFloat, PathType

from .utils import QUAT_WXYZ_FIELDS, Annotations, Lidar, Sweep, prevent_fsspec_deadlock, query_SE3, read_feather

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)

ANNOTATION_UUID_FIELDS: Final[Tuple[str, str]] = ("track_uuid", "timestamp_ns")
SWEEP_UUID_FIELDS: Final[Tuple[str, str]] = ("log_id", "timestamp_ns")
POINT_COORDINATE_FIELDS: Final[Tuple[str, str, str]] = ("x", "y", "z")

LIDAR_GLOB_PATTERN: Final[str] = "sensors/lidar/*.feather"


pl.Config.with_columns_kwargs = True


@unique
class FileCachingMode(str, Enum):
    """File caching mode."""

    DISK = "DISK"


@dataclass
class Av2(Dataset[Sweep]):  # type: ignore
    """Pytorch dataloader for the sensor dataset.

    Args:
        dataset_dir: Path to the dataset directory.
        split_name: Name of the dataset split.
        max_annotation_range: Max Euclidean distance between the egovehicle origin and the annotation cuboid centers.
        max_lidar_range: Max Euclidean distance between the egovehicle origin and the lidar points.
        num_accumulated_sweeps: Number of temporally accumulated sweeps (accounting for egovehicle motion).
        file_caching_mode: File caching mode.
    """

    dataset_dir: PathType
    split_name: str
    max_annotation_range: float = inf
    max_lidar_range: float = inf
    num_accumulated_sweeps: int = 1
    file_caching_mode: Optional[FileCachingMode] = None

    file_index: List[Tuple[str, int]] = field(init=False)

    def __post_init__(self) -> None:
        """Build the file index."""
        prevent_fsspec_deadlock()
        self._build_file_index()
        self._log_dataloader_configuration()

    def items(self) -> ItemsView[str, Any]:
        """Return the attribute_name, attribute pairs for the dataloader."""
        return self.__dict__.items()

    @property
    def cache_dir(self) -> PathType:
        return Path.home() / ".cache" / "av2" / f"{self.num_accumulated_sweeps}_sweeps" / self.split_name

    @property
    def split_dir(self) -> PathType:
        """Sensor dataset split directory."""
        return self.dataset_dir / self.split_name

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
        return Sweep(annotations=annotations, lidar=lidar, sweep_uuid=self.sweep_uuid(index))

    def _build_file_index(self) -> None:
        """Build the file index for the dataset."""
        logger.info("Building file index. This may take a moment ...")

        log_dirs = sorted(self.split_dir.glob("*"))
        path_lists: Optional[List[List[Tuple[str, int]]]] = joblib.Parallel(n_jobs=-1)(
            joblib.delayed(Av2._file_index_helper)(log_dir, LIDAR_GLOB_PATTERN) for log_dir in log_dirs
        )
        if path_lists is None:
            raise RuntimeError("Error scanning the dataset directory!")
        elif len(path_lists) == 0:
            raise RuntimeError("No file paths found. Please validate `self.dataset_dir` and `self.split_name`.")

        self.file_index = sorted(itertools.chain.from_iterable(path_lists))

    def read_annotations(self, index: int) -> Annotations:
        """Read the sweep annotations.

        Args:
            index: Dataset index.

        Returns:
            The annotations object.
        """
        log_id, timestamp_ns = self.sweep_uuid(index)
        cache_path = self.cache_dir / log_id / "annotations.feather"

        if self.file_caching_mode == FileCachingMode.DISK and cache_path.exists():
            dataframe = read_feather(cache_path)
        else:
            annotations_path = self.annotations_path(log_id)
            dataframe = read_feather(annotations_path)
            dataframe = self._populate_annotations_velocity(index, dataframe)

        if self.file_caching_mode == FileCachingMode.DISK and not cache_path.exists():
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            dataframe.write_ipc(cache_path)

        dataframe = dataframe.filter((pl.col("num_interior_pts") > 0) & (pl.col("timestamp_ns") == timestamp_ns))
        annotations = Annotations(dataframe)
        return annotations

    def _populate_annotations_velocity(self, index: int, annotations: pl.DataFrame) -> pl.DataFrame:
        """Populate the annotations with their estimated velocities.

        Args:
            index: Dataset index.
            annotations: DataFrame of annotations loaded from a feather file.

        Returns:
            The dataFrame populated with velocities.
        """
        current_log_id, _ = self.sweep_uuid(index)
        pose_path = self.pose_path(current_log_id)
        city_SE3_ego = read_feather(pose_path)

        annotations = annotations.sort(["track_uuid", "timestamp_ns"]).with_row_count()
        annotations = annotations.with_columns(row_nr=pl.col("row_nr").cast(pl.Int64))

        annotations_with_poses = annotations.select(
            [pl.col("timestamp_ns"), pl.col(["tx_m", "ty_m", "tz_m"]).map_alias(lambda x: f"{x}_obj")]
        ).join(city_SE3_ego, on="timestamp_ns")
        mats = quat_to_mat(annotations_with_poses.select(pl.col(list(QUAT_WXYZ_FIELDS))).to_numpy())
        translation = annotations_with_poses.select(pl.col(["tx_m", "ty_m", "tz_m"])).to_numpy()

        t_xyz = annotations_with_poses.select(pl.col(["tx_m_obj", "ty_m_obj", "tz_m_obj"])).to_numpy()
        t_xyz_city = pl.from_numpy(
            (t_xyz[:, None] @ mats.transpose(0, 2, 1) + translation[:, None]).squeeze(),
            ["tx_m_city", "ty_m_city", "tz_m_city"],
        )

        annotations_city = pl.concat(
            [annotations.select(pl.col(["row_nr", "timestamp_ns", "track_uuid"])), t_xyz_city],
            how="horizontal",
        )

        velocities = annotations_city.groupby_rolling(
            index_column="row_nr", period="3i", offset="-2i", by=["track_uuid"], closed="right"
        ).agg(
            [
                (pl.col("tx_m_city").diff() / (pl.col("timestamp_ns").diff() * 1e-9)).mean().alias("vx_m"),
                (pl.col("ty_m_city").diff() / (pl.col("timestamp_ns").diff() * 1e-9)).mean().alias("vy_m"),
                (pl.col("tz_m_city").diff() / (pl.col("timestamp_ns").diff() * 1e-9)).mean().alias("vz_m"),
            ]
        )
        annotations = annotations.join(velocities, on=["track_uuid", "row_nr"])
        return annotations

    def read_lidar(self, index: int) -> Lidar:
        """Read the lidar sweep.

        Args:
            index: Dataset index.

        Returns:
            Tensor of annotations.
        """
        log_id, timestamp_ns = self.sweep_uuid(index)
        cache_path = self.cache_dir / self.split_name / log_id / "sensors" / "lidar" / f"{timestamp_ns}.feather"

        if self.file_caching_mode == FileCachingMode.DISK and cache_path.exists():
            dataframe = read_feather(cache_path)
        else:
            window = self.file_index[max(index - self.num_accumulated_sweeps + 1, 0) : index]
            filtered_window: List[Tuple[str, int]] = list(filter(lambda sweep_uuid: sweep_uuid[0] == log_id, window))

            lidar_path = self.lidar_path(log_id, timestamp_ns)
            dataframe_list = [read_feather(lidar_path)]
            if len(window) > 0:
                poses = read_feather(self.pose_path(log_id))
                ego_current_SE3_city = query_SE3(poses, timestamp_ns).inverse()
                for log_id, timestamp_ns in filtered_window:
                    city_SE3_ego_past = query_SE3(poses, timestamp_ns)
                    ego_current_SE3_ego_past = ego_current_SE3_city.compose(city_SE3_ego_past)

                    dataframe = read_feather(self.lidar_path(log_id, timestamp_ns))
                    point_cloud: NDArrayFloat = (
                        dataframe.select(pl.col(POINT_COORDINATE_FIELDS)).to_numpy().astype(np.float64)
                    )
                    points_ego_current = ego_current_SE3_ego_past.transform_point_cloud(point_cloud).astype(np.float32)
                    dataframe = pl.concat(
                        [
                            pl.from_numpy(points_ego_current, POINT_COORDINATE_FIELDS),
                            dataframe.select(pl.col("*").exclude(POINT_COORDINATE_FIELDS)),
                        ],
                        how="horizontal",
                    )
                    dataframe_list.append(dataframe)
            dataframe = pl.concat(dataframe_list)
        if self.file_caching_mode == FileCachingMode.DISK and not cache_path.exists():
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            dataframe.write_ipc(cache_path)

        dataframe = self._post_process_lidar(dataframe)
        lidar = Lidar(dataframe)
        return lidar

    def _post_process_lidar(self, dataframe: pl.DataFrame) -> pl.DataFrame:
        """Apply post-processing operations on the point cloud.

        Args:
            dataframe: Lidar dataframe.

        Returns:
            The filtered lidar dataframe.
        """
        return dataframe.filter(
            pl.col(["x"]).pow(2) + pl.col(["y"]).pow(2) + pl.col(["z"]).pow(2) <= self.max_lidar_range**2
        )

    @staticmethod
    def _file_index_helper(root_dir: PathType, file_pattern: str) -> List[Tuple[str, int]]:
        """Build the file index in a multiprocessing context.

        Args:
            root_dir: Root directory.
            file_pattern: File pattern string.

        Returns:
            The list of keys within the glob context.
        """
        return [(key.parts[-4], int(key.stem)) for key in root_dir.glob(file_pattern)]
