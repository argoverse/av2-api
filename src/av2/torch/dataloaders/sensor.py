"""Pytorch dataloader for the Argoverse 2 dataset."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from math import inf
from pathlib import Path
from typing import Any, Dict, Final, ItemsView, List, Tuple, cast

import numpy as np
import pandas as pd
import polars as pl
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset

from av2.geometry.geometry import quat_to_mat
from av2.utils.io import read_feather
from av2.utils.typing import NDArrayFloat, PathType

from .utils import LIDAR_GLOB_PATTERN, Annotations, Lidar, Sweep, prevent_fsspec_deadlock, query_SE3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)

ANNOTATION_UUID_FIELDS: Final[Tuple[str, str]] = ("track_uuid", "timestamp_ns")
SWEEP_UUID_FIELDS: Final[Tuple[str, str]] = ("log_id", "timestamp_ns")
POINT_COORDINATE_FIELDS: Final[Tuple[str, str, str]] = ("x", "y", "z")

pl.Config.with_columns_kwargs = True


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
        current_log_id, _ = self.sweep_uuid(index)
        pose_path = self.pose_path(current_log_id)
        city_SE3_ego = self._read_feather(pose_path)

        annotations: pl.DataFrame = pl.from_pandas(annotations)
        annotations = annotations.sort(["track_uuid", "timestamp_ns"]).select(
            [pl.arange(0, len(annotations)).alias("index"), pl.col("*")]
        )

        annotations_with_poses = annotations.join(city_SE3_ego, on="timestamp_ns")
        mats = quat_to_mat(annotations_with_poses.select(pl.col(["qw", "qx", "qy", "qz"])).to_numpy())
        translation = annotations_with_poses.select(pl.col(["tx_m_right", "ty_m_right", "tz_m_right"])).to_numpy()

        xyz = annotations_with_poses.select(pl.col(["tx_m", "ty_m", "tz_m"])).to_numpy()
        xyz_city = mats @ xyz[:, :, None] + translation[:, :, None]

        annotations = pl.concat(
            (annotations.drop(["tx_m", "ty_m", "tz_m"]), pl.from_numpy(xyz_city.squeeze(), ["tx_m", "ty_m", "tz_m"])),
            how="horizontal",
        )

        velocities = annotations.groupby_rolling(
            index_column="index", period="3i", offset="-2i", by=["track_uuid"], closed="right"
        ).agg(
            [
                (pl.col("tx_m").diff().mean() / (pl.col("timestamp_ns").diff().mean() * 1e-9)).first().alias("vx_m"),
                (pl.col("ty_m").diff().mean() / (pl.col("timestamp_ns").diff().mean() * 1e-9)).first().alias("vy_m"),
                (pl.col("tz_m").diff().mean() / (pl.col("timestamp_ns").diff().mean() * 1e-9)).first().alias("vz_m"),
            ]
        )
        annotations = annotations.join(velocities, on=["track_uuid", "index"]).drop("index")
        return annotations.to_pandas()

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
        return pl.from_pandas(read_feather(path))
