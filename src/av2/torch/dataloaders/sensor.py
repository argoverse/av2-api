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
import pandas as pd
from filelock import FileLock
from torch.utils.data import Dataset
from upath import UPath

from av2.geometry.geometry import quat_to_mat
from av2.utils.io import read_feather
from av2.utils.typing import PathType

from .utils import QUAT_WXYZ_FIELDS, Annotations, Lidar, Sweep, prevent_fsspec_deadlock, query_pose, velocity_kernel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

XYZ_FIELDS: Final = ("x", "y", "z")
LIDAR_GLOB_PATTERN: Final = "sensors/lidar/*.feather"


@unique
class FileCachingMode(str, Enum):
    """File caching mode."""

    DISK = "DISK"


@dataclass
class Av2(Dataset[Sweep]):
    """Pytorch dataloader for the sensor dataset.

    Args:
        root_dir: Path to the dataset directory.
        dataset_name: Dataset name.
        split_name: Name of the dataset split.
        min_annotation_range: Min Euclidean distance between the egovehicle origin and the annotation cuboid centers.
        max_annotation_range: Max Euclidean distance between the egovehicle origin and the annotation cuboid centers.
        min_lidar_range: Min Euclidean distance between the egovehicle origin and the lidar points.
        max_lidar_range: Max Euclidean distance between the egovehicle origin and the lidar points.
        min_interior_pts: Min number of points inside each annotation.
        num_accumulated_sweeps: Number of temporally accumulated sweeps (accounting for egovehicle motion).
        file_caching_mode: File caching mode.
        return_annotations: Boolean flag indicating whether to return annotations.
        return_velocity_estimates: Boolean flag indicating whether to return annotations' velocity estimates.
    """

    root_dir: PathType
    dataset_name: str
    split_name: str
    min_annotation_range: float = 0.0
    max_annotation_range: float = inf
    min_lidar_range: float = 0.0
    max_lidar_range: float = inf
    min_interior_pts: int = 0
    num_accumulated_sweeps: int = 1
    file_caching_mode: Optional[FileCachingMode] = None
    return_annotations: bool = False
    return_velocity_estimates: bool = False

    file_index: List[Tuple[str, int]] = field(init=False)

    def __post_init__(self) -> None:
        """Build the file index."""
        if self.return_annotations and self.dataset_name == "lidar":
            raise RuntimeError("The lidar dataset does not have annotations!")
        if not self.return_annotations and self.return_velocity_estimates:
            raise RuntimeError("with_annotations must be enabled to return annotations' velocities.")
        prevent_fsspec_deadlock()
        self._build_file_index()
        self._log_dataloader_configuration()

    def __repr__(self) -> str:
        """Dataloader info."""
        info = "Dataloader configuration settings:\n"
        for key, value in sorted(self.items()):
            if key == "file_index":
                continue
            info += f"\t{key}: {value}\n"
        return info

    def items(self) -> ItemsView[str, Any]:
        """Return the attribute_name, attribute pairs for the dataloader."""
        return self.__dict__.items()

    @property
    def file_caching_dir(self) -> PathType:
        """File caching directory."""
        return Path("/") / "tmp" / "cache" / "av2" / self.dataset_name / self.split_name

    @property
    def split_dir(self) -> PathType:
        """Sensor dataset split directory."""
        return UPath(self.root_dir) / self.dataset_name / self.split_name

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
        annotations: Optional[Annotations] = None
        if self.return_annotations:
            annotations = self.read_annotations(index)

        lidar = self.read_lidar(index)
        log_id, timestamp_ns = self.sweep_uuid(index)
        return Sweep(annotations=annotations, lidar=lidar, log_id=log_id, timestamp_ns=timestamp_ns)

    def _build_file_index(self) -> None:
        """Build the file index for the dataset."""
        file_cache_path = self.file_caching_dir.parent / f"file_index_{self.split_name}.feather"
        if file_cache_path.exists():
            file_index = read_feather(file_cache_path).to_numpy().tolist()
        else:
            logger.info("Building file index. This may take a moment ...")
            log_dirs = sorted(self.split_dir.glob("*"))
            path_lists: Optional[List[List[Tuple[str, int]]]] = joblib.Parallel(n_jobs=-1, backend="multiprocessing")(
                joblib.delayed(Av2._file_index_helper)(log_dir, LIDAR_GLOB_PATTERN) for log_dir in log_dirs
            )
            logger.info("File indexing complete.")
            if path_lists is None:
                raise RuntimeError("Error scanning the dataset directory!")
            if len(path_lists) == 0:
                raise RuntimeError("No file paths found. Please validate `self.dataset_dir` and `self.split_name`.")

            file_index = sorted(itertools.chain.from_iterable(path_lists))
            self.file_caching_dir.parent.mkdir(parents=True, exist_ok=True)
            dataframe = pd.DataFrame(file_index, columns=["log_id", "timestamp_ns"])
            dataframe.to_feather(file_cache_path, compression="uncompressed")
        self.file_index = file_index

    def read_annotations(self, index: int) -> Annotations:
        """Read the sweep annotations.

        Args:
            index: Dataset index.

        Returns:
            The annotations object.
        """
        log_id, timestamp_ns = self.sweep_uuid(index)
        cache_path = self.file_caching_dir / log_id / "annotations.feather"
        annotations_path = self.annotations_path(log_id)

        dataframe = self._read_frame(annotations_path, cache_path)
        if self.return_velocity_estimates:
            dataframe = self._populate_annotations_velocity(index, dataframe)

        distances = np.linalg.norm(dataframe.loc[:, ["tx_m", "ty_m", "tz_m"]].to_numpy(), axis=-1)
        num_interior_pts = dataframe.loc[:, "num_interior_pts"].to_numpy().astype(int)
        timestamps = dataframe.loc[:, "timestamp_ns"].to_numpy().astype(int)
        mask = (
            (num_interior_pts >= self.min_interior_pts)
            & (timestamps == timestamp_ns)
            & (distances >= self.min_annotation_range)
            & (distances <= self.max_annotation_range)
        )
        dataframe = dataframe.loc[mask].reset_index(drop=True)
        annotations = Annotations(dataframe)
        return annotations

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

        file_caching_path = self.file_caching_dir / current_log_id / "city_SE3_egovehicle.feather"
        city_SE3_ego = self._read_frame(pose_path, file_caching_path)

        annotations = annotations.sort_values(["track_uuid", "timestamp_ns"])
        annotations_with_poses = annotations.merge(
            city_SE3_ego, on=["timestamp_ns"], suffixes=("_obj", None), how="left"
        )
        quat = annotations_with_poses.loc[:, list(QUAT_WXYZ_FIELDS)].to_numpy().astype(np.float64)
        mats = quat_to_mat(quat)
        translation = annotations_with_poses.loc[:, ["tx_m", "ty_m", "tz_m"]].to_numpy().astype(np.float64)
        t_xyz = annotations_with_poses.loc[:, ["tx_m_obj", "ty_m_obj", "tz_m_obj"]].to_numpy().astype(np.float64)
        annotations.loc[:, ["tx_m_city", "ty_m_city", "tz_m_city"]] = (
            t_xyz[:, None] @ mats.transpose(0, 2, 1) + translation[:, None]
        ).squeeze()

        velocities = (
            annotations.loc[:, ["track_uuid", "timestamp_ns", "tx_m_city", "ty_m_city", "tz_m_city"]]
            .groupby(["track_uuid"])
            .diff()
            .rolling(window=3, min_periods=1, closed="right", center=True, method="table")
            .apply(velocity_kernel, raw=True, engine="numba")
            .to_numpy()
            .astype(np.float64)
        )
        annotations[["vx", "vy", "vz"]] = velocities[:, 1:]
        annotations = annotations.drop(["tx_m_city", "ty_m_city", "tz_m_city"], axis=1)
        return annotations.reset_index(drop=True)

    def read_lidar(self, index: int) -> Lidar:
        """Read the lidar sweep.

        Args:
            index: Dataset index.

        Returns:
            Tensor of annotations.
        """
        log_id, timestamp_ns_j = self.sweep_uuid(index)
        temporal_window = self.file_index[max(index - self.num_accumulated_sweeps + 1, 0) : index + 1][::-1]
        filtered_window: List[Tuple[str, int]] = list(
            filter(lambda sweep_uuid: sweep_uuid[0] == log_id, temporal_window)
        )

        poses = self._read_frame(
            src_path=self.pose_path(log_id),
            file_caching_path=self.file_caching_dir / log_id / "city_SE3_egovehicle.feather",
        )
        ego_tj_SE3_city = query_pose(poses, timestamp_ns_j).inverse()
        dataframe_list: List[pd.DataFrame] = []
        for _, (log_id, timestamp_ns_i) in enumerate(filtered_window):
            dataframe = self._read_frame(
                src_path=self.lidar_path(log_id, timestamp_ns_i),
                file_caching_path=self.file_caching_dir / log_id / "sensors" / "lidar" / f"{timestamp_ns_i}.feather",
            )
            xyz_ti = dataframe.loc[:, list(XYZ_FIELDS)].to_numpy().astype(np.float64)
            xyz_tj = xyz_ti
            dt = timestamp_ns_j - timestamp_ns_i
            assert dt >= 0

            # Timestamps do not match, we're in a new reference frame.
            if dt != 0:
                city_SE3_ego_ti = query_pose(poses, timestamp_ns_i)
                ego_tj_SE3_ego_ti = ego_tj_SE3_city.compose(city_SE3_ego_ti)
                xyz_tj = ego_tj_SE3_ego_ti.transform_point_cloud(xyz_ti)

            dataframe[list(XYZ_FIELDS)] = pd.DataFrame(xyz_tj.astype(np.float32), columns=list(XYZ_FIELDS))
            dataframe["timedelta_ns"] = np.full(len(xyz_tj), fill_value=dt)
            dataframe_list.append(dataframe)
        dataframe = pd.concat(dataframe_list).reset_index(drop=True)
        dataframe = self._post_process_lidar(dataframe)
        return Lidar(dataframe)

    def _post_process_lidar(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Apply post-processing operations on the point cloud.

        Args:
            dataframe: Lidar dataframe.

        Returns:
            The filtered lidar dataframe.
        """
        distances = np.linalg.norm(dataframe.loc[:, list(XYZ_FIELDS)].to_numpy(), axis=1)
        dataframe["distance"] = distances
        mask = (distances >= self.min_lidar_range) & (distances <= self.max_lidar_range)
        dataframe = dataframe.loc[mask]
        dataframe = dataframe.sort_values(["timedelta_ns", "distance"])
        return dataframe

    @staticmethod
    def _file_index_helper(root_dir: PathType, file_pattern: str) -> List[Tuple[str, int]]:
        """Build the file index in a multiprocessing context.

        Args:
            root_dir: Root directory.
            file_pattern: File pattern string.

        Returns:
            The list of keys within the glob context.
        """
        prevent_fsspec_deadlock()
        return [(key.parts[-4], int(key.stem)) for key in root_dir.glob(file_pattern)]

    def _read_frame(self, src_path: PathType, file_caching_path: PathType) -> pd.DataFrame:
        """Read a dataframe from a remote source or a locally cached location.

        Args:
            src_path: Path to the non-cached file.
            file_caching_path: Path to the cached file.

        Returns:
            DataFrame representation of the feather file.
        """
        if self.file_caching_mode == FileCachingMode.DISK:
            file_caching_path.parent.mkdir(parents=True, exist_ok=True)
            lock_name = str(file_caching_path) + ".lock"
            lock = FileLock(lock_name)
            with lock:
                if not file_caching_path.exists():
                    dataframe = read_feather(src_path)
                    dataframe.to_feather(file_caching_path, compression="uncompressed")
                else:
                    try:
                        dataframe = read_feather(file_caching_path)
                    except Exception as _:
                        dataframe = read_feather(src_path)
                        dataframe.to_feather(file_caching_path, compression="uncompressed")
        else:
            dataframe = read_feather(src_path)
        return dataframe
