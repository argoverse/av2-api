# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Dataloader for the Argoverse 2 (AV2) sensor dataset."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Dict, Final, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from rich.progress import track

from av2.datasets.sensor.constants import RingCameras, StereoCameras
from av2.datasets.sensor.utils import convert_path_to_named_record
from av2.geometry.camera.pinhole_camera import PinholeCamera
from av2.map.map_api import ArgoverseStaticMap
from av2.structures.cuboid import CuboidList
from av2.structures.sweep import Sweep
from av2.structures.timestamped_image import TimestampedImage
from av2.utils.constants import HOME
from av2.utils.io import TimestampedCitySE3EgoPoses, read_city_SE3_ego, read_feather, read_img
from av2.utils.metric_time import TimeUnit, to_metric_time

logger = logging.Logger(__name__)

LIDAR_PATTERN: Final[str] = "**/sensors/lidar/*.feather"
CAMERA_PATTERN: Final[str] = "**/sensors/cameras/*/*.jpg"

Millisecond = TimeUnit.Millisecond
Nanosecond = TimeUnit.Nanosecond
Second = TimeUnit.Second

# for both ring cameras, and stereo cameras.
CAM_FPS: Final[int] = 20
LIDAR_FRAME_RATE_HZ: Final[int] = 10

# constants defined in milliseconds
# below evaluates to 50 ms
CAM_SHUTTER_INTERVAL_MS: Final[float] = to_metric_time(ts=1 / CAM_FPS, src=Second, dst=Millisecond)

# below evaluates to 100 ms
LIDAR_SWEEP_INTERVAL_MS: Final[float] = to_metric_time(ts=1 / LIDAR_FRAME_RATE_HZ, src=Second, dst=Millisecond)

ALLOWED_TIMESTAMP_BUFFER_MS: Final[int] = 2  # allow 2 ms of buffer
LIDAR_SWEEP_INTERVAL_W_BUFFER_MS: Final[float] = LIDAR_SWEEP_INTERVAL_MS + ALLOWED_TIMESTAMP_BUFFER_MS
LIDAR_SWEEP_INTERVAL_W_BUFFER_NS: Final[float] = to_metric_time(
    ts=LIDAR_SWEEP_INTERVAL_W_BUFFER_MS, src=Millisecond, dst=Nanosecond
)


@dataclass
class SynchronizedSensorData:
    """Represents information associated with a single sweep.

    Enables motion compensation between the sweep and associated images.

    Args:
        sweep: Lidar sweep.
        timestamp_city_SE3_ego_dict: Mapping from vehicle timestamp to the egovehicle's pose in the city frame.
        log_id: Unique identifier for the AV2 vehicle log.
        sweep_number: Index of the sweep in [0, N-1], of all N sweeps in the log.
        num_sweeps_in_log: Number of sweeps in the log.
        annotations: Cuboids that have been annotated within the sweep, or None.
        synchronized_imagery: Mapping from camera name to timestamped imagery, or None.
    """

    sweep: Sweep
    timestamp_city_SE3_ego_dict: TimestampedCitySE3EgoPoses
    log_id: str
    sweep_number: int
    num_sweeps_in_log: int
    timestamp_ns: int
    annotations: Optional[CuboidList] = None
    avm: Optional[ArgoverseStaticMap] = None
    synchronized_imagery: Optional[Dict[str, TimestampedImage]] = None


@dataclass
class SensorDataloader:
    """Sensor dataloader for the Argoverse 2 sensor dataset.

    NOTE: We build a cache of sensor records and synchronization information to reduce I/O overhead.

    Args:
        dataset_dir: Path to the sensor dataset directory.
        with_annotations: Flag to return annotations in the __getitem__ method.
        with_cams: Flag to load and return synchronized imagery in the __getitem__ method.
        with_cache: Flag to enable file directory caching.
        matching_criterion: either "nearest" or "forward".

    Returns:
        Argoverse 2 sensor dataloader.
    """

    dataset_dir: Path
    with_annotations: bool = True
    with_cache: bool = True
    cam_names: Tuple[Union[RingCameras, StereoCameras], ...] = tuple(RingCameras)
    matching_criterion = "nearest"

    sensor_cache: pd.DataFrame = field(init=False)

    # Initialize synchronized metadata variable.
    # This is only populated when self.use_imagery is set.
    synchronization_cache: Optional[pd.DataFrame] = None

    def __post_init__(self) -> None:
        """Index the dataset for fast sensor data lookup.

        Synchronization database and sensor records are separate tables. Sensor records are an enumeration of
        the records. The synchronization database is a hierarchichal index (Pandas MultiIndex) that functions
        as a lookup table with correspondences between nearest images.
        Given reference LiDAR timestamp -> obtain 7 closest ring camera + 2 stereo camera timestamps.
        First level: Log id (1000 uuids)
        Second level: Sensor name (lidar, ring_front_center, ring_front_left, ..., stereo_front_right).
        Third level: Nanosecond timestamp (64-bit integer corresponding to vehicle time when the data was collected).
        SENSOR RECORDS:
            log_id                               sensor_name         timestamp_ns
            0c6e62d7-bdfa-3061-8d3d-03b13aa21f68 lidar               315971436059707000
                                                 lidar               315971436159903000
                                                 lidar               315971436260099000
                                                 lidar               315971436359632000
                                                 lidar               315971436459828000
            ...                                                                     ...
            ff0dbfc5-8a7b-3a6e-8936-e5e812e45408 stereo_front_right  315972918949927214
                                                 stereo_front_right  315972918999927217
                                                 stereo_front_right  315972919049927212
                                                 stereo_front_right  315972919099927219
                                                 stereo_front_right  315972919149927218
        SYNCHRONIZATION RECORDS:
                                                                             ring_front_center       stereo_front_right
        log_id                               sensor_name timestamp_ns
        0c6e62d7-bdfa-3061-8d3d-03b13aa21f68 lidar       315971436059707000  315971436049927217  ...  315971436049927215
                                                         315971436159903000  315971436149927219  ...  315971436149927217
                                                         315971436260099000  315971436249927221  ...  315971436249927219
                                                         315971436359632000  315971436349927219  ...  315971436349927221
                                                         315971436459828000  315971436449927218  ...  315971436449927207
                                                         ...                 ...                 ...                 ...
        ff0dbfc5-8a7b-3a6e-8936-e5e812e45408 lidar       315972918660124000  315972918649927220  ...  315972918649927214
                                                         315972918759657000  315972918749927214  ...  315972918749927212
                                                         315972918859853000  315972918849927218  ...  315972918849927213
                                                         315972918960050000  315972918949927220  ...  315972918949927214
                                                         315972919060249000  315972919049927214  ...  315972919049927212
        """
        # Load split, log_id, sensor_type, and timestamp_ns information.
        self.sensor_cache = self._build_sensor_cache()

        # Populate synchronization database.
        if self.cam_names:
            synchronization_cache_path = HOME / ".cache" / "av2" / "synchronization_cache.feather"
            synchronization_cache_path.parent.mkdir(parents=True, exist_ok=True)

            # If caching is enabled AND the path exists, then load from the cache file.
            if self.with_cache and synchronization_cache_path.exists():
                self.synchronization_cache = read_feather(synchronization_cache_path)
            else:
                self.synchronization_cache = self._build_synchronization_cache()

            # If caching is enabled and we haven't created the cache, then save to disk.
            if self.with_cache and not synchronization_cache_path.exists():
                self.synchronization_cache.to_feather(str(synchronization_cache_path))

            # Finally, create a MultiIndex set the sync records index and sort it.
            self.synchronization_cache.set_index(keys=["split", "log_id", "sensor_name"], inplace=True)
            self.synchronization_cache.sort_index(inplace=True)

    @cached_property
    def num_logs(self) -> int:
        """Return the number of unique logs."""
        return len(self.sensor_cache.index.unique("log_id"))

    @cached_property
    def num_sweeps(self) -> int:
        """Return the number of unique lidar sweeps."""
        return int(self.sensor_counts["lidar"])

    @cached_property
    def sensor_counts(self) -> pd.Series:
        """Return the number of records for each sensor."""
        sensor_counts: pd.Series = self.sensor_cache.index.get_level_values("sensor_name").value_counts()
        return sensor_counts

    @property
    def num_sensors(self) -> int:
        """Return the number of sensors present throughout the dataset."""
        return len(self.sensor_counts)

    def _build_sensor_cache(self) -> pd.DataFrame:
        """Load the sensor records from the root directory.

        We glob the filesystem for all LiDAR and camera filepaths, and then convert each file path
        to a "sensor record".

        A sensor record is a 3-tuple consisting of the following:
            log_id: uuid corresponding to ~15 seconds of sensor data.
            sensor_name: the name of the sensor (e.g., 'lidar').
            timestamp_ns: vehicle nanosecond timestamp at which the sensor data was recorded.

        Returns:
            Sensor record index.
        """
        logger.info("Building metadata ...")

        # Create the cache file path.
        sensor_cache_path = HOME / ".cache" / "av2" / "sensor_cache.feather"
        sensor_cache_path.parent.mkdir(parents=True, exist_ok=True)

        if self.with_cache and sensor_cache_path.exists():
            logger.info("Cache found. Loading from disk ...")
            sensor_cache = read_feather(sensor_cache_path)
        else:
            lidar_records = self.populate_lidar_records()
            # Load camera records if enabled.
            if self.cam_names:
                logger.info("Loading camera data ...")
                cam_records = self.populate_image_records()
                # Concatenate lidar and camera records.
                sensor_cache = pd.concat([lidar_records, cam_records])
            else:
                sensor_cache = lidar_records

            # Save the metadata if caching is enable.
            if self.with_cache:
                sensor_cache.reset_index(drop=True).to_feather(sensor_cache_path)

        # Set index as tuples of the form: (split, log_id, sensor_name, timestamp_ns) and sort the index.
        # sorts by split, log_id, and then by sensor name, and then by timestamp.
        sensor_cache.set_index(["split", "log_id", "sensor_name", "timestamp_ns"], inplace=True)
        sensor_cache.sort_index(inplace=True)

        # Return all of the sensor records.
        return sensor_cache

    def populate_lidar_records(self) -> pd.DataFrame:
        """Obtain (log_id, sensor_name, timestamp_ns) 3-tuples for all LiDAR sweeps in the dataset.

        Returns:
            DataFrame of shape (N,3) with `log_id`, `sensor_name`, and `timestamp_ns` columns.
                N is the number of sweeps for all logs in the dataset, and the `sensor_name` column
                should be populated with `lidar` in every entry.
        """
        lidar_paths = sorted(self.dataset_dir.glob(LIDAR_PATTERN), key=lambda x: int(x.stem))
        lidar_record_list = [
            convert_path_to_named_record(x) for x in track(lidar_paths, description="Loading lidar records ...")
        ]

        # Concatenate into single dataframe (list-of-dicts to DataFrame).
        lidar_records = pd.DataFrame(lidar_record_list)
        return lidar_records

    def populate_image_records(self) -> pd.DataFrame:
        """Obtain (log_id, sensor_name, timestamp_ns) 3-tuples for all images in the dataset.

        Returns:
            DataFrame of shape (N,3) with `log_id`, `sensor_name`, and `timestamp_ns` columns.
                N is the total number of images for all logs in the dataset, and the `sensor_name` column
                should be populated with the name of the camera that captured the corresponding image in
                every entry.
        """
        # Get sorted list of camera paths.
        cam_paths = sorted(self.dataset_dir.glob(CAMERA_PATTERN), key=lambda x: int(x.stem))

        # Load entire set of camera records.
        cam_record_list = [
            convert_path_to_named_record(x) for x in track(cam_paths, description="Loading camera records ...")
        ]

        # Concatenate into single dataframe (list-of-dicts to DataFrame).
        cam_records = pd.DataFrame(cam_record_list)
        return cam_records

    def __len__(self) -> int:
        """Return the number of lidar sweeps in the dataset.

        The lidar sensor operates at 10 Hz. There are roughly 15 seconds of sensor data per log.

        Returns:
            Number of lidar sweeps.
        """
        return self.num_sweeps

    def __iter__(self) -> SensorDataloader:
        """Initialize pointer to the current iterate."""
        self._ptr = 0
        return self

    def __next__(self) -> SynchronizedSensorData:
        """Return the next datum in the dataset."""
        result = self.__getitem__(self._ptr)
        self._ptr += 1
        return result

    def __getitem__(self, idx: int) -> SynchronizedSensorData:
        """Load the lidar point cloud and optionally the camera imagery and annotations.

        Grab the lidar sensor data and optionally the camera sensor data and annotations at the lidar record
            corresponding to the specified index.

        Args:
            idx: Index in [0, self.num_sweeps - 1].

        Returns:
            Mapping from sensor name to data for the lidar record corresponding to the specified index.
        """
        # Grab the lidar record at the specified index.
        # Selects data at a particular level of a MultiIndex.
        record: Tuple[str, str, int] = self.sensor_cache.xs(key="lidar", level=2).iloc[idx].name

        # Grab the identifying record fields.
        split, log_id, timestamp_ns = record
        log_lidar_records = self.sensor_cache.xs((split, log_id, "lidar")).index
        num_frames = len(log_lidar_records)

        idx = np.where(log_lidar_records == timestamp_ns)[0].item()

        log_dir = self.dataset_dir / split / log_id
        sensor_dir = log_dir / "sensors"
        lidar_feather_path = sensor_dir / "lidar" / f"{timestamp_ns}.feather"
        sweep = Sweep.from_feather(lidar_feather_path=lidar_feather_path)
        timestamp_city_SE3_ego_dict = read_city_SE3_ego(log_dir=log_dir)

        avm = ArgoverseStaticMap.from_map_dir(log_dir / "map", build_raster=True)

        # Construct output datum.
        datum = SynchronizedSensorData(
            sweep=sweep,
            log_id=log_id,
            timestamp_city_SE3_ego_dict=timestamp_city_SE3_ego_dict,
            sweep_number=idx,
            num_sweeps_in_log=num_frames,
            avm=avm,
            timestamp_ns=timestamp_ns,
        )

        # Load annotations if enabled.
        if self.with_annotations:
            datum.annotations = self._load_annotations(split, log_id, timestamp_ns)

        # Load camera imagery if enabled.
        if self.cam_names:
            datum.synchronized_imagery = self._load_synchronized_cams(split, sensor_dir, log_id, timestamp_ns)

        # Return datum at the specified index.
        return datum

    def _build_synchronization_cache(self) -> pd.DataFrame:
        """Build the synchronization records for lidar-camera synchronization.

        This function builds a set of records to efficiently associate auxiliary sensors
        to a target sensor. We use this function to associate the nanosecond vehicle
        timestamps of the lidar sweep to the nearest images from all 9 cameras (7 ring + 2 stereo).
        Once this dataframe is built, synchronized data can be queried in O(1) time.

        NOTE: This function is NOT intended to be used outside of SensorDataset initialization.

        Returns:
            (self.num_sweeps, self.num_sensors) DataFrame where each row corresponds to the nanosecond camera timestamp
            that is closest (in absolute value) to the corresponding nanonsecond lidar sweep timestamp.
        """
        logger.info("Building synchronization database ...")

        # Create list to store synchronized data frames.
        sync_list: List[pd.DataFrame] = []
        unique_sensor_names: List[str] = self.sensor_cache.index.unique(level=2).tolist()

        # Associate a 'source' sensor to a 'target' sensor for all available sensors.
        # For example, we associate the lidar sensor with each ring camera which
        # produces a mapping from lidar -> all-other-sensors.
        for src_sensor_name in unique_sensor_names:
            src_records = self.sensor_cache.xs(src_sensor_name, level=2, drop_level=False).reset_index()
            src_records = src_records.rename({"timestamp_ns": src_sensor_name}, axis=1).sort_values(src_sensor_name)

            # _Very_ important to convert to timedelta. Tolerance below causes precision loss otherwise.
            src_records[src_sensor_name] = pd.to_timedelta(src_records[src_sensor_name])
            for target_sensor_name in unique_sensor_names:
                if src_sensor_name == target_sensor_name:
                    continue
                target_records = self.sensor_cache.xs(target_sensor_name, level=2).reset_index()
                target_records = target_records.rename({"timestamp_ns": target_sensor_name}, axis=1).sort_values(
                    target_sensor_name
                )

                # Merge based on matching criterion.
                # _Very_ important to convert to timedelta. Tolerance below causes precision loss otherwise.
                target_records[target_sensor_name] = pd.to_timedelta(target_records[target_sensor_name])
                tolerence = pd.to_timedelta(CAM_SHUTTER_INTERVAL_MS / 2 * 1e6)
                if "ring" in src_sensor_name:
                    tolerence = pd.to_timedelta(LIDAR_SWEEP_INTERVAL_W_BUFFER_NS / 2)
                src_records = pd.merge_asof(
                    src_records,
                    target_records,
                    left_on=src_sensor_name,
                    right_on=target_sensor_name,
                    by=["split", "log_id"],
                    direction=self.matching_criterion,
                    tolerance=tolerence,
                )
            sync_list.append(src_records)
        sync_records = pd.concat(sync_list).reset_index(drop=True)
        return sync_records

    def find_closest_target_fpath(
        self,
        split: str,
        log_id: str,
        src_sensor_name: str,
        src_timestamp_ns: int,
        target_sensor_name: str,
    ) -> Optional[Path]:
        """Find the file path to the target sensor from a source sensor.

        Args:
            split: Dataset split.
            log_id: Vehicle log uuid.
            src_sensor_name: Name of the source sensor.
            src_timestamp_ns: Nanosecond timestamp of the source sensor (vehicle time).
            target_sensor_name: Name of the target sensor.

        Returns:
            The target sensor file path if it exists, otherwise None.

        Raises:
            RuntimeError: if the synchronization database (sync_records) has not been created.
        """
        if self.synchronization_cache is None:
            raise RuntimeError("Requested synchronized data, but the synchronization database has not been created.")

        src_timedelta_ns = pd.Timedelta(src_timestamp_ns)
        src_to_target_records = self.synchronization_cache.loc[(split, log_id, src_sensor_name)].set_index(
            src_sensor_name
        )
        index = src_to_target_records.index
        if src_timedelta_ns not in index:
            # This timestamp does not correspond to any lidar sweep.
            return None

        # Grab the synchronization record.
        target_timestamp_ns = src_to_target_records.loc[src_timedelta_ns, target_sensor_name]
        if pd.isna(target_timestamp_ns):
            # No match was found within tolerance.
            return None

        sensor_dir = self.dataset_dir / split / log_id / "sensors"
        valid_cameras = [x.value for x in list(RingCameras)] + [x.value for x in list(StereoCameras)]
        timestamp_ns_str = str(target_timestamp_ns.asm8.item())
        if target_sensor_name in valid_cameras:
            target_path = sensor_dir / "cameras" / target_sensor_name / f"{timestamp_ns_str}.jpg"
        else:
            target_path = sensor_dir / target_sensor_name / f"{timestamp_ns_str}.feather"
        return target_path

    def get_closest_img_fpath(self, split: str, log_id: str, cam_name: str, lidar_timestamp_ns: int) -> Optional[Path]:
        """Find the file path to the closest image from the reference camera name to the lidar timestamp.

        Args:
            split: Dataset split.
            log_id: Vehicle log uuid.
            cam_name: Name of the camera.
            lidar_timestamp_ns: Name of the target sensor.

        Returns:
            File path to observation from the camera (if it exists), otherwise None.
        """
        return self.find_closest_target_fpath(
            split=split,
            log_id=log_id,
            src_sensor_name="lidar",
            src_timestamp_ns=lidar_timestamp_ns,
            target_sensor_name=cam_name,
        )

    def get_closest_lidar_fpath(self, split: str, log_id: str, cam_name: str, cam_timestamp_ns: int) -> Optional[Path]:
        """Find the file path to the closest image from the lidar to the reference camera.

        Args:
            split: Dataset split.
            log_id: Vehicle log uuid.
            cam_name: Name of the camera.
            cam_timestamp_ns: Name of the target sensor.

        Returns:
            File path to observation from the lidar (if it exists), otherwise None.
        """
        return self.find_closest_target_fpath(
            split=split,
            log_id=log_id,
            src_sensor_name=cam_name,
            src_timestamp_ns=cam_timestamp_ns,
            target_sensor_name="lidar",
        )

    def _load_annotations(self, split: str, log_id: str, sweep_timestamp_ns: int) -> CuboidList:
        """Load the sweep annotations at the provided timestamp.

        Args:
            split: Split name.
            log_id: Log unique id.
            sweep_timestamp_ns: Nanosecond timestamp.

        Returns:
            Cuboid list of annotations.
        """
        annotations_feather_path = self.dataset_dir / split / log_id / "annotations.feather"

        # Load annotations from disk.
        # NOTE: This contains annotations for the ENTIRE sequence.
        # The sweep annotations are selected below.
        cuboid_list = CuboidList.from_feather(annotations_feather_path)
        cuboids = list(filter(lambda x: x.timestamp_ns == sweep_timestamp_ns, cuboid_list.cuboids))
        return CuboidList(cuboids=cuboids)

    def _load_synchronized_cams(
        self, split: str, sensor_dir: Path, log_id: str, sweep_timestamp_ns: int
    ) -> Optional[Dict[str, TimestampedImage]]:
        """Load the synchronized imagery for a lidar sweep.

        Args:
            split: Dataset split.
            sensor_dir: Sensor directory.
            log_id: Log unique id.
            sweep_timestamp_ns: Nanosecond timestamp.

        Returns:
            Mapping between camera names and synchronized images.

        Raises:
            RuntimeError: if the synchronization database (sync_records) has not been created.
        """
        if self.synchronization_cache is None:
            raise RuntimeError("Requested synchronized data, but the synchronization database has not been created.")

        cam_paths = [
            self.find_closest_target_fpath(
                split=split,
                log_id=log_id,
                src_sensor_name="lidar",
                target_sensor_name=cam_name.value,
                src_timestamp_ns=sweep_timestamp_ns,
            )
            for cam_name in self.cam_names
        ]

        log_dir = sensor_dir.parent
        cams: Dict[str, TimestampedImage] = {}
        for p in cam_paths:
            if p is not None:
                cams[p.parent.stem] = TimestampedImage(
                    img=read_img(p, channel_order="BGR"),
                    camera_model=PinholeCamera.from_feather(log_dir=log_dir, cam_name=p.parent.stem),
                    timestamp_ns=int(p.stem),
                )
        return cams
