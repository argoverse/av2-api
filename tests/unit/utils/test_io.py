# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Unit tests for IO related utilities."""

from pathlib import Path

import numpy as np

import av2.utils.io as io_utils
from av2.utils.io import read_ego_SE3_sensor, read_feather


def test_read_feather(test_data_root_dir: Path) -> None:
    """Read an Apache Feather file."""
    feather_path = (
        test_data_root_dir
        / "sensor_dataset_logs"
        / "test_log"
        / "calibration"
        / "intrinsics.feather"
    )
    feather_file = read_feather(feather_path)

    assert feather_file is not None
    assert len(feather_file) > 0


def test_read_ego_SE3_sensor(test_data_root_dir: Path) -> None:
    """Read sensor extrinsics for a particular log."""
    ego_SE3_sensor_path = test_data_root_dir / "sensor_dataset_logs" / "test_log"
    sensor_name_to_sensor_pose = read_ego_SE3_sensor(ego_SE3_sensor_path)

    assert sensor_name_to_sensor_pose is not None
    assert len(sensor_name_to_sensor_pose) > 0


def test_read_lidar_sweep() -> None:
    """Read 3d point coordinates from a LiDAR sweep file from an example log."""
    log_id = "adcf7d18-0510-35b0-a2fa-b4cea13a6d76"
    EXAMPLE_LOG_DATA_ROOT = (
        Path(__file__).resolve().parent.parent
        / "test_data"
        / "sensor_dataset_logs"
        / log_id
    )

    fpath = EXAMPLE_LOG_DATA_ROOT / "sensors" / "lidar" / "315973157959879000.feather"
    arr = io_utils.read_lidar_sweep(fpath, attrib_spec="xyz")

    assert arr.shape == (100660, 3)
    assert arr.dtype == np.dtype(np.float64)
