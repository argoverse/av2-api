# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Unit tests on Argoverse 2.0 Sensor Dataset dataloader."""

from pathlib import Path

import numpy as np

from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from av2.geometry.se3 import SE3
from av2.utils.typing import NDArrayFloat


def test_get_subsampled_ego_trajectory(test_data_root_dir: Path) -> None:
    """Ensure we can sample the poses at a specific frequency.

    Args:
        test_data_root_dir: Path to the root dir for test data (provided via fixture).
    """
    log_id = "adcf7d18-0510-35b0-a2fa-b4cea13a6d76"

    dataroot = test_data_root_dir / "sensor_dataset_logs"
    loader = AV2SensorDataLoader(data_dir=dataroot, labels_dir=dataroot)

    # retrieve every pose! (sub-nanosecond precision)
    traj_ns = loader.get_subsampled_ego_trajectory(log_id=log_id, sample_rate_hz=1e9)

    assert traj_ns.shape == (2637, 2)

    # retrieve poses @ 1 Hz
    traj_1hz = loader.get_subsampled_ego_trajectory(log_id=log_id, sample_rate_hz=1)

    # 16 second log segment.
    assert traj_1hz.shape == (16, 2)


def test_get_city_SE3_ego(test_data_root_dir: Path) -> None:
    """Ensure we can obtain the egovehicle's pose in the city coordinate frame at a specific timestamp.

    Args:
        test_data_root_dir: Path to the root dir for test data (provided via fixture).
    """
    log_id = "adcf7d18-0510-35b0-a2fa-b4cea13a6d76"

    timestamp_ns = 315973157899927216

    dataroot = test_data_root_dir / "sensor_dataset_logs"
    loader = AV2SensorDataLoader(data_dir=dataroot, labels_dir=dataroot)

    city_SE3_egovehicle = loader.get_city_SE3_ego(
        log_id=log_id, timestamp_ns=timestamp_ns
    )

    assert isinstance(city_SE3_egovehicle, SE3)
    expected_translation: NDArrayFloat = np.array([1468.87, 211.51, 13.14])
    assert np.allclose(city_SE3_egovehicle.translation, expected_translation, atol=1e-2)
