# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Unit tests on sensor data synchronization utilities."""

import tempfile
from pathlib import Path
from typing import Dict, Final, List

from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from av2.datasets.sensor.constants import RingCameras
from av2.datasets.sensor.sensor_dataloader import SensorDataloader

SENSOR_TIMESTAMPS_MS_DICT: Final[Dict[str, List[int]]] = {
    "ring_rear_left": [0, 50, 100, 150, 200, 250, 300, 350, 400, 450],
    "ring_side_left": [15, 65, 115, 165, 215, 265, 315, 365, 415, 465],
    "ring_front_left": [30, 80, 130, 180, 230, 280, 330, 380, 430, 480],
    "ring_front_center": [42, 92, 142, 192, 242, 292, 342, 392, 442, 492],
    "ring_front_right": [5, 55, 105, 155, 205, 255, 305, 355, 405, 455],
    "ring_side_right": [20, 70, 120, 170, 220, 270, 320, 370, 420, 470],
    "ring_rear_right": [35, 85, 135, 185, 235, 285, 335, 385, 435, 485],
    "lidar": [2, 102, 202, 303, 402, 502, 603, 702, 802, 903],
}


def _create_dummy_sensor_dataloader(log_id: str) -> SensorDataloader:
    """Create a dummy sensor dataloader."""
    sensor_dataset_dir = Path(tempfile.TemporaryDirectory().name)
    for sensor_name, timestamps_ms in SENSOR_TIMESTAMPS_MS_DICT.items():
        for t in timestamps_ms:
            if "ring" in sensor_name:
                fpath = Path(
                    sensor_dataset_dir,
                    "dummy",
                    log_id,
                    "sensors",
                    "cameras",
                    sensor_name,
                    f"{int(t*1e6)}.jpg",
                )
                Path(fpath).parent.mkdir(exist_ok=True, parents=True)
                fpath.open("w").close()
            elif "lidar" in sensor_name:
                fpath = Path(
                    sensor_dataset_dir,
                    "dummy",
                    log_id,
                    "sensors",
                    sensor_name,
                    f"{int(t*1e6)}.feather",
                )
                Path(fpath).parent.mkdir(exist_ok=True, parents=True)
                fpath.open("w").close()
    return SensorDataloader(dataset_dir=sensor_dataset_dir, with_cache=False)


def test_sensor_data_loader_milliseconds() -> None:
    """Test that the sensor dataset dataloader can synchronize lidar and image data.

    Given toy data in milliseconds, we write out dummy files at corresponding timestamps.

    (Sensor timestamps are real, and come from log 00a6ffc1-6ce9-3bc3-a060-6006e9893a1a).

      sensor_name  timestamp_ns  ring_front_center  ...
    0       lidar       2000000         42000000.0
    1       lidar     102000000         92000000.0
    2       lidar     202000000        192000000.0
    3       lidar     303000000        292000000.0
    4       lidar     402000000        392000000.0
    5       lidar     502000000        492000000.0
    6       lidar     603000000                NaN
    7       lidar     702000000                NaN
    8       lidar     802000000                NaN
    9       lidar     903000000                NaN
    """
    # 7x10 images, and 10 sweeps. Timestamps below given in human-readable milliseconds.
    log_id = "00a6ffc1-6ce9-3bc3-a060-6006e9893a1a"
    loader = _create_dummy_sensor_dataloader(log_id=log_id)

    #  LiDAR 402 -> matches to ring front center 392.
    img_fpath = loader.find_closest_target_fpath(
        split="dummy",
        log_id=log_id,
        src_sensor_name="lidar",
        src_timestamp_ns=int(402 * 1e6),
        target_sensor_name="ring_front_center",
    )

    assert isinstance(img_fpath, Path)
    # result should be 392 milliseconds (and then a conversion to nanoseconds by adding 6 zeros)
    assert img_fpath.name == "392" + "000000" + ".jpg"

    # nothing should be within bounds for this (valid lidar timestamp 903)
    img_fpath = loader.find_closest_target_fpath(
        split="dummy",
        log_id=log_id,
        src_sensor_name="lidar",
        target_sensor_name="ring_front_center",
        src_timestamp_ns=int(903 * 1e6),
    )
    assert img_fpath is None

    # nothing should be within bounds for this (invalid lidar timestamp 904)
    img_fpath = loader.find_closest_target_fpath(
        split="dummy",
        log_id=log_id,
        src_sensor_name="lidar",
        target_sensor_name="ring_front_center",
        src_timestamp_ns=int(904 * 1e6),
    )
    assert img_fpath is None

    # ring front center 392 -> matches to LiDAR 402.
    lidar_fpath = loader.find_closest_target_fpath(
        split="dummy",
        log_id=log_id,
        src_sensor_name="ring_front_center",
        target_sensor_name="lidar",
        src_timestamp_ns=int(392 * 1e6),
    )

    assert isinstance(lidar_fpath, Path)
    # result should be 402 milliseconds (and then a conversion to nanoseconds by adding 6 zeros)
    assert lidar_fpath.name == "402" + "000000.feather"

    # way outside of bounds
    lidar_fpath = loader.find_closest_target_fpath(
        split="dummy",
        log_id=log_id,
        src_sensor_name="ring_front_center",
        target_sensor_name="lidar",
        src_timestamp_ns=int(7000 * 1e6),
    )
    assert lidar_fpath is None

    # use the non-pandas implementation as a "brute-force" (BF) check.
    # read out the dataset root from the other dataloader's attributes.
    bf_loader = AV2SensorDataLoader(
        data_dir=loader.dataset_dir / "dummy", labels_dir=loader.dataset_dir / "dummy"
    )

    # for every image, make sure query result matches the brute-force query result.
    for ring_camera_enum in RingCameras:
        ring_camera_name = ring_camera_enum.value
        for cam_timestamp_ms in SENSOR_TIMESTAMPS_MS_DICT[ring_camera_name]:
            cam_timestamp_ns = int(cam_timestamp_ms * 1e6)
            result = loader.get_closest_lidar_fpath(
                split="dummy",
                log_id=log_id,
                cam_name=ring_camera_name,
                cam_timestamp_ns=cam_timestamp_ns,
            )
            bf_result = bf_loader.get_closest_lidar_fpath(
                log_id=log_id, cam_timestamp_ns=cam_timestamp_ns
            )
            assert result == bf_result

    # for every lidar sweep, make sure query result matches the brute-force query result.
    for lidar_timestamp_ms in SENSOR_TIMESTAMPS_MS_DICT["lidar"]:
        lidar_timestamp_ns = int(lidar_timestamp_ms * 1e6)
        for ring_camera_enum in list(RingCameras):
            ring_camera_name = ring_camera_enum.value
            result = loader.get_closest_img_fpath(
                split="dummy",
                log_id=log_id,
                cam_name=ring_camera_name,
                lidar_timestamp_ns=lidar_timestamp_ns,
            )
            bf_result = bf_loader.get_closest_img_fpath(
                log_id=log_id,
                cam_name=ring_camera_name,
                lidar_timestamp_ns=lidar_timestamp_ns,
            )
            assert result == bf_result


if __name__ == "__main__":
    test_sensor_data_loader_milliseconds()
