# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Unit tests on sensor data synchronization utilities."""

import os
from pathlib import Path

from av2.datasets.sensor.sensor_dataloader import SensorDataloader


def test_sensor_data_loader_milliseconds(tmpdir: "os.PathLike[str]") -> None:
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

    Args:
        tmpdir: Temp directory used in the test (provided via built-in fixture).
    """
    tmpdir = Path(tmpdir)
    log_id = "00a6ffc1-6ce9-3bc3-a060-6006e9893a1a"

    # 7x10 images, and 10 sweeps. Timestamps below given in human-readable milliseconds.
    sensor_timestamps_ms_dict = {
        "ring_rear_left": [0, 50, 100, 150, 200, 250, 300, 350, 400, 450],
        "ring_side_left": [15, 65, 115, 165, 215, 265, 315, 365, 415, 465],
        "ring_front_left": [30, 80, 130, 180, 230, 280, 330, 380, 430, 480],
        "ring_front_center": [42, 92, 142, 192, 242, 292, 342, 392, 442, 492],
        "ring_front_right": [5, 55, 105, 155, 205, 255, 305, 355, 405, 455],
        "ring_side_right": [20, 70, 120, 170, 220, 270, 320, 370, 420, 470],
        "ring_rear_right": [35, 85, 135, 185, 235, 285, 335, 385, 435, 485],
        "lidar": [2, 102, 202, 303, 402, 502, 603, 702, 802, 903],
    }

    for sensor_name, timestamps_ms in sensor_timestamps_ms_dict.items():
        for t in timestamps_ms:
            if "ring" in sensor_name:
                fpath = tmpdir / log_id / "sensors" / "cameras" / sensor_name / f"{int(t*1e6)}.jpg"
            elif "lidar" in sensor_name:
                fpath = tmpdir / log_id / "sensors" / sensor_name / f"{int(t*1e6)}.feather"
            fpath.parent.mkdir(exist_ok=True, parents=True)
            # create an empty file
            f = open(fpath, "w")
            f.close()

    loader = SensorDataloader(sensor_dataset_dir=tmpdir, with_cache=False)

    #  LiDAR 402 -> matches to ring front center 392.
    img_fpath = loader.get_closest_img_fpath(
        log_id=log_id, cam_name="ring_front_center", lidar_timestamp_ns=int(402 * 1e6)
    )
    assert isinstance(img_fpath, Path)
    # result should be 392 milliseconds (and then a conversion to nanoseconds by adding 6 zeros)
    print(img_fpath)
    assert img_fpath.name == "392" + "000000" + ".jpg"

    # nothing should be within bounds for this (valid lidar timestamp 903)
    img_fpath = loader.get_closest_img_fpath(
        log_id=log_id, cam_name="ring_front_center", lidar_timestamp_ns=int(903 * 1e6)
    )
    assert img_fpath is None

    # nothing should be within bounds for this (invalid lidar timestamp 904)
    img_fpath = loader.get_closest_img_fpath(
        log_id=log_id, cam_name="ring_front_center", lidar_timestamp_ns=int(904 * 1e6)
    )
    assert img_fpath is None

    # ring front center 392 -> matches to LiDAR 402.
    lidar_fpath = loader.get_closest_lidar_fpath(
        log_id=log_id, cam_name="ring_front_center", cam_timestamp_ns=int(392 * 1e6)
    )
    assert isinstance(lidar_fpath, Path)
    # result should be 402 milliseconds (and then a conversion to nanoseconds by adding 6 zeros)
    assert lidar_fpath.name == "402" + "000000.feather"

    # way outside of bounds
    lidar_fpath = loader.get_closest_lidar_fpath(
        log_id=log_id, cam_name="ring_front_center", cam_timestamp_ns=int(7000 * 1e6)
    )
    assert lidar_fpath is None
