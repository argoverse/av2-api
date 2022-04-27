# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Utilities for the sensor dataset."""

import logging
from pathlib import Path
from typing import Dict, Union

logger = logging.Logger(__name__)


def convert_path_to_named_record(path: Path) -> Dict[str, Union[str, int]]:
    """Convert the unique record for any sensor or pose data (log_id, sensor_name, timestamp_ns).

    Args:
        path: Path to the data.

    Returns:
        Mapping of name to record field.
    """
    sensor_path = path.parent
    sensor_name = sensor_path.stem
    log_path = sensor_path.parent.parent if sensor_name == "lidar" else sensor_path.parent.parent.parent

    # log_id is 2 directories up for the lidar filepaths, but 3 levels up for images
    # {log_id}/sensors/cameras/ring_*/*.jpg vs.
    # {log_id}/sensors/lidar/*.feather
    return {
        "split": log_path.parent.stem,
        "log_id": log_path.stem,
        "sensor_name": sensor_name,
        "timestamp_ns": int(path.stem),
    }
