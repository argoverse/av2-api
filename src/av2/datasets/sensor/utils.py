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
    sensor_name = path.parent.stem

    # log_id is 2 directories up for the lidar filepaths, but 3 levels up for images
    # {log_id}/sensors/cameras/ring_*/*.jpg vs.
    # {log_id}/sensors/lidar/*.feather
    parent_idx = 2 if sensor_name == "lidar" else 3
    log_id = path.parents[parent_idx].stem
    sensor_name, timestamp_ns = path.parent.stem, int(path.stem)
    return {"log_id": log_id, "sensor_name": sensor_name, "timestamp_ns": timestamp_ns}
