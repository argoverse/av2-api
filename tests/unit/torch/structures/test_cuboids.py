"""Unit tests for PyTorch Cuboids sub-module."""

from pathlib import Path
from typing import Final, List

import numpy as np
import pandas as pd
import torch
from kornia.geometry.conversions import euler_from_quaternion
from torch.testing._comparison import assert_close

from av2.torch import XYZLWH_QWXYZ_COLUMNS
from av2.torch.structures.cuboids import CuboidMode, Cuboids

TEST_DATA_DIR: Final = Path(__file__).parent.parent.parent.resolve() / "test_data"
SAMPLE_LOG_DIR: Final = (
    TEST_DATA_DIR / "sensor_dataset_logs" / "adcf7d18-0510-35b0-a2fa-b4cea13a6d76"
)


def test_build_cuboids() -> None:
    """Test building the Cuboids structure."""
    annotations_path = SAMPLE_LOG_DIR / "annotations.feather"
    annotations_frame = pd.read_feather(annotations_path)
    cuboids_npy = (
        annotations_frame[list(XYZLWH_QWXYZ_COLUMNS)].to_numpy().astype(np.float32)
    )

    cuboids = Cuboids(annotations_frame)
    cuboids_xyzlwht = cuboids.as_tensor()

    cuboids_xyzlwh_qwxyz = torch.as_tensor(cuboids_npy)
    assert_close(cuboids_xyzlwht[:, :6], cuboids_xyzlwh_qwxyz[:, :6])

    w, x, y, z = cuboids_xyzlwh_qwxyz[:, 6:10].t()
    _, _, yaw = euler_from_quaternion(w, x, y, z)
    assert_close(cuboids_xyzlwht[:, 6], yaw)
    assert_close(
        cuboids.as_tensor(cuboid_mode=CuboidMode.XYZLWH_QWXYZ), cuboids_xyzlwh_qwxyz
    )

    track_uuid_expected: List[str] = annotations_frame["track_uuid"].to_list()
    assert cuboids.track_uuid == track_uuid_expected

    category_expected: List[str] = annotations_frame["category"].to_list()
    assert cuboids.category == category_expected
