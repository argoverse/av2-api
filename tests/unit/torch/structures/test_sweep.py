"""Unit tests for PyTorch Cuboids sub-module."""

from pathlib import Path
from typing import Final

import pandas as pd

from av2.torch.structures.cuboids import Cuboids
from av2.torch.structures.lidar import Lidar
from av2.torch.structures.sweep import Sweep
from av2.torch.structures.utils import SE3_from_frame

TEST_DATA_DIR: Final = Path(__file__).parent.parent.parent.resolve() / "test_data"
SAMPLE_LOG_DIR: Final = (
    TEST_DATA_DIR / "sensor_dataset_logs" / "adcf7d18-0510-35b0-a2fa-b4cea13a6d76"
)


def test_build_sweep() -> None:
    """Test building the Sweep structure."""
    annotations_path = SAMPLE_LOG_DIR / "annotations.feather"
    annotations_frame = pd.read_feather(annotations_path)
    cuboids = Cuboids(annotations_frame)

    lidar_paths = sorted((SAMPLE_LOG_DIR / "sensors" / "lidar").glob("*.feather"))
    lidar_path = lidar_paths[0]
    lidar_frame = pd.read_feather(lidar_path)
    lidar = Lidar(lidar_frame)

    city_pose_path = SAMPLE_LOG_DIR / "city_SE3_egovehicle.feather"
    city_pose_frame = pd.read_feather(city_pose_path)
    city_SE3_ego = SE3_from_frame(city_pose_frame)

    sweep_uuid = annotations_path.parent.stem, int(lidar_path.stem)
    sweep = Sweep(
        city_SE3_ego=city_SE3_ego, lidar=lidar, sweep_uuid=sweep_uuid, cuboids=cuboids
    )
    assert sweep is not None
