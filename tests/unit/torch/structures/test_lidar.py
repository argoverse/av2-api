"""Unit tests for PyTorch Lidar sub-module."""

from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import torch
from torch.testing._comparison import assert_close

from av2.torch import LIDAR_COLUMNS
from av2.torch.structures.lidar import Lidar

TEST_DATA_DIR: Final = Path(__file__).parent.parent.parent.resolve() / "test_data"
SAMPLE_LOG_DIR: Final = (
    TEST_DATA_DIR / "sensor_dataset_logs" / "adcf7d18-0510-35b0-a2fa-b4cea13a6d76"
)


def test_build_lidar() -> None:
    """Test building a Lidar structure."""
    lidar_paths = sorted((SAMPLE_LOG_DIR / "sensors" / "lidar").glob("*.feather"))
    lidar_path = lidar_paths[0]
    frame = pd.read_feather(lidar_path)
    lidar_tensor = torch.as_tensor(
        frame[list(LIDAR_COLUMNS)].to_numpy().astype(np.float32)
    )

    lidar = Lidar(frame)
    assert_close(lidar.as_tensor(), lidar_tensor)
    assert_close(
        lidar.as_tensor(
            columns=(
                "y",
                "z",
            )
        ),
        lidar_tensor[:, 1:3],
    )
