# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Unit tests for sweeps."""

from pathlib import Path

import numpy as np
import pytest

from av2.structures.sweep import Sweep
from av2.utils.typing import NDArrayByte, NDArrayFloat, NDArrayInt


@pytest.fixture
def dummy_sweep(test_data_root_dir: Path) -> Sweep:
    """Get a fake sweep containing two points."""
    path = (
        test_data_root_dir
        / "sensor_dataset_logs"
        / "dummy"
        / "sensors"
        / "lidar"
        / "315968663259918000.feather"
    )
    return Sweep.from_feather(path)


def test_sweep_from_feather(dummy_sweep: Sweep) -> None:
    """Test loading a sweep from a feather file."""
    xyz_expected: NDArrayFloat = np.array(
        [[-22.1875, 20.484375, 0.55029296875], [-20.609375, 19.1875, 1.30078125]]
    )
    intensity_expected: NDArrayByte = np.array([38, 5], dtype=np.uint8)
    laser_number_expected: NDArrayByte = np.array([19, 3], dtype=np.uint8)
    offset_ns_expected: NDArrayInt = np.array([253440, 283392], dtype=np.int32)
    timestamp_ns_expected: int = 315968663259918000

    assert np.array_equal(dummy_sweep.xyz, xyz_expected)
    assert np.array_equal(dummy_sweep.intensity, intensity_expected)
    assert np.array_equal(dummy_sweep.laser_number, laser_number_expected)
    assert np.array_equal(dummy_sweep.offset_ns, offset_ns_expected)
    assert dummy_sweep.timestamp_ns == timestamp_ns_expected
