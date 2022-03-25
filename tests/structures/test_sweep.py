# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Unit tests for sweeps."""

from pathlib import Path

import numpy as np
import pytest

from av2.geometry.se3 import SE3
from av2.structures.sweep import Sweep
from av2.utils.typing import NDArrayByte, NDArrayFloat, NDArrayInt


@pytest.fixture
def dummy_sweep(test_data_root_dir: Path) -> Sweep:
    """Get a fake sweep containing two points."""
    path = test_data_root_dir / "sensor_dataset_logs" / "dummy" / "sensors" / "lidar" / "315968663259918000.feather"
    return Sweep.from_feather(path)


def test_sweep_from_feather(dummy_sweep: Sweep) -> None:
    """Test loading a sweep from a feather file."""
    xyz_expected: NDArrayFloat = np.array([[-22.1875, 20.484375, 0.55029296875], [-20.609375, 19.1875, 1.30078125]])
    intensity_expected: NDArrayByte = np.array([38, 5], dtype=np.uint8)
    laser_number_expected: NDArrayByte = np.array([19, 3], dtype=np.uint8)
    offset_ns_expected: NDArrayInt = np.array([253440, 283392], dtype=np.int32)
    timestamp_ns_expected: int = 315968663259918000

    assert np.array_equal(dummy_sweep.xyz, xyz_expected)
    assert np.array_equal(dummy_sweep.intensity, intensity_expected)
    assert np.array_equal(dummy_sweep.laser_number, laser_number_expected)
    assert np.array_equal(dummy_sweep.offset_ns, offset_ns_expected)
    assert dummy_sweep.timestamp_ns == timestamp_ns_expected


def test_prune_to_2d_bbox() -> None:
    """Ensure we can discard Sweep points that lie outside a specified box.

    Box indicated with dots:
       ..|..
       . | .
       . | .
      ---|------
       . . .
         |
         |
    """
    # fmt: off
    pts_xy: NDArrayFloat = np.array(
        [
            [-2.0, 2.0],  # will be discarded
            [2.0, 0],  # will be discarded
            [1.0, 2.0],
            [0.0, 1.0]
        ])
    # fmt: on
    pts_xyz: NDArrayFloat = np.hstack([pts_xy, np.ones((4, 1))])

    dummy_pose = SE3(np.eye(3), np.zeros(3))
    ego_SE3_down_lidar = dummy_pose
    ego_SE3_up_lidar = dummy_pose

    sweep = Sweep(
        xyz=pts_xyz,
        intensity=np.array([0, 1, 2, 3], dtype=np.uint8),
        laser_number=np.array([5, 6, 7, 8], dtype=np.uint8),
        offset_ns=np.array([9, 10, 11, 12], dtype=np.uint8),
        timestamp_ns=0,  # dummy value
        ego_SE3_up_lidar=ego_SE3_up_lidar,
        ego_SE3_down_lidar=ego_SE3_down_lidar,
    )

    xmin = -1
    ymin = -1
    xmax = 1
    ymax = 2

    pruned_agg_sweep = sweep.prune_to_2d_bbox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
    gt_pts: NDArrayFloat = np.array([[1.0, 2.0, 1.0], [0.0, 1.0, 1.0]])  # only last 2 points should remain
    assert np.allclose(pruned_agg_sweep.xyz, gt_pts)
