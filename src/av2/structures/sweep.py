# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Sweep representation of lidar sensor data."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from av2.geometry.se3 import SE3
from av2.utils.io import read_ego_SE3_sensor, read_feather
from av2.utils.typing import NDArrayByte, NDArrayFloat, NDArrayInt


class Sweep:
    """Models a lidar sweep from a lidar sensor.

    A sweep refers to a set of points which were captured in a fixed interval [t,t+delta), where delta ~= (1/sensor_hz).
    Reference: https://en.wikipedia.org/wiki/Lidar

    NOTE: Argoverse 2 distributes sweeps which are from two, stacked Velodyne 32 beam sensors.
        These sensors each have different, overlapping fields-of-view.
        Both lidars have their own reference frame: up_lidar and down_lidar, respectively.
        We have egomotion-compensated the lidar sensor data to the egovehicle reference timestamp (`timestamp_ns`).
    """

    def __init__(
        self,
        xyz: NDArrayFloat,
        intensity: NDArrayByte,
        laser_number: NDArrayByte,
        offset_ns: Optional[NDArrayInt],
        timestamp_ns: int,
        ego_SE3_up_lidar: Optional[SE3],
        ego_SE3_down_lidar: Optional[SE3],
    ) -> None:
        """Initialize a lidar Sweep object.

        Args:
            xyz: (N,3) Points in Cartesian space (x,y,z) in meters.
            intensity: (N,) Intensity values in the interval [0,255] corresponding to each point.
            laser_number: (N,) Laser numbers in the interval [0,63] corresponding to the beam which generated the point.
            offset_ns: (N,) Nanosecond offsets _from_ the start of the sweep.
                Optional, as the TbV does not contain this information.
            timestamp_ns: Nanosecond timestamp _at_ the start of the sweep.
            ego_SE3_up_lidar: Pose of the up lidar in the egovehicle reference frame. Translation is in meters.
            ego_SE3_down_lidar: Pose of the down lidar in the egovehicle reference frame. Translation is in meters.
        """
        self.xyz: NDArrayFloat = xyz
        self.intensity: NDArrayByte = intensity
        self.laser_number: NDArrayByte = laser_number
        self.offset_ns: Optional[NDArrayInt] = offset_ns
        self.timestamp_ns: int = timestamp_ns
        self.ego_SE3_up_lidar: SE3 = ego_SE3_up_lidar
        self.ego_SE3_down_lidar: SE3 = ego_SE3_down_lidar

    def __len__(self) -> int:
        """Returns the number of returns in the aggregated sweep."""
        return int(self.xyz.shape[0])

    def __len__(self) -> int:
        """Return the number of LiDAR returns in the aggregated sweep."""
        return int(self.xyz.shape[0])

    @classmethod
    def from_feather(cls, lidar_feather_path: Path) -> Sweep:
        """Load a lidar sweep from a feather file.

        NOTE: The feather file is expected in AV2 format.
        NOTE: The sweep is in the _ego_ reference frame.

        The file should be a Apache Feather file and contain the following columns:
            x: Coordinate of each lidar return along the x-axis.
            y: Coordinate of each lidar return along the y-axis.
            z: Coordinate of each lidar return along the z-axis.
            intensity: Measure of radiant power per unit solid angle.
            laser_number: Laser which emitted the point return.
            offset_ns: Nanosecond delta from the sweep timestamp for the point return.

        Args:
            lidar_feather_path: Path to the lidar sweep feather file.

        Returns:
            Sweep object.
        """
        timestamp_ns = int(lidar_feather_path.stem)
        lidar = read_feather(lidar_feather_path)

        xyz = lidar.loc[:, ["x", "y", "z"]].to_numpy().astype(float)
        intensity = lidar.loc[:, ["intensity"]].to_numpy().squeeze()
        laser_number = lidar.loc[:, ["laser_number"]].to_numpy().squeeze()
        offset_ns = lidar.loc[:, ["offset_ns"]].to_numpy().squeeze() if "offset_ns" in lidar.keys() else None

        log_dir = lidar_feather_path.parent.parent.parent
        sensor_name_to_pose = read_ego_SE3_sensor(log_dir=log_dir)
        ego_SE3_up_lidar = sensor_name_to_pose.get("up_lidar", None)
        ego_SE3_down_lidar = sensor_name_to_pose.get("down_lidar", None)

        return cls(
            xyz=xyz,
            intensity=intensity,
            laser_number=laser_number,
            offset_ns=offset_ns,
            timestamp_ns=timestamp_ns,
            ego_SE3_up_lidar=ego_SE3_up_lidar,
            ego_SE3_down_lidar=ego_SE3_down_lidar,
        )

    def equalize_intensity_distribution(self) -> None:
        """Re-distribute mass of distribution."""
        self.intensity = equalize_distribution(self.intensity)

    def _transform_sweep_from(self, target_SE3_src: SE3) -> Sweep:
        """Transform 3d points in the sweep to a new reference frame.

        Note: This method exists so that `SE3` can play around with it, without a cyclic dependency.

        Args:
            target_SE3_src: SE(3) transformation. Assumes the sweep points are provided in the `src` frame.

        Returns:
            a new Sweep object, with points provided in the the `target` frame.
        """
        return Sweep(
            xyz=target_SE3_src.transform_from(self.xyz),
            intensity=self.intensity,
            laser_number=self.laser_number,
            offset_ns=self.offset_ns,
            timestamp_ns=self.timestamp_ns,
            ego_SE3_up_lidar=self.ego_SE3_up_lidar,
            ego_SE3_down_lidar=self.ego_SE3_down_lidar,
        )

    def prune_to_2d_bbox(self, xmin: float, ymin: float, xmax: float, ymax: float) -> Sweep:
        """Returns a new sweep, pruned to a specified two-dimensional box region.

        Args:
            xmin: lower bound (inclusive) on x-dimension.
            ymin: lower bound (inclusive) on y-dimension.
            xmax: upper bound (inclusive) on x-dimension.
            ymax: upper bound (inclusive) on y-dimension.

        Returns:
            new Sweep, with a subset of values that correspond to original points that fall within the
                specified 2-d bounding box.
        """
        x = self.xyz[:, 0]
        y = self.xyz[:, 1]
        is_valid = np.logical_and.reduce([xmin <= x, x <= xmax, ymin <= y, y <= ymax])

        pruned_sweep = Sweep(
            xyz=self.xyz[is_valid],
            intensity=self.intensity[is_valid],
            laser_number=self.laser_number[is_valid],
            offset_ns=self.offset_ns,
            timestamp_ns=self.timestamp_ns,
            ego_SE3_up_lidar=self.ego_SE3_up_lidar,
            ego_SE3_down_lidar=self.ego_SE3_down_lidar,
        )
        return pruned_sweep


def normalize_array(array: NDArrayFloat) -> NDArrayFloat:
    """Normalize array values, i.e. bring them to a range of [0,1].

    Args:
        array: Numpy array of any shape, representing unnormalized values.

    Returns:
        array: Numpy array of any shape, representing normalized values.
    """
    array -= array.min()  # shift min val to 0
    array /= array.max()  # shrink max val to 1
    return array


def equalize_distribution(reflectance: NDArrayByte) -> NDArrayByte:
    """Re-distribute mass of distribution.

    Note: we add one to reflectance to map 0 values to 0 under logarithm.

    Args:
        reflectance: intensity per return, in the range [0,255]

    Returns:
        log_reflectance_b: log of intensity per return, in the range [0,255]
    """
    log_reflectance: NDArrayFloat = np.log((reflectance + 1.0))  # note: must add a float, not an int.
    # normalize to [0,1]
    log_reflectance_n: NDArrayFloat = normalize_array(log_reflectance)
    # convert to byte in [0,255]
    log_reflectance_b: NDArrayByte = np.round(log_reflectance_n * 255).astype(np.uint8)
    return log_reflectance_b
