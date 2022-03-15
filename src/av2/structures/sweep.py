# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Sweep representation of lidar sensor data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from av2.geometry.se3 import SE3
from av2.utils.io import read_ego_SE3_sensor, read_feather
from av2.utils.typing import NDArrayByte, NDArrayFloat, NDArrayInt


@dataclass
class Sweep:
    """Models a lidar sweep from a lidar sensor.

    A sweep refers to a set of points which were captured in a fixed interval [t,t+delta), where delta ~= (1/sensor_hz).
    Reference: https://en.wikipedia.org/wiki/Lidar

    NOTE: Argoverse 2 distributes sweeps which are from two, stacked Velodyne 32 beam sensors.
        These sensors each have different, overlapping fields-of-view.
        Both lidars have their own reference frame: up_lidar and down_lidar, respectively.
        We have egomotion-compensated the lidar sensor data to the egovehicle reference timestamp (`timestamp_ns`).

    Args:
        xyz: (N,3) Points in Cartesian space (x,y,z) in meters.
        intensity: (N,1) Intensity values in the interval [0,255] corresponding to each point.
        laser_number: (N,1) Laser numbers in the interval [0,63] corresponding to the beam which generated the point.
        offset_ns: (N,1) Nanosecond offsets _from_ the start of the sweep.
        timestamp_ns: Nanosecond timestamp _at_ the start of the sweep.
        ego_SE3_up_lidar: Pose of the up lidar in the egovehicle reference frame. Translation is in meters.
        ego_SE3_down_lidar: Pose of the down lidar in the egovehicle reference frame. Translation is in meters.
    """

    xyz: NDArrayFloat
    intensity: NDArrayByte
    laser_number: NDArrayByte
    offset_ns: NDArrayInt
    timestamp_ns: int
    ego_SE3_up_lidar: SE3
    ego_SE3_down_lidar: SE3

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
        offset_ns = lidar.loc[:, ["offset_ns"]].to_numpy().squeeze()

        log_dir = lidar_feather_path.parent.parent.parent
        sensor_name_to_pose = read_ego_SE3_sensor(log_dir=log_dir)
        ego_SE3_up_lidar = sensor_name_to_pose["up_lidar"]
        ego_SE3_down_lidar = sensor_name_to_pose["down_lidar"]

        return cls(
            xyz=xyz,
            intensity=intensity,
            laser_number=laser_number,
            offset_ns=offset_ns,
            timestamp_ns=timestamp_ns,
            ego_SE3_up_lidar=ego_SE3_up_lidar,
            ego_SE3_down_lidar=ego_SE3_down_lidar,
        )
