# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Primitive types for vector maps (point, polyline)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from av2.utils.typing import NDArrayFloat


@dataclass
class Point:
    """Represents a single 3-d point."""

    x: float
    y: float
    z: float

    @property
    def xyz(self) -> NDArrayFloat:
        """Return (3,) vector."""
        return np.array([self.x, self.y, self.z])

    def __eq__(self, other: object) -> bool:
        """Check for equality with another Point object."""
        if not isinstance(other, Point):
            return False

        return all([self.x == other.x, self.y == other.y, self.z == other.z])


@dataclass
class Polyline:
    """Represents an ordered point set with consecutive adjacency."""

    waypoints: List[Point]

    @property
    def xyz(self) -> NDArrayFloat:
        """Return (N,3) array representing ordered waypoint coordinates."""
        return np.vstack([wpt.xyz for wpt in self.waypoints])

    @classmethod
    def from_json_data(cls, json_data: List[Dict[str, float]]) -> Polyline:
        """Generate object instance from list of dictionaries, read from JSON data.

        Args:
            json_data: JSON formatted data.

        Returns:
            `Polyline` object.
        """
        return cls(waypoints=[Point(x=v["x"], y=v["y"], z=v["z"]) for v in json_data])

    @classmethod
    def from_array(cls, array: NDArrayFloat) -> Polyline:
        """Generate object instance from a (N,3) Numpy array.

        Args:
            array: array of shape (N,3) representing N ordered three-dimensional points.

        Returns:
            Polyline object.
        """
        return cls(waypoints=[Point(*pt.tolist()) for pt in array])

    def __eq__(self, other: object) -> bool:
        """Check for equality with another Polyline object."""
        if not isinstance(other, Polyline):
            return False

        if len(self.waypoints) != len(other.waypoints):
            return False

        return all([wpt == wpt_ for wpt, wpt_ in zip(self.waypoints, other.waypoints)])

    def __len__(self) -> int:
        """Return the number of waypoints in the polyline."""
        return len(self.waypoints)
