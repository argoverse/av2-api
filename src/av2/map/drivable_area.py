# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Class describing a drivable area polygon."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from av2.map.map_primitives import Point
from av2.utils.typing import NDArrayFloat


@dataclass
class DrivableArea:
    """Represents a single polygon, not a polyline.

    Args:
        id: unique identifier.
        area_boundary: 3d vertices of polygon, representing the drivable area's boundary.
    """

    id: int
    area_boundary: List[Point]

    @property
    def xyz(self) -> NDArrayFloat:
        """Return (N,3) array representing the ordered 3d coordinates of the polygon vertices."""
        return np.vstack([wpt.xyz for wpt in self.area_boundary])

    @classmethod
    def from_dict(cls, json_data: Dict[str, Any]) -> DrivableArea:
        """Generate object instance from dictionary read from JSON data."""
        point_list = [Point(x=v["x"], y=v["y"], z=v["z"]) for v in json_data["area_boundary"]]
        # append the first vertex to the end of vertex list
        point_list.append(point_list[0])

        return cls(id=json_data["id"], area_boundary=point_list)
