# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Class representing a pedestrian crossing (crosswalk)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

from av2.map.map_primitives import Polyline
from av2.utils.typing import NDArrayFloat


@dataclass
class PedestrianCrossing:
    """Represents a pedestrian crossing (i.e. crosswalk) as two edges along its principal axis.

    Both lines should be pointing in nominally the same direction and a pedestrian is expected to
    move either roughly parallel to both lines or anti-parallel to both lines.

    Args:
        id: unique identifier of this pedestrian crossing.
        edge1: 3d polyline representing one edge of the crosswalk, with 2 waypoints.
        edge2: 3d polyline representing the other edge of the crosswalk, with 2 waypoints.
    """

    id: int
    edge1: Polyline
    edge2: Polyline

    def get_edges_2d(self) -> Tuple[NDArrayFloat, NDArrayFloat]:
        """Retrieve the two principal edges of the crosswalk, in 2d.

        Returns:
            edge1: array of shape (2,2), a 2d polyline representing one edge of the crosswalk, with 2 waypoints.
            edge2: array of shape (2,2), a 2d polyline representing the other edge of the crosswalk, with 2 waypoints.
        """
        return (self.edge1.xyz[:, :2], self.edge2.xyz[:, :2])

    def __eq__(self, other: object) -> bool:
        """Check if two pedestrian crossing objects are equal, up to a tolerance."""
        if not isinstance(other, PedestrianCrossing):
            return False

        return np.allclose(self.edge1.xyz, other.edge1.xyz) and np.allclose(
            self.edge2.xyz, other.edge2.xyz
        )

    @classmethod
    def from_dict(cls, json_data: Dict[str, Any]) -> PedestrianCrossing:
        """Generate a PedestrianCrossing object from a dictionary read from JSON data."""
        edge1 = Polyline.from_json_data(json_data["edge1"])
        edge2 = Polyline.from_json_data(json_data["edge2"])

        return PedestrianCrossing(id=json_data["id"], edge1=edge1, edge2=edge2)

    @property
    def polygon(self) -> NDArrayFloat:
        """Return the vertices of the polygon representing the pedestrian crossing.

        Returns:
            array of shape (N,3) representing vertices. The first and last vertex that are provided are identical.
        """
        v0, v1 = self.edge1.xyz
        v2, v3 = self.edge2.xyz
        return np.array([v0, v1, v3, v2, v0])
