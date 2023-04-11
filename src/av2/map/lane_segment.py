# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Classes describing lane segments and their associated properties."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, unique
from typing import Any, Dict, Final, List, Optional

import av2.geometry.infinity_norm_utils as infinity_norm_utils
import av2.geometry.interpolate as interp_utils
import av2.geometry.polyline_utils as polyline_utils
from av2.map.map_primitives import Polyline
from av2.utils.typing import NDArrayFloat

WPT_INFINITY_NORM_INTERP_NUM: Final[int] = 50

logger = logging.getLogger(__name__)


@unique
class LaneType(str, Enum):
    """Describes the sorts of objects that may use the lane for travel."""

    VEHICLE: str = "VEHICLE"
    BIKE: str = "BIKE"
    BUS: str = "BUS"


@unique
class LaneMarkType(str, Enum):
    """Color and pattern of a painted lane marking, located on either the left or ride side of a lane segment.

    The `NONE` type indicates that lane boundary is not marked by any paint; its extent should be implicitly inferred.
    """

    DASH_SOLID_YELLOW: str = "DASH_SOLID_YELLOW"
    DASH_SOLID_WHITE: str = "DASH_SOLID_WHITE"
    DASHED_WHITE: str = "DASHED_WHITE"
    DASHED_YELLOW: str = "DASHED_YELLOW"
    DOUBLE_SOLID_YELLOW: str = "DOUBLE_SOLID_YELLOW"
    DOUBLE_SOLID_WHITE: str = "DOUBLE_SOLID_WHITE"
    DOUBLE_DASH_YELLOW: str = "DOUBLE_DASH_YELLOW"
    DOUBLE_DASH_WHITE: str = "DOUBLE_DASH_WHITE"
    SOLID_YELLOW: str = "SOLID_YELLOW"
    SOLID_WHITE: str = "SOLID_WHITE"
    SOLID_DASH_WHITE: str = "SOLID_DASH_WHITE"
    SOLID_DASH_YELLOW: str = "SOLID_DASH_YELLOW"
    SOLID_BLUE: str = "SOLID_BLUE"
    NONE: str = "NONE"
    UNKNOWN: str = "UNKNOWN"


@dataclass
class LocalLaneMarking:
    """Information about a lane marking, representing either the left or right boundary of a lane segment.

    Args:
        mark_type: type of marking that represents the lane boundary, e.g. "SOLID_WHITE" or "DASHED_YELLOW".
        src_lane_id: id of lane segment to which this lane marking belongs.
        bound_side: string representing which side of a lane segment this marking represents, i.e. "left" or "right".
        polyline: array of shape (N,3) representing the waypoints of the lane segment's marked boundary.
    """

    mark_type: LaneMarkType
    src_lane_id: int
    bound_side: str
    polyline: NDArrayFloat


@dataclass(frozen=False)
class LaneSegment:
    """Vector representation of a single lane segment within a log-specific Argoverse 2.0 map.

    Args:
        id: unique identifier for this lane segment (guaranteed to be unique only within this local map).
        is_intersection: boolean value representing whether or not this lane segment lies within an intersection.
        lane_type: designation of which vehicle types may legally utilize this lane for travel.
        right_lane_boundary: 3d polyline representing the right lane boundary.
        left_lane_boundary: 3d polyline representing the right lane boundary
        right_mark_type: type of painted marking found along the right lane boundary.
        left_mark_type: type of painted marking found along the left lane boundary.
        predecessors: unique identifiers of lane segments that are predecessors of this object.
        successors: unique identifiers of lane segments that represent successor of this object.
            Note: this list will be empty if no successors exist.
        right_neighbor_id: unique identifier of the lane segment representing this object's right neighbor.
        left_neighbor_id: unique identifier of the lane segment representing this object's left neighbor.
    """

    id: int
    is_intersection: bool
    lane_type: LaneType
    right_lane_boundary: Polyline
    left_lane_boundary: Polyline
    right_mark_type: LaneMarkType
    left_mark_type: LaneMarkType
    predecessors: List[int]
    successors: List[int]
    right_neighbor_id: Optional[int] = None
    left_neighbor_id: Optional[int] = None

    @classmethod
    def from_dict(cls, json_data: Dict[str, Any]) -> LaneSegment:
        """Convert JSON to a LaneSegment instance."""
        return cls(
            id=json_data["id"],
            lane_type=LaneType(json_data["lane_type"]),
            right_lane_boundary=Polyline.from_json_data(json_data["right_lane_boundary"]),
            left_lane_boundary=Polyline.from_json_data(json_data["left_lane_boundary"]),
            right_mark_type=LaneMarkType(json_data["right_lane_mark_type"]),
            left_mark_type=LaneMarkType(json_data["left_lane_mark_type"]),
            right_neighbor_id=json_data["right_neighbor_id"],
            left_neighbor_id=json_data["left_neighbor_id"],
            predecessors=json_data["predecessors"],
            successors=json_data["successors"],
            is_intersection=json_data["is_intersection"],
        )

    @property
    def left_lane_marking(self) -> LocalLaneMarking:
        """Retrieve the left lane marking associated with this lane segment."""
        return LocalLaneMarking(
            mark_type=self.left_mark_type, src_lane_id=self.id, bound_side="left", polyline=self.left_lane_boundary.xyz
        )

    @property
    def right_lane_marking(self) -> LocalLaneMarking:
        """Retrieve the right lane marking associated with this lane segment."""
        return LocalLaneMarking(
            mark_type=self.right_mark_type,
            src_lane_id=self.id,
            bound_side="right",
            polyline=self.right_lane_boundary.xyz,
        )

    @property
    def polygon_boundary(self) -> NDArrayFloat:
        """Extract coordinates of the polygon formed by the lane segment's left and right boundaries.

        Returns:
            array of shape (N,3).
        """
        return polyline_utils.convert_lane_boundaries_to_polygon(
            self.right_lane_boundary.xyz, self.left_lane_boundary.xyz
        )

    def is_within_l_infinity_norm_radius(self, query_center: NDArrayFloat, search_radius_m: float) -> bool:
        """Whether any waypoint of lane boundaries falls within search_radius_m of query center, by l-infinity norm.

        Could have very long segment, with endpoints and all waypoints outside of radius, therefore cannot check just
        its endpoints.

        Args:
            query_center: array of shape (3,) representing 3d coordinates of query center.
            search_radius_m: distance threshold in meters (by infinity norm) to use for search.

        Returns:
            whether the lane segment has any waypoint within search_radius meters of the query center.
        """
        try:
            right_ln_bnd_interp = interp_utils.interp_arc(
                t=WPT_INFINITY_NORM_INTERP_NUM, points=self.right_lane_boundary.xyz[:, :2]
            )
            left_ln_bnd_interp = interp_utils.interp_arc(
                t=WPT_INFINITY_NORM_INTERP_NUM, points=self.left_lane_boundary.xyz[:, :2]
            )
        except Exception:
            logger.exception("Interpolation failed for lane segment %d", self.id)
            right_ln_bnd_interp = self.right_lane_boundary.xyz[:, :2]
            left_ln_bnd_interp = self.left_lane_boundary.xyz[:, :2]

        left_in_bounds = infinity_norm_utils.has_pts_in_infinity_norm_radius(
            right_ln_bnd_interp, query_center, search_radius_m
        )
        right_in_bounds = infinity_norm_utils.has_pts_in_infinity_norm_radius(
            left_ln_bnd_interp, query_center, search_radius_m
        )
        return left_in_bounds or right_in_bounds
