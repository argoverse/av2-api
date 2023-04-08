# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Tests for the classes associated w/ LaneSegment's and their properties (e.g. markings)."""

import unittest

from av2.map.lane_segment import LaneMarkType, LaneSegment, LaneType
from av2.map.map_primitives import Point, Polyline


class TestLaneSegment(unittest.TestCase):
    """LaneSegment unit testing class."""

    def test_from_dict(self) -> None:
        """Ensure object is generated correctly from a dictionary."""
        json_data = {
            "id": 93269421,
            "is_intersection": False,
            "lane_type": "VEHICLE",
            "left_lane_boundary": [
                {"x": 873.97, "y": -101.75, "z": -19.7},
                {"x": 880.31, "y": -101.44, "z": -19.7},
                {"x": 890.29, "y": -100.56, "z": -19.66},
            ],
            "left_lane_mark_type": "SOLID_YELLOW",
            "left_neighbor_id": None,
            "right_lane_boundary": [
                {"x": 874.01, "y": -105.15, "z": -19.58},
                {"x": 890.58, "y": -104.26, "z": -19.58},
            ],
            "right_lane_mark_type": "SOLID_WHITE",
            "right_neighbor_id": 93269520,
            "predecessors": [],
            "successors": [93269500],
        }
        lane_segment = LaneSegment.from_dict(json_data)

        assert isinstance(lane_segment, LaneSegment)

        assert lane_segment.id == 93269421
        assert not lane_segment.is_intersection
        assert lane_segment.lane_type == LaneType("VEHICLE")
        # fmt: off
        assert lane_segment.right_lane_boundary == Polyline(waypoints=[
            Point(874.01, -105.15, -19.58),
            Point(890.58, -104.26, -19.58)
        ])
        assert lane_segment.left_lane_boundary == Polyline(waypoints=[
            Point(873.97, -101.75, -19.7),
            Point(880.31, -101.44, -19.7),
            Point(890.29, -100.56, -19.66)
        ])
        # fmt: on
        assert lane_segment.right_mark_type == LaneMarkType("SOLID_WHITE")
        assert lane_segment.left_mark_type == LaneMarkType("SOLID_YELLOW")
        assert lane_segment.successors == [93269500]
        assert lane_segment.right_neighbor_id == 93269520
        assert lane_segment.left_neighbor_id is None
