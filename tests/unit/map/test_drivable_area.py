# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Unit tests for the DrivableArea class."""


import unittest

from av2.map.drivable_area import DrivableArea


class TestDrivableArea(unittest.TestCase):
    """Class for unit testing the drivable area raster map."""

    def test_from_dict(self) -> None:
        """Ensure object is generated correctly from a dictionary.

        Note: 3 arbitrary waypoints taken from the dummy log map file.
        """
        json_data = {
            "area_boundary": [
                {"x": 905.09, "y": -148.95, "z": -19.19},
                {"x": 919.48, "y": -150.0, "z": -18.86},
                {"x": 905.14, "y": -150.0, "z": -19.18},
            ],
            "id": 4499430,
        }

        drivable_area = DrivableArea.from_dict(json_data)

        assert isinstance(drivable_area, DrivableArea)
        assert drivable_area.id == 4499430
        assert (
            len(drivable_area.area_boundary) == 4
        )  # first vertex is repeated as the last vertex
