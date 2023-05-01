# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Unit tests on utilities for converting AV2 city coordinates to UTM or WGS84 coordinate systems."""

import numpy as np

import av2.geometry.utm as geo_utils
from av2.geometry.utm import CityName
from av2.utils.typing import NDArrayFloat


def test_convert_city_coords_to_wgs84_atx() -> None:
    """Convert city coordinates from Austin, TX to GPS coordinates."""
    points_city: NDArrayFloat = np.array(
        [
            [1745.37, -1421.37],
            [1738.54, -1415.03],
            [1731.53, -1410.81],
        ]
    )

    wgs84_coords = geo_utils.convert_city_coords_to_wgs84(
        points_city, city_name=CityName.ATX
    )

    expected_wgs84_coords: NDArrayFloat = np.array(
        [
            [30.261642967615092, -97.72246957081633],
            [30.26170086362131, -97.72253982250783],
            [30.261739638233472, -97.72261222631731],
        ]
    )
    assert np.allclose(wgs84_coords, expected_wgs84_coords, atol=1e-4)


def test_convert_city_coords_to_wgs84_wdc() -> None:
    """Convert city coordinates from Washington, DC to GPS coordinates."""
    points_city: NDArrayFloat = np.array(
        [
            [1716.85, 4470.38],
            [2139.70, 4606.14],
        ]
    )

    wgs84_coords = geo_utils.convert_city_coords_to_wgs84(
        points_city, city_name=CityName.WDC
    )
    expected_wgs84_coords: NDArrayFloat = np.array(
        [
            [38.9299801515994, -77.0168603173312],
            [38.931286945069985, -77.0120195048271],
        ]
    )
    assert np.allclose(wgs84_coords, expected_wgs84_coords, atol=1e-4)


def test_convert_gps_to_utm() -> None:
    """Convert Pittsburgh city origin (given in WGS84) to UTM coordinates."""
    lat, long = 40.44177902989321, -80.01294377242584
    utm_coords = geo_utils.convert_gps_to_utm(lat, long, city_name=CityName.PIT)

    expected_utm_coords = 583710, 4477260
    assert np.allclose(utm_coords, expected_utm_coords, atol=0.01)
