# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Utilities for converting AV2 city coordinates to UTM or WGS84 coordinate systems."""

from enum import Enum
from typing import Union

import numpy as np
from pyproj import Proj

from av2.utils.typing import NDArrayFloat, NDArrayInt

class CityName(str, Enum):
    """Abbreviations of names of cities featured in Argoverse 2."""
    ATX = "ATX" # Austin, Texas
    DTW = "DTW" # Detroit, Michigan
    MIA = "MIA" # Miami, Florida
    PAO = "PAO" # Palo Alto, California
    PIT = "PIT" # Pittsburgh, PA
    WDC = "WDC" # Washington, DC


# All are North UTM zones (Northern hemisphere)
UTM_ZONE_MAP = {
    CityName.ATX: 14,
    CityName.DTW: 17,
    CityName.MIA: 17,
    CityName.PAO: 10,
    CityName.PIT: 17,
    CityName.WDC: 18
}


# as (lat, long) tuples
CITY_ORIGIN_LATLONG_DICT = {
    CityName.ATX: ( 30.27464237939507,   -97.7404457407424),
    CityName.DTW: ( 42.29993066912924,  -83.17555750783717),
    CityName.MIA: ( 25.77452579915163,  -80.19656914449405),
    CityName.PAO: (         37.416065, -122.13571963362166),
    CityName.PIT: ( 40.44177902989321,  -80.01294377242584),
    CityName.WDC: (         38.889377,   -77.0355047439081)
}


def convert_gps_to_utm(lat: float, long: float, city_name):
    """Convert GPS coordinates to UTM coordinates.

    Args:
        lat: latitude of query point.
        long: longitude of query point.
        city_name: name of city, where query point is located.

    Returns:
        easting: corresponding UTM Easting.
        northing: corresponding UTM Northing.
    """
    projector = Proj(proj="utm", zone=UTM_ZONE_MAP[city_name], ellps="WGS84", datum="WGS84", units="m")

    # convert to UTM.
    easting, northing = projector(long, lat)

    return easting, northing


def convert_city_coords_to_utm(points_city: Union[NDArrayFloat, NDArrayInt], city_name: CityName) -> NDArrayFloat:
    """Convert city coordinates to UTM coordinates.

    Args:
        points_city: 2d coordinates, representing query points in the city coordinate frame.
        city_name: name of city, where query points are located.

    Returns:
        Points in the UTM coordinate system.
    """
    lat, long = CITY_ORIGIN_LATLONG_DICT[city_name]
    # get (easting, northing) of origin
    origin_utm = convert_gps_to_utm(lat=lat, long=long, city_name=city_name)

    points_utm = points_city + origin_utm
    return points_utm


def convert_city_coords_to_wgs84(points_city: Union[NDArrayFloat, NDArrayInt], city_name: CityName) -> NDArrayFloat:
    """ """

    points_utm = convert_city_coords_to_utm(points_city, city_name)

    projector = Proj(proj="utm", zone=UTM_ZONE_MAP[city_name], ellps="WGS84", datum="WGS84", units="m")

    points_wgs84 = []
    for easting, northing in points_utm:
        long, lat = projector(easting, northing, inverse=True)
        points_wgs84.append((lat, long))

    return np.array(points_wgs84)

