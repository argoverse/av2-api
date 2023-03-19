"""Useful constants for scene flow evaluation."""

from enum import Enum, unique
from typing import Dict, Final, List


@unique
class ObjectCategory(str, Enum):
    """Names of all objects in the detection taxonomy."""

    ANIMAL = 0
    ARTICULATED_BUS = 1
    BICYCLE = 2
    BICYCLIST = 3
    BOLLARD = 4
    BOX_TRUCK = 5
    BUS = 6
    CONSTRUCTION_BARREL = 7
    CONSTRUCTION_CONE = 8
    DOG = 9
    LARGE_VEHICLE = 10
    MESSAGE_BOARD_TRAILER = 11
    MOBILE_PEDESTRIAN_CROSSING_SIGN = 12
    MOTORCYCLE = 13
    MOTORCYCLIST = 14
    OFFICIAL_SIGNALER = 15
    PEDESTRIAN = 16
    RAILED_VEHICLE = 17
    REGULAR_VEHICLE = 18
    SCHOOL_BUS = 19
    SIGN = 20
    STOP_SIGN = 21
    STROLLER = 22
    TRAFFIC_LIGHT_TRAILER = 23
    TRUCK = 24
    TRUCK_CAB = 25
    VEHICULAR_TRAILER = 26
    WHEELCHAIR = 27
    WHEELED_DEVICE = 28
    WHEELED_RIDER = 29
    NONE = -1


BACKGROUND_CATEGORIES: Final = [
    ObjectCategory.BOLLARD,
    ObjectCategory.CONSTRUCTION_BARREL,
    ObjectCategory.CONSTRUCTION_CONE,
    ObjectCategory.MOBILE_PEDESTRIAN_CROSSING_SIGN,
    ObjectCategory.SIGN,
    ObjectCategory.STOP_SIGN,
]
PEDESTRIAN_CATEGORIES: Final[List[ObjectCategory]] = [
    ObjectCategory.ANIMAL,
    ObjectCategory.STROLLER,
    ObjectCategory.WHEELCHAIR,
    ObjectCategory.OFFICIAL_SIGNALER,
]
SMALL_VEHICLE_CATEGORIES: Final[List[ObjectCategory]] = [
    ObjectCategory.BICYCLE,
    ObjectCategory.BICYCLIST,
    ObjectCategory.MOTORCYCLE,
    ObjectCategory.MOTORCYCLIST,
    ObjectCategory.WHEELED_DEVICE,
    ObjectCategory.WHEELED_RIDER,
]
VEHICLE_CATEGORIES: Final[List[ObjectCategory]] = [
    ObjectCategory.ARTICULATED_BUS,
    ObjectCategory.BOX_TRUCK,
    ObjectCategory.BUS,
    ObjectCategory.LARGE_VEHICLE,
    ObjectCategory.RAILED_VEHICLE,
    ObjectCategory.REGULAR_VEHICLE,
    ObjectCategory.SCHOOL_BUS,
    ObjectCategory.TRUCK,
    ObjectCategory.TRUCK_CAB,
    ObjectCategory.VEHICULAR_TRAILER,
    ObjectCategory.TRAFFIC_LIGHT_TRAILER,
    ObjectCategory.MESSAGE_BOARD_TRAILER,
]
ANIMAL_CATEGORIES: Final[List[ObjectCategory]] = [ObjectCategory.ANIMAL, ObjectCategory.DOG]


NO_CLASSES: Final = {"All": [k for k in range(-1, 30)]}
FOREGROUND_BACKGROUND: Final[Dict[str, List[int]]] = {
    "Background": [-1],
    "Foreground": [
        k.value
        for k in (
            BACKGROUND_CATEGORIES
            + PEDESTRIAN_CATEGORIES
            + SMALL_VEHICLE_CATEGORIES
            + VEHICLE_CATEGORIES
            + ANIMAL_CATEGORIES
        )
    ],
}
PED_CYC_VEH_ANI: Final[Dict[str, List[int]]] = {
    "Background": [-1],
    "Object": [k.value for k in BACKGROUND_CATEGORIES],
    "Pedestrian": [k.value for k in PEDESTRIAN_CATEGORIES],
    "Small Vehicle": [k.value for k in SMALL_VEHICLE_CATEGORIES],
    "Vehicle": [k.value for k in VEHICLE_CATEGORIES],
    "Animal": [k.value for k in ANIMAL_CATEGORIES],
}

FLOW_COLS: Final[List[str]] = ["flow_tx_m", "flow_ty_m", "flow_ty_m"]
