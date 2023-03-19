"""Useful constants for scene flow evaluation."""

from enum import Enum, unique
from typing import Dict, Final, List

from av2.datasets.sensor.constants import AnnotationCategories

CATEGORY_MAP: Final = {**{k.value: i + 1 for i, k in list(enumerate(AnnotationCategories))}, **{"NONE": 0}}

BACKGROUND_CATEGORIES: Final = (
    "BOLLARD",
    "CONSTRUCTION_BARREL",
    "CONSTRUCTION_CONE",
    "MOBILE_PEDESTRIAN_CROSSING_SIGN",
    "SIGN",
    "STOP_SIGN",
)
PEDESTRIAN_CATEGORIES: Final = (
    "ANIMAL",
    "STROLLER",
    "WHEELCHAIR",
    "OFFICIAL_SIGNALER",
)
SMALL_VEHICLE_CATEGORIES: Final = (
    "BICYCLE",
    "BICYCLIST",
    "MOTORCYCLE",
    "MOTORCYCLIST",
    "WHEELED_DEVICE",
    "WHEELED_RIDER",
)
VEHICLE_CATEGORIES: Final = (
    "ARTICULATED_BUS",
    "BOX_TRUCK",
    "BUS",
    "LARGE_VEHICLE",
    "RAILED_VEHICLE",
    "REGULAR_VEHICLE",
    "SCHOOL_BUS",
    "TRUCK",
    "TRUCK_CAB",
    "VEHICULAR_TRAILER",
    "TRAFFIC_LIGHT_TRAILER",
    "MESSAGE_BOARD_TRAILER",
)
ANIMAL_CATEGORIES: Final = ("ANIMAL", "DOG")


NO_CLASSES: Final = {"All": list(range(31))}
FOREGROUND_BACKGROUND: Final = {
    "Background": [-1],
    "Foreground": [
        CATEGORY_MAP[k]
        for k in (
            BACKGROUND_CATEGORIES
            + PEDESTRIAN_CATEGORIES
            + SMALL_VEHICLE_CATEGORIES
            + VEHICLE_CATEGORIES
            + ANIMAL_CATEGORIES
        )
    ],
}
PED_CYC_VEH_ANI: Final = {
    "Background": [-1],
    "Object": [CATEGORY_MAP[k] for k in BACKGROUND_CATEGORIES],
    "Pedestrian": [CATEGORY_MAP[k] for k in PEDESTRIAN_CATEGORIES],
    "Small Vehicle": [CATEGORY_MAP[k] for k in SMALL_VEHICLE_CATEGORIES],
    "Vehicle": [CATEGORY_MAP[k] for k in VEHICLE_CATEGORIES],
    "Animal": [CATEGORY_MAP[k] for k in ANIMAL_CATEGORIES],
}

FLOW_COLS: Final = ["flow_tx_m", "flow_ty_m", "flow_ty_m"]
