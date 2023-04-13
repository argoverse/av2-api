# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""3D object detection evaluation constants."""

from enum import Enum, unique
from typing import Dict, Final, Tuple

from av2.utils.constants import PI

MAX_SCALE_ERROR: Final = 1.0
MAX_YAW_RAD_ERROR: Final = PI

# Higher is better.
MIN_AP: Final = 0.0
MIN_CDS: Final = 0.0

# Lower is better.
MAX_NORMALIZED_ATE: Final = 1.0
MAX_NORMALIZED_ASE: Final = 1.0
MAX_NORMALIZED_AOE: Final = 1.0

# Max number of boxes considered per class per scene.
MAX_NUM_BOXES: Final = 500

NUM_DECIMALS: Final = 3

TRANSLATION_COLS: Final = (
    "tx_m",
    "ty_m",
    "tz_m",
)
DIMENSION_COLS: Final = (
    "length_m",
    "width_m",
    "height_m",
)
QUAT_WXYZ_COLS: Final = (
    "qw",
    "qx",
    "qy",
    "qz",
)

HIERARCHY: Final[Dict[str, Tuple[str, ...]]] = {
    "FINEGRAIN": (
        "REGULAR_VEHICLE",
        "LARGE_VEHICLE",
        "BUS",
        "BOX_TRUCK",
        "TRUCK",
        "VEHICULAR_TRAILER",
        "TRUCK_CAB",
        "SCHOOL_BUS",
        "ARTICULATED_BUS",
        "PEDESTRIAN",
        "WHEELED_RIDER",
        "BICYCLE",
        "BICYCLIST",
        "MOTORCYCLE",
        "MOTORCYCLIST",
        "WHEELED_DEVICE",
        "WHEELED_RIDER",
        "WHEELCHAIR",
        "STROLLER",
        "DOG",
        "BOLLARD",
        "CONSTRUCTION_CONE",
        "SIGN",
        "CONSTRUCTION_BARREL",
        "STOP_SIGN",
        "MOBILE_PEDESTRIAN_CROSSING_SIGN",
        "MESSAGE_BOARD_TRAILER",
    ),
    "GROUP": (
        "VEHICLE",
        "VEHICLE",
        "VEHICLE",
        "VEHICLE",
        "VEHICLE",
        "VEHICLE",
        "VEHICLE",
        "VEHICLE",
        "VEHICLE",
        "VULNERABLE",
        "VULNERABLE",
        "VULNERABLE",
        "VULNERABLE",
        "VULNERABLE",
        "VULNERABLE",
        "VULNERABLE",
        "VULNERABLE",
        "VULNERABLE",
        "VULNERABLE",
        "VULNERABLE",
        "MOVABLE",
        "MOVABLE",
        "MOVABLE",
        "MOVABLE",
        "MOVABLE",
        "MOVABLE",
        "MOVABLE",
    ),
    "OBJECT": (
        "OBJECT",
        "OBJECT",
        "OBJECT",
        "OBJECT",
        "OBJECT",
        "OBJECT",
        "OBJECT",
        "OBJECT",
        "OBJECT",
        "OBJECT",
        "OBJECT",
        "OBJECT",
        "OBJECT",
        "OBJECT",
        "OBJECT",
        "OBJECT",
        "OBJECT",
        "OBJECT",
        "OBJECT",
        "OBJECT",
        "OBJECT",
        "OBJECT",
        "OBJECT",
        "OBJECT",
        "OBJECT",
        "OBJECT",
        "OBJECT",
    ),
}

LCA: Final[Dict[str, Tuple[str, ...]]] = {
    "ARTICULATED_BUS": ("ARTICULATED_BUS",),
    "BICYCLE": ("BICYCLE",),
    "BICYCLIST": ("BICYCLIST",),
    "BOLLARD": ("BOLLARD",),
    "BOX_TRUCK": ("BOX_TRUCK",),
    "BUS": ("BUS",),
    "CONSTRUCTION_BARREL": ("CONSTRUCTION_BARREL",),
    "CONSTRUCTION_CONE": ("CONSTRUCTION_CONE",),
    "DOG": ("DOG",),
    "LARGE_VEHICLE": ("LARGE_VEHICLE",),
    "MESSAGE_BOARD_TRAILER": ("MESSAGE_BOARD_TRAILER",),
    "MOBILE_PEDESTRIAN_CROSSING_SIGN": ("MOBILE_PEDESTRIAN_CROSSING_SIGN",),
    "MOTORCYCLE": ("MOTORCYCLE",),
    "MOTORCYCLIST": ("MOTORCYCLIST",),
    "PEDESTRIAN": ("PEDESTRIAN",),
    "REGULAR_VEHICLE": ("REGULAR_VEHICLE",),
    "SCHOOL_BUS": ("SCHOOL_BUS",),
    "SIGN": ("SIGN",),
    "STOP_SIGN": ("STOP_SIGN",),
    "STROLLER": ("STROLLER",),
    "TRUCK": ("TRUCK",),
    "TRUCK_CAB": ("TRUCK_CAB",),
    "VEHICULAR_TRAILER": ("VEHICULAR_TRAILER",),
    "WHEELCHAIR": ("WHEELCHAIR",),
    "WHEELED_DEVICE": ("WHEELED_DEVICE",),
    "WHEELED_RIDER": ("WHEELED_RIDER",),
    "VEHICLE": (
        "REGULAR_VEHICLE",
        "LARGE_VEHICLE",
        "BUS",
        "BOX_TRUCK",
        "TRUCK",
        "VEHICULAR_TRAILER",
        "TRUCK_CAB",
        "SCHOOL_BUS",
        "ARTICULATED_BUS",
    ),
    "VULNERABLE": (
        "PEDESTRIAN",
        "WHEELED_RIDER",
        "BICYCLE",
        "BICYCLIST",
        "MOTORCYCLE",
        "MOTORCYCLIST",
        "WHEELED_DEVICE",
        "WHEELED_RIDER",
        "WHEELCHAIR",
        "STROLLER",
        "DOG",
    ),
    "MOVABLE": (
        "BOLLARD",
        "CONSTRUCTION_CONE",
        "SIGN",
        "CONSTRUCTION_BARREL",
        "STOP_SIGN",
        "MOBILE_PEDESTRIAN_CROSSING_SIGN",
        "MESSAGE_BOARD_TRAILER",
    ),
    "OBJECT": (
        "REGULAR_VEHICLE",
        "LARGE_VEHICLE",
        "BUS",
        "BOX_TRUCK",
        "TRUCK",
        "VEHICULAR_TRAILER",
        "TRUCK_CAB",
        "SCHOOL_BUS",
        "ARTICULATED_BUS",
        "PEDESTRIAN",
        "WHEELED_RIDER",
        "BICYCLE",
        "BICYCLIST",
        "MOTORCYCLE",
        "MOTORCYCLIST",
        "WHEELED_DEVICE",
        "WHEELED_RIDER",
        "WHEELCHAIR",
        "STROLLER",
        "DOG",
        "BOLLARD",
        "CONSTRUCTION_CONE",
        "SIGN",
        "CONSTRUCTION_BARREL",
        "STOP_SIGN",
        "MOBILE_PEDESTRIAN_CROSSING_SIGN",
        "MESSAGE_BOARD_TRAILER",
    ),
}

LCA_COLUMNS: Final = ("LCA=0", "LCA=1", "LCA=2")


@unique
class TruePositiveErrorNames(str, Enum):
    """True positive error names."""

    ATE = "ATE"
    ASE = "ASE"
    AOE = "AOE"


@unique
class MetricNames(str, Enum):
    """Metric names."""

    AP = "AP"
    ATE = TruePositiveErrorNames.ATE.value
    ASE = TruePositiveErrorNames.ASE.value
    AOE = TruePositiveErrorNames.AOE.value
    CDS = "CDS"


@unique
class AffinityType(str, Enum):
    """Affinity types for assigning detections to ground truth cuboids."""

    CENTER = "CENTER"


@unique
class DistanceType(str, Enum):
    """Distance types for computing metrics on true positive detections."""

    TRANSLATION = "TRANSLATION"
    SCALE = "SCALE"
    ORIENTATION = "ORIENTATION"


@unique
class InterpType(str, Enum):
    """Interpolation type for interpolating precision over recall samples."""

    ALL = "ALL"


@unique
class FilterMetricType(str, Enum):
    """Metric used to filter cuboids for evaluation."""

    EUCLIDEAN = "EUCLIDEAN"
