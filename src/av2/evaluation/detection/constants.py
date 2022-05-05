# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""3D object detection evaluation constants."""

from enum import Enum, unique
from typing import Final, List

from av2.utils.constants import PI

MAX_SCALE_ERROR: Final[float] = 1.0
MAX_YAW_RAD_ERROR: Final[float] = PI

# Higher is better.
MIN_AP: Final[float] = 0.0
MIN_CDS: Final[float] = 0.0

# Lower is better.
MAX_NORMALIZED_ATE: Final[float] = 1.0
MAX_NORMALIZED_ASE: Final[float] = 1.0
MAX_NORMALIZED_AOE: Final[float] = 1.0

# Max number of boxes considered per class per scene.
MAX_NUM_BOXES: Final[int] = 500

NUM_DECIMALS: Final[int] = 3

TRANSLATION_COLS: Final[List[str]] = ["tx_m", "ty_m", "tz_m"]
DIMENSION_COLS: Final[List[str]] = ["length_m", "width_m", "height_m"]
QUAT_WXYZ_COLS: Final[List[str]] = ["qw", "qx", "qy", "qz"]


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
class CompetitionCategories(str, Enum):
    """Sensor dataset annotation categories."""

    ARTICULATED_BUS = "ARTICULATED_BUS"
    BICYCLE = "BICYCLE"
    BICYCLIST = "BICYCLIST"
    BOLLARD = "BOLLARD"
    BOX_TRUCK = "BOX_TRUCK"
    BUS = "BUS"
    CONSTRUCTION_BARREL = "CONSTRUCTION_BARREL"
    CONSTRUCTION_CONE = "CONSTRUCTION_CONE"
    DOG = "DOG"
    LARGE_VEHICLE = "LARGE_VEHICLE"
    MESSAGE_BOARD_TRAILER = "MESSAGE_BOARD_TRAILER"
    MOBILE_PEDESTRIAN_CROSSING_SIGN = "MOBILE_PEDESTRIAN_CROSSING_SIGN"
    MOTORCYCLE = "MOTORCYCLE"
    MOTORCYCLIST = "MOTORCYCLIST"
    PEDESTRIAN = "PEDESTRIAN"
    REGULAR_VEHICLE = "REGULAR_VEHICLE"
    SCHOOL_BUS = "SCHOOL_BUS"
    SIGN = "SIGN"
    STOP_SIGN = "STOP_SIGN"
    STROLLER = "STROLLER"
    TRUCK = "TRUCK"
    TRUCK_CAB = "TRUCK_CAB"
    VEHICULAR_TRAILER = "VEHICULAR_TRAILER"
    WHEELCHAIR = "WHEELCHAIR"
    WHEELED_DEVICE = "WHEELED_DEVICE"
    WHEELED_RIDER = "WHEELED_RIDER"


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
