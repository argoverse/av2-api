# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""3D object detection evaluation constants."""

from enum import Enum, unique
from typing import Final, Tuple

from av2.utils.constants import PI

MAX_SCALE_ERROR: Final[float] = 1.0
MAX_YAW_ERROR: Final[float] = PI

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


@unique
class AnnotationColumns(str, Enum):
    """Columns needed for evaluation in the input detection and ground truth dataframes."""

    LOG_ID = "log_id"
    TIMESTAMP_NS = "timestamp_ns"
    CATEGORY = "category"
    LENGTH_M = "length_m"
    WIDTH_M = "width_m"
    HEIGHT_M = "height_m"
    QW = "qw"
    QX = "qx"
    QY = "qy"
    QZ = "qz"
    TX_M = "tx_m"
    TY_M = "ty_m"
    TZ_M = "tz_m"
    SCORE = "score"

    @classmethod
    @property
    def DIMENSION_NAMES(cls) -> Tuple[str, str, str]:
        """Return the column names for the evaluation cuboids dataframes."""
        return cls.LENGTH_M.value, cls.WIDTH_M.value, cls.HEIGHT_M.value

    @classmethod
    @property
    def QUAT_COEFFICIENTS_WXYZ(cls) -> Tuple[str, str, str, str]:
        """Return the quanternion coefficient names (scalar first) for the evaluation cuboids dataframes."""
        return cls.QW.value, cls.QX.value, cls.QY.value, cls.QZ.value

    @classmethod
    @property
    def TRANSLATION_NAMES(cls) -> Tuple[str, str, str]:
        """Return the translation component names for the evaluation cuboids dataframes."""
        return cls.TX_M.value, cls.TY_M.value, cls.TZ_M.value


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

    ANIMAL = "ANIMAL"
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
    OFFICIAL_SIGNALER = "OFFICIAL_SIGNALER"
    PEDESTRIAN = "PEDESTRIAN"
    RAILED_VEHICLE = "RAILED_VEHICLE"
    REGULAR_VEHICLE = "REGULAR_VEHICLE"
    SCHOOL_BUS = "SCHOOL_BUS"
    SIGN = "SIGN"
    STOP_SIGN = "STOP_SIGN"
    STROLLER = "STROLLER"
    TRAFFIC_LIGHT_TRAILER = "TRAFFIC_LIGHT_TRAILER"
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
