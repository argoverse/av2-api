"""Constants for scene flow evaluation."""

from __future__ import annotations

from enum import Enum, unique
from typing import Final

from av2.datasets.sensor.constants import AnnotationCategories

SCENE_FLOW_DYNAMIC_THRESHOLD: Final = 0.05
SWEEP_PAIR_TIME_DELTA: Final = 0.1

CATEGORY_TO_INDEX: Final = {
    **{"NONE": 0},
    **{k.value: i + 1 for i, k in enumerate(AnnotationCategories)},
}


@unique
class SceneFlowMetricType(str, Enum):
    """Scene Flow metrics."""

    ACCURACY_RELAX = "ACCURACY_RELAX"
    ACCURACY_STRICT = "ACCURACY_STRICT"
    ANGLE_ERROR = "ANGLE_ERROR"
    EPE = "EPE"


@unique
class SegmentationMetricType(str, Enum):
    """Segmentation metrics."""

    TP = "TP"
    TN = "TN"
    FP = "FP"
    FN = "FN"


@unique
class InanimateCategories(str, Enum):
    """Annotation categories representing inanimate objects that aren't vehicles."""

    BOLLARD = "BOLLARD"
    CONSTRUCTION_BARREL = "CONSTRUCTION_BARREL"
    CONSTRUCTION_CONE = "CONSTRUCTION_CONE"
    MOBILE_PEDESTRIAN_CROSSING_SIGN = "MOBILE_PEDESTRIAN_CROSSING_SIGN"
    SIGN = "SIGN"
    STOP_SIGN = "STOP_SIGN"


@unique
class LeggedCategories(str, Enum):
    """Annotation categories representing objects that move using legs."""

    ANIMAL = "ANIMAL"
    DOG = "DOG"
    OFFICIAL_SIGNALER = "OFFICIAL_SIGNALER"
    PEDESTRIAN = "PEDESTRIAN"


@unique
class SmallVehicleCategories(str, Enum):
    """Annotation categories representing small vehicles."""

    BICYCLE = "BICYCLE"
    BICYCLIST = "BICYCLIST"
    MOTORCYCLE = "MOTORCYCLE"
    MOTORCYCLIST = "MOTORCYCLIST"
    STROLLER = "STROLLER"
    WHEELCHAIR = "WHEELCHAIR"
    WHEELED_DEVICE = "WHEELED_DEVICE"
    WHEELED_RIDER = "WHEELED_RIDER"


@unique
class VehicleCategories(str, Enum):
    """Annotation categories representing regular vehicles."""

    ARTICULATED_BUS = "ARTICULATED_BUS"
    BOX_TRUCK = "BOX_TRUCK"
    BUS = "BUS"
    LARGE_VEHICLE = "LARGE_VEHICLE"
    MESSAGE_BOARD_TRAILER = "MESSAGE_BOARD_TRAILER"
    RAILED_VEHICLE = "RAILED_VEHICLE"
    REGULAR_VEHICLE = "REGULAR_VEHICLE"
    SCHOOL_BUS = "SCHOOL_BUS"
    TRAFFIC_LIGHT_TRAILER = "TRAFFIC_LIGHT_TRAILER"
    TRUCK = "TRUCK"
    TRUCK_CAB = "TRUCK_CAB"
    VEHICULAR_TRAILER = "VEHICULAR_TRAILER"


@unique
class MetricBreakdownCategories(str, Enum):
    """Meta-categories for the scene flow task."""

    ALL = "All"
    BACKGROUND = "Background"
    FOREGROUND = "Foreground"


NO_CLASS_BREAKDOWN: Final = {MetricBreakdownCategories.ALL: list(range(31))}
FOREGROUND_BACKGROUND_BREAKDOWN: Final = {
    MetricBreakdownCategories.BACKGROUND: [0],
    MetricBreakdownCategories.FOREGROUND: [
        CATEGORY_TO_INDEX[k.value]
        for k in (
            list(InanimateCategories)
            + list(LeggedCategories)
            + list(SmallVehicleCategories)
            + list(VehicleCategories)
        )
    ],
}

FLOW_COLUMNS: Final = ("flow_tx_m", "flow_ty_m", "flow_tz_m")
