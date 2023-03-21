"""Constants for scene flow evaluation."""

from __future__ import annotations

from enum import Enum, unique
from typing import Final

import av2.evaluation.scene_flow.eval as eval
from av2.datasets.sensor.constants import AnnotationCategories

SCENE_FLOW_DYNAMIC_THRESHOLD: Final = 0.05
SWEEP_PAIR_TIME_DELTA: Final = 0.1

CATEGORY_TO_INDEX: Final = {**{"NONE": 0}, **{k.value: i + 1 for i, k in enumerate(AnnotationCategories)}}

FLOW_METRICS: Final = {
    "Accuracy Relax": eval.compute_accuracy_relax,
    "Accuracy Strict": eval.compute_accuracy_strict,
    "Angle Error": eval.compute_angle_error,
    "EPE": eval.compute_end_point_error,
}
SEGMENTATION_METRICS: Final = {
    "TP": eval.compute_true_positives,
    "TN": eval.compute_true_negatives,
    "FP": eval.compute_false_positives,
    "FN": eval.compute_false_negatives,
}


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
            list(InanimateCategories) + list(LeggedCategories) + list(SmallVehicleCategories) + list(VehicleCategories)
        )
    ],
}

FLOW_COLUMNS: Final = ("flow_tx_m", "flow_ty_m", "flow_tz_m")
