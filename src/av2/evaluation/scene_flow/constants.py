"""Constants for scene flow evaluation."""

from enum import Enum, unique
from typing import Dict, Final, List

import av2.evaluation.scene_flow.eval as eval
from av2.datasets.sensor.constants import AnnotationCategories

SCENE_FLOW_DYNAMIC_THRESHOLD: Final = 0.05

CATEGORY_TO_INDEX: Final = {**{"NONE": 0}, **{k.value: i + 1 for i, k in enumerate(AnnotationCategories)}}

FLOW_METRICS: Final = {
    "EPE": eval.compute_end_point_error,
    "Accuracy Strict": eval.compute_accuracy_strict,
    "Accuracy Relax": eval.compute_accuracy_relax,
    "Angle Error": eval.compute_angle_error,
}
SEG_METRICS: Final = {
    "TP": eval.compute_true_positives,
    "TN": eval.compute_true_negatives,
    "FP": eval.compute_false_positives,
    "FN": eval.compute_false_negatives,
}

BACKGROUND_CATEGORIES: Final = (
    "BOLLARD",
    "CONSTRUCTION_BARREL",
    "CONSTRUCTION_CONE",
    "MOBILE_PEDESTRIAN_CROSSING_SIGN",
    "SIGN",
    "STOP_SIGN",
)
LEGGED_CATEGORIES: Final = ("ANIMAL", "DOG", "OFFICIAL_SIGNALER", "PEDESTRIAN")
SMALL_VEHICLE_CATEGORIES: Final = (
    "STROLLER",
    "WHEELCHAIR",
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


NO_CLASSES: Final = {"All": list(range(31))}
FOREGROUND_BACKGROUND: Final = {
    "Background": [0],
    "Foreground": [
        CATEGORY_TO_INDEX[k]
        for k in (BACKGROUND_CATEGORIES + LEGGED_CATEGORIES + SMALL_VEHICLE_CATEGORIES + VEHICLE_CATEGORIES)
    ],
}
PED_CYC_VEH_ANI: Final = {
    "Background": [0],
    "Object": [CATEGORY_TO_INDEX[k] for k in BACKGROUND_CATEGORIES],
    "Legged": [CATEGORY_TO_INDEX[k] for k in LEGGED_CATEGORIES],
    "Small Vehicle": [CATEGORY_TO_INDEX[k] for k in SMALL_VEHICLE_CATEGORIES],
    "Vehicle": [CATEGORY_TO_INDEX[k] for k in VEHICLE_CATEGORIES],
}

FLOW_COLUMNS: Final = ("flow_tx_m", "flow_ty_m", "flow_tz_m")
