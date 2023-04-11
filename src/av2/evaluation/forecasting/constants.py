"""Constants for forecasting challenge."""

from typing import Final

import numpy as np

from av2.evaluation.common.constants import CompetitionCategories

NUM_TIMESTEPS: Final = 6
TIME_DELTA: Final = 0.5
NUM_ELEMS: Final = 101
NUM_JOBS: Final = 8
NUM_DECIMALS: Final = 3


av2_classes: Final = tuple(x.value for x in CompetitionCategories)

CATEGORY_TO_VELOCITY: Final = {
    "REGULAR_VEHICLE": 2.36,
    "PEDESTRIAN": 0.80,
    "BICYCLIST": 3.61,
    "MOTORCYCLIST": 4.08,
    "WHEELED_RIDER": 2.03,
    "BOLLARD": 0.02,
    "CONSTRUCTION_CONE": 0.02,
    "SIGN": 0.05,
    "CONSTRUCTION_BARREL": 0.03,
    "STOP_SIGN": 0.09,
    "MOBILE_PEDESTRIAN_CROSSING_SIGN": 0.03,
    "LARGE_VEHICLE": 1.56,
    "BUS": 3.10,
    "BOX_TRUCK": 2.59,
    "TRUCK": 2.76,
    "VEHICULAR_TRAILER": 1.72,
    "TRUCK_CAB": 2.36,
    "SCHOOL_BUS": 4.44,
    "ARTICULATED_BUS": 4.58,
    "MESSAGE_BOARD_TRAILER": 0.41,
    "BICYCLE": 0.97,
    "MOTORCYCLE": 1.58,
    "WHEELED_DEVICE": 0.37,
    "WHEELCHAIR": 1.50,
    "STROLLER": 0.91,
    "DOG": 0.72,
}

FORECAST_SCALAR: Final = np.linspace(0, 1, NUM_TIMESTEPS + 1)
DISTANCE_THRESHOLDS_M: Final = [0.5, 1, 2, 4]
TP_THRESHOLD_M: Final = 2
MAX_DISPLACEMENT: Final = 50

VELOCITY_PROFILE: Final = ["static", "linear", "non-linear"]
