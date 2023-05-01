"""Constants for the forecasting challenge."""

from typing import Final

import numpy as np

from av2.evaluation import SensorCompetitionCategories

NUM_TIMESTEPS: Final = 6

AV2_CATEGORIES: Final = tuple(x.value for x in SensorCompetitionCategories)
CATEGORY_TO_VELOCITY_M_PER_S: Final = {
    "ARTICULATED_BUS": 4.58,
    "BICYCLE": 0.97,
    "BICYCLIST": 3.61,
    "BOLLARD": 0.02,
    "BOX_TRUCK": 2.59,
    "BUS": 3.10,
    "CONSTRUCTION_BARREL": 0.03,
    "CONSTRUCTION_CONE": 0.02,
    "DOG": 0.72,
    "LARGE_VEHICLE": 1.56,
    "MESSAGE_BOARD_TRAILER": 0.41,
    "MOBILE_PEDESTRIAN_CROSSING_SIGN": 0.03,
    "MOTORCYCLE": 1.58,
    "MOTORCYCLIST": 4.08,
    "PEDESTRIAN": 0.80,
    "REGULAR_VEHICLE": 2.36,
    "SCHOOL_BUS": 4.44,
    "SIGN": 0.05,
    "STOP_SIGN": 0.09,
    "STROLLER": 0.91,
    "TRUCK": 2.76,
    "TRUCK_CAB": 2.36,
    "VEHICULAR_TRAILER": 1.72,
    "WHEELCHAIR": 1.50,
    "WHEELED_DEVICE": 0.37,
    "WHEELED_RIDER": 2.03,
}

DISTANCE_THRESHOLDS_M: Final = (0.5, 1, 2, 4)
FORECAST_SCALAR: Final = np.linspace(0, 1, NUM_TIMESTEPS + 1)
MAX_DISPLACEMENT_M: Final = 50
NUM_DECIMALS: Final = 3
TIME_DELTA_S: Final = 0.5
TP_THRESHOLD_M: Final = 2
VELOCITY_TYPES: Final = ("static", "linear", "non-linear")
