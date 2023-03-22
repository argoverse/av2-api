from typing import Final

import numpy as np

NUM_TIMESTEPS: Final = 6
TIME_DELTA: Final = 0.5
NUM_ELEMS: Final = 101
NUM_JOBS: Final = 8

av2_classes = [
    "REGULAR_VEHICLE",
    "PEDESTRIAN",
    "BICYCLIST",
    "MOTORCYCLIST",
    "WHEELED_RIDER",
    "BOLLARD",
    "CONSTRUCTION_CONE",
    "SIGN",
    "CONSTRUCTION_BARREL",
    "STOP_SIGN",
    "MOBILE_PEDESTRIAN_CROSSING_SIGN",
    "LARGE_VEHICLE",
    "BUS",
    "BOX_TRUCK",
    "TRUCK",
    "VEHICULAR_TRAILER",
    "TRUCK_CAB",
    "SCHOOL_BUS",
    "ARTICULATED_BUS",
    "MESSAGE_BOARD_TRAILER",
    "BICYCLE",
    "MOTORCYCLE",
    "WHEELED_DEVICE",
    "WHEELCHAIR",
    "STROLLER",
    "DOG",
]

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

FORECAST_SCALAR = np.linspace(0, 1, NUM_TIMESTEPS + 1)
DISTANCE_THRESHOLDS_M: Final = [0.5, 1, 2, 4]
VELOCITY_PROFILE = ["static", "linear", "non-linear"]


def agent_velocity(agent):
    if "future_translation" in agent:  # ground_truth
        return (agent["future_translation"][0][:2] - agent["current_translation"][:2]) / TIME_DELTA

    else:  # predictions
        res = []
        for i in range(agent["prediction"].shape[0]):
            res.append((agent["prediction"][i][0][:2] - agent["current_translation"][:2]) / TIME_DELTA)

        return res


def trajectory_type(agent, class_velocity):
    if "future_translation" in agent:  # ground_truth
        time = agent["future_translation"].shape[0] * TIME_DELTA
        static_target = agent["current_translation"][:2]
        linear_target = agent["current_translation"][:2] + time * agent["velocity"][:2]

        final_position = agent["future_translation"][-1][:2]

        threshold = 1 + FORECAST_SCALAR[len(agent["future_translation"])] * class_velocity.get(agent["name"], 0)
        if np.linalg.norm(final_position - static_target) < threshold:
            return "static"
        elif np.linalg.norm(final_position - linear_target) < threshold:
            return "linear"
        else:
            return "non-linear"

    else:  # predictions
        res = []
        time = agent["prediction"].shape[1] * TIME_DELTA

        threshold = 1 + FORECAST_SCALAR[len(agent["prediction"])] * class_velocity.get(agent["name"], 0)
        for i in range(agent["prediction"].shape[0]):
            static_target = agent["current_translation"][:2]
            linear_target = agent["current_translation"][:2] + time * agent["velocity"][i][:2]

            final_position = agent["prediction"][i][-1][:2]

            if np.linalg.norm(final_position - static_target) < threshold:
                res.append("static")
            elif np.linalg.norm(final_position - linear_target) < threshold:
                res.append("linear")
            else:
                res.append("non-linear")

        return res


def center_distance(pred_box, gt_box):
    return np.linalg.norm(pred_box - gt_box)
