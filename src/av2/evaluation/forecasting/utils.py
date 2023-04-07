"""Forecasting evaluation utilities."""
from typing import Any, Dict, Final, Iterable, List, Union, cast
from av2.utils.typing import NDArrayInt, NDArrayFloat, Frame, Frames, Sequences
import numpy as np

NUM_TIMESTEPS: Final = 6
TIME_DELTA: Final = 0.5
NUM_ELEMS: Final = 101
NUM_JOBS: Final = 8

av2_classes: Final[List[str]] = [
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

CATEGORY_TO_VELOCITY: Final[Dict[str, float]] = {
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
VELOCITY_PROFILE: Final = ["static", "linear", "non-linear"]


def agent_velocity(agent: Dict[str, Any]) -> NDArrayFloat:
    """Get the agent velocity.

    Velocity calculated as the delta between the current translation and the translation in the next timestep.

    Args:
        agent: Dictionary containing the agent's translations

    Returns:
        Ground truth velocity for label agents or
        List of velocities corresponding to each movement prediction for predicted agents
    """
    if "future_translation" in agent:  # ground_truth
        return cast(NDArrayFloat, (agent["future_translation"][0][:2] - agent["current_translation"][:2]) / TIME_DELTA)

    else:  # predictions
        res = []
        for i in range(agent["prediction"].shape[0]):
            res.append((agent["prediction"][i][0][:2] - agent["current_translation"][:2]) / TIME_DELTA)

        return np.stack(res)


def trajectory_type(agent: Dict[str, Any], class_velocity: Dict[str, float]) -> Union[str, List[str]]:
    """Get the trajectory type, which is either static, linear or non-linear.

    Trajectory is static if the prediction error of a static forecast is below threshold.
    Trajectory is linear if the prediction error of the linear forecast is below threshold.
    Linear forecast is created by linearly extrapolating the current velocity.
    Trajectory is non-linear if it is neither static or linear.

    Args:
        agent: Dictionary containing the agent's translations and velocity
        class_velocity: Average velocity of each class, used to determine the prediction error threshold for each class

    Returns:
        String corresponding to the trajectory type for agents in the ground-truth annotations or
        List of strings, one trajectory for each movement prediction for predicted agents
    """
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


def center_distance(pred_box: NDArrayFloat, gt_box: NDArrayFloat) -> float:
    """Get euclidean distance between two centers.

    Args:
        pred_box: center 1
        gt_box: center 2

    Returns:
        distance between two centers
    """
    return cast(float, np.linalg.norm(pred_box - gt_box))


def array_dict_iterator(array_dict: Frame, length: int) -> Iterable[Frame]:
    """Get an iterator over each index in array_dict.

    Args:
        array_dict: dictionary of numpy arrays
        length: number of elements to iterate over

    Returns:
        Iterator, each element is a dictionary of numpy arrays, indexed from 0 to (length-1)
    """
    return (index_array_values(array_dict, i) for i in range(length))


def index_array_values(array_dict: Frame, index: Union[int, NDArrayInt]) -> Frame:
    """Index each numpy array in dictionary.

    Args:
        array_dict: dictionary of numpy arrays
        index: index used to access each numpy array in array_dict

    Returns:
        Dictionary of numpy arrays, each indexed by the provided index
    """
    return {k: v[index] if isinstance(v, np.ndarray) else v for k, v in array_dict.items()}


def annotate_frame_metadata(predictions: Sequences, ground_truth: Sequences, metadata_keys: List[str]) -> Sequences:
    """Index each numpy array in dictionary.

    Args:
        predictions: Forecast predictions
        ground_truth: Forecast ground truth
        metadata_keys : Ground truth keys to copy to predictions

    Returns:
        predictions: Forecast predictions with new metadat key
    """
    for seq_id in ground_truth.keys():
        for timestamp in ground_truth[seq_id].keys():
            copy_keys = {}

            for key in metadata_keys:
                copy_keys[key] = ground_truth[seq_id][timestamp][0][key]

            for i in range(len(predictions[seq_id][timestamp])):
                for key in metadata_keys:
                    predictions[seq_id][timestamp][i][key] = copy_keys[key]
