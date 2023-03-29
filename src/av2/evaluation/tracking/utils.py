"""Tracking evaluation utilities.

Detection and track data in a single frame are kept as a dictionary of names to numpy arrays.
This module provides helper functions for manipulating this data format.
"""
import os
import pickle
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Union, cast

import numpy as np
from tqdm import tqdm

from av2.utils.typing import NDArrayInt

Frame = Dict[str, Any]
Frames = List[Frame]
Sequences = Dict[str, Frames]

av2_classes: List[str] = [
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


def progressbar(itr: Iterable[Any], desc: Optional[str] = None, **kwargs: Dict[str, Any]) -> tqdm:
    """Create and return a tqdm progress bar object for the given iterable.

    Args:
        itr: The iterable object to be looped over.
        desc: A short description of the progress bar. Defaults to None.
        kwargs: Optional arguments to be passed to the tqdm constructor.

    Returns:
        tqdm progress bar
    """
    pbar = tqdm(itr, **kwargs)
    if desc:
        pbar.set_description(desc)
    return pbar


def save(obj: Any, path: str) -> None:  # noqa
    """Save an object to a file using pickle serialization.

    Args:
        obj: An object to be saved.
        path: A string representing the file path to save the object to.
    """
    dir = os.path.dirname(path)
    if dir != "":
        os.makedirs(dir, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load(path: str) -> Any:  # noqa
    """Load an object from file using pickle module.

    Args:
        path: File path

    Returns:
        Object or None if the file does not exist
    """
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def group_frames(frames_list: Frames) -> Sequences:
    """Group list of frames into dictionary by sequence id.

    Args:
        frames_list: List of frames, each containing a detections snapshot for a timestamp

    Returns:
        Dictionary of frames indexed by sequence id
    """
    frames_by_seq_id = defaultdict(list)
    frames_list = sorted(frames_list, key=lambda f: cast(int, f["timestamp_ns"]))
    for frame in frames_list:
        frames_by_seq_id[frame["seq_id"]].append(frame)
    return dict(frames_by_seq_id)


def ungroup_frames(frames_by_seq_id: Sequences) -> Frames:
    """Ungroup dictionary of frames into a list of frames.

    Args:
        frames_by_seq_id: dictionary of frames

    Returns:
        List of frames
    """
    ungrouped_frames = []
    for frames in frames_by_seq_id.values():
        ungrouped_frames.extend(frames)
    return ungrouped_frames


def index_array_values(array_dict: Frame, index: Union[int, NDArrayInt]) -> Frame:
    """Index each numpy array in dictionary.

    Args:
        array_dict: dictionary of numpy arrays
        index: index used to access each numpy array in array_dict

    Returns:
        Dictionary of numpy arrays, each indexed by the provided index
    """
    return {k: v[index] if isinstance(v, np.ndarray) else v for k, v in array_dict.items()}


def array_dict_iterator(array_dict: Frame, length: int) -> Iterable[Frame]:
    """Get an iterator over each index in array_dict.

    Args:
        array_dict: dictionary of numpy arrays
        length: number of elements to iterate over

    Returns:
        Iterator, each element is a dictionary of numpy arrays, indexed from 0 to (length-1)
    """
    return (index_array_values(array_dict, i) for i in range(length))


def concatenate_array_values(array_dicts: Frames) -> Frame:
    """Concatenates numpy arrays in list of dictionaries.

    Handles inconsistent keys (will skip missing keys)
    Does not concatenate non-numpy values (int, str), sets to value if all values are equal

    Args:
        array_dicts: list of dictionaries

    Returns:
        single dictionary of names to numpy arrays
    """
    combined = defaultdict(list)
    for array_dict in array_dicts:
        for k, v in array_dict.items():
            combined[k].append(v)
    concatenated = {}
    for k, vs in combined.items():
        if all(isinstance(v, np.ndarray) for v in vs):
            if any(v.size > 0 for v in vs):
                concatenated[k] = np.concatenate([v for v in vs if v.size > 0])
            else:
                concatenated[k] = vs[0]
        elif all(vs[0] == v for v in vs):
            concatenated[k] = vs[0]
    return concatenated


def filter_by_class_thresholds(frames_by_seq_id: Sequences, thresholds_by_class: Dict[str, float]) -> Sequences:
    """Filter detections, keeping only detections with score higher than the provided threshold for that class.

    If a class threshold is not provided, all detections in that class is filtered.

    Args:
        frames_by_seq_id: Dictionary of frames
        thresholds_by_class: Dictionary containing the score thresholds for each class

    Returns:
        Dictionary of frames, filtered by class score thresholds
    """
    frames = ungroup_frames(frames_by_seq_id)
    return group_frames(
        [
            concatenate_array_values(
                [
                    index_array_values(
                        frame,
                        (frame["name"] == class_name) & (frame["score"] >= threshold),
                    )
                    for class_name, threshold in thresholds_by_class.items()
                ]
            )
            for frame in frames
        ]
    )
