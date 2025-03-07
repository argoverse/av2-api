"""Argoverse 2 scenario mining evaluation.

Evaluation Metrics:
    HOTA: see https://arxiv.org/abs/2009.07736
    MOTA: see https://jivp-eurasipjournals.springeropen.com/articles/10.1155/2008/246309
    AMOTA: see https://arxiv.org/abs/2008.08063
"""

import contextlib
import json
import pickle
from copy import copy
from functools import partial
from itertools import chain
from pathlib import Path
from pprint import pprint
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union, cast

import click
import numpy as np
from packaging import version

#Replacing deprecated numpy types in TrackEval
if version.parse(np.__version__) >= version.parse("1.24.0"):
    print('hi')
    np.float = np.float32
    np.int = np.int32
    np.bool = bool

from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from trackeval.datasets._base_dataset import _BaseDataset
import matplotlib.pyplot as plt

from av2.evaluation.detection.utils import (
    compute_objects_in_roi_mask,
    load_mapped_avm_and_egoposes,
)
from av2.evaluation.scenario_mining import constants
from av2.evaluation.scenario_mining import utils as sm_utils
from av2.evaluation.scenario_mining.constants import SUBMETRIC_TO_METRIC_CLASS_NAME
from av2.utils.typing import NDArrayFloat, NDArrayInt
from av2.evaluation.typing import Sequences
import trackeval

import time
import traceback
from multiprocessing.pool import Pool
from functools import partial
import os
from trackeval import utils
from trackeval.utils import TrackEvalException
from trackeval import _timing
from trackeval.metrics import Count


class TrackEvalDataset(_BaseDataset):  # type: ignore
    """Dataset class to support tracking evaluation using the TrackEval library."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Store config."""
        super().__init__()
        self.gt_tracks = config["GT_TRACKS"]
        self.predicted_tracks = config["PREDICTED_TRACKS"]
        self.full_class_list = config.get("CLASSES", config["CLASSES_TO_EVAL"])
        self.class_list = config["CLASSES_TO_EVAL"]
        self.tracker_list = config["TRACKERS_TO_EVAL"]
        self.seq_list = config["SEQ_IDS_TO_EVAL"]
        self.output_fol = config["OUTPUT_FOLDER"]
        self.output_sub_fol = config["OUTPUT_SUB_FOLDER"]
        self.zero_distance = config["ZERO_DISTANCE"]
        self.should_classes_combine = config["SHOULD_CLASSES_COMBINE"]
        print(f"Using zero_distance={self.zero_distance}m")

    @staticmethod
    def get_default_dataset_config() -> Dict[str, Any]:
        """Get the default config.

        Returns:
            dictionary of the default config
        """
        default_config = {
            "GT_TRACKS": None,  # tracker_name -> seq id -> frames
            "PREDICTED_TRACKS": None,  # tracker_name -> seq id -> frames
            "SEQ_IDS_TO_EVAL": None,  # list of sequences ids to eval
            "CLASSES_TO_EVAL": None,
            "TRACKERS_TO_EVAL": None,
            "OUTPUT_FOLDER": None,  # Where to save eval results (if None, same as TRACKERS_FOLDER)
            "OUTPUT_SUB_FOLDER": "",  # Output files are saved in OUTPUT_FOLDER/tracker_name/OUTPUT_SUB_FOLDER
            "ZERO_DISTANCE": 2,
            "SHOULD_CLASSES_COMBINE": True,
        }
        return default_config

    def _load_raw_file(
        self, tracker: str, seq_id: Union[str, int], is_gt: bool
    ) -> Dict[str, Any]:
        """Get raw track data, from either trackers or ground truth."""
        tracks = (self.gt_tracks if is_gt else self.predicted_tracks)[tracker][seq_id]
        source = "gt" if is_gt else "tracker"

        ts = np.array([frame["timestamp_ns"] for frame in tracks])
        assert np.all(ts[:-1] < ts[1:]), "timestamps are not increasing"

        raw_data = {
            f"{source}_ids": [frame["track_id"] for frame in tracks],
            f"{source}_classes": [
                np.array([self.full_class_list.index(n) for n in frame["name"]])
                for frame in tracks
            ],
            f"{source}_dets": [
                np.concatenate((frame["translation_m"], frame["size"]), axis=-1)
                for frame in tracks
            ],
            "num_timesteps": len(tracks),
            "seq": seq_id,
        }
        if "score" in tracks[0]:
            raw_data[f"{source}_confidences"] = [frame["score"] for frame in tracks]
        return raw_data

    def get_preprocessed_seq_data(
        self, raw_data: Dict[str, Any], cls: str
    ) -> Dict[str, Any]:
        """Filter data to keep only one class and map id to 0 - n.

        Args:
            raw_data: dictionary of track data
            cls: name of class to keep

        Returns:
            Dictionary of processed track data of the specified class
        """
        data_keys = [
            "gt_ids",
            "tracker_ids",
            "gt_classes",
            "tracker_classes",
            "gt_dets",
            "tracker_dets",
            "tracker_confidences",
            "similarity_scores",
            "num_timesteps",
            "seq",
        ]
        data = {k: copy(raw_data[k]) for k in data_keys}
        cls_id = self.full_class_list.index(cls)

        for t in range(raw_data["num_timesteps"]):
            gt_to_keep_mask = data["gt_classes"][t] == cls_id
            data["gt_classes"][t] = data["gt_classes"][t][gt_to_keep_mask]
            data["gt_ids"][t] = data["gt_ids"][t][gt_to_keep_mask]
            data["gt_dets"][t] = data["gt_dets"][t][gt_to_keep_mask, :]

            tracker_to_keep_mask = data["tracker_classes"][t] == cls_id
            data["tracker_classes"][t] = data["tracker_classes"][t][
                tracker_to_keep_mask
            ]
            data["tracker_ids"][t] = data["tracker_ids"][t][tracker_to_keep_mask]
            data["tracker_dets"][t] = data["tracker_dets"][t][tracker_to_keep_mask, :]
            data["tracker_confidences"][t] = data["tracker_confidences"][t][
                tracker_to_keep_mask
            ]

            data["similarity_scores"][t] = data["similarity_scores"][t][
                :, tracker_to_keep_mask
            ][gt_to_keep_mask]

        # Map ids to 0 - n.
        unique_gt_ids = set(chain.from_iterable(data["gt_ids"]))
        unique_tracker_ids = set(chain.from_iterable(data["tracker_ids"]))
        data["gt_ids"] = self._map_ids(data["gt_ids"], unique_gt_ids)
        data["tracker_ids"] = self._map_ids(data["tracker_ids"], unique_tracker_ids)

        data["num_tracker_dets"] = sum(len(dets) for dets in data["tracker_dets"])
        data["num_gt_dets"] = sum(len(dets) for dets in data["gt_dets"])
        data["num_tracker_ids"] = len(unique_tracker_ids)
        data["num_gt_ids"] = len(unique_gt_ids)

        # Ensure again that ids are unique per timestep after preproc.
        self._check_unique_ids(data, after_preproc=True)

        return data

    def _map_ids(self, ids: List[Any], unique_ids: Iterable[Any]) -> List[NDArrayInt]:
        id_map = {id: i for i, id in enumerate(unique_ids)}
        return [
            np.array([id_map[id] for id in id_array], dtype=int) for id_array in ids
        ]

    def _calculate_similarities(
        self, gt_dets_t: NDArrayFloat, tracker_dets_t: NDArrayFloat
    ) -> NDArrayFloat:
        """Euclidean distance of the x, y translation coordinates."""
        gt_xy = gt_dets_t[:, :2]
        tracker_xy = tracker_dets_t[:, :2]
        sim = self._calculate_euclidean_similarity(
            gt_xy, tracker_xy, zero_distance=self.zero_distance
        )
        return cast(NDArrayFloat, sim)


def _plot_confusion_matrix(
    gt_classes: NDArrayInt, pred_classes: NDArrayInt, output_dir: str
) -> None:
    """Plots the confusion matrix for scenario mining.

    A true label indicates that the scenario matches the description.
    A false label indicates the scenario does not match the description.
    """
    # Create confusion matrix (2x2 for binary classification)
    cm = np.zeros((2, 2), dtype=int)

    # Fill the confusion matrix
    for true, pred in zip(gt_classes, pred_classes):
        cm[1 - true, 1 - pred] += 1

    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=(4, 4))
    cax = ax.imshow(cm, cmap="Wistia", interpolation="nearest")

    # Add text annotations (True Positives, False Positives, etc.)
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, cm[i, j], ha="center", va="center", color="black", fontsize=16
            )

    # Set axis labels and ticks
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Scenario Mining - Description Matches")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Positive", "Negative"])
    ax.set_yticklabels(["Positive", "Negative"])
    plt.setp(ax.get_yticklabels(), rotation=90, ha="right")

    # Display the plot
    if output_dir:
        plt.savefig(output_dir + "/eval_cm.png")
    plt.close()


def evaluate_tracking(
    labels: Sequences,
    track_predictions: Sequences,
    classes: List[str],
    tracker_name: str,
    output_dir: str,
    iou_threshold: float = 0.5,
) -> Tuple[Dict[str, Any], float]:
    """Evaluate a set of tracks against ground truth annotations using the TrackEval evaluation suite.

    Each sequences/log is evaluated separately.

    Args:
        labels: Dict[seq_id: List[frame]] Dictionary of ground truth annotations.
        track_predictions: Dict[seq_id: List[frame]] Dictionary of tracks.
        classes: List of classes to evaluate.
        tracker_name: Name of tracker.
        output_dir: Folder to save evaluation results.
        iou_threshold: IoU threshold for a True Positive match between a detection to a ground truth bounding box.

    frame is a dictionary with the following format
        {
            sequences_id: [
                {
                    "timestamp_ns": int, # nano seconds
                    "track_id": np.ndarray[I],
                    "translation_m": np.ndarray[I, 3],
                    "size": np.ndarray[I, 3],
                    "yaw": np.ndarray[I],
                    "velocity_m_per_s": np.ndarray[I, 3],
                    "label": np.ndarray[I],
                    "score": np.ndarray[I],
                    "name": np.ndarray[I],
                    ...
                }
            ]
        }
    where I is the number of objects in the frame.

    Returns:
        Dictionary of metric values.
    """
    labels_id_ts = set(
        (frame["seq_id"], frame["timestamp_ns"])
        for frame in sm_utils.ungroup_frames(labels)
    )
    predictions_id_ts = set(
        (frame["seq_id"], frame["timestamp_ns"])
        for frame in sm_utils.ungroup_frames(track_predictions)
    )
    assert (
        labels_id_ts == predictions_id_ts
    ), "sequences ids and timestamp_ns in labels and predictions don't match"
    metrics_config = {
        "METRICS": ["HOTA"],
        "THRESHOLD": iou_threshold,
    }
    metric_names = cast(List[str], metrics_config["METRICS"])
    metrics_list = [
        getattr(trackeval.metrics, metric)(metrics_config) for metric in metric_names
    ]
    dataset_config = {
        **TrackEvalDataset.get_default_dataset_config(),
        "GT_TRACKS": {tracker_name: labels},
        "PREDICTED_TRACKS": {tracker_name: track_predictions},
        "SEQ_IDS_TO_EVAL": list(labels.keys()),
        "CLASSES_TO_EVAL": classes,
        "TRACKERS_TO_EVAL": [tracker_name],
        "OUTPUT_FOLDER": output_dir,
        "SHOULD_CLASSES_COMBINE": False,
    }

    evaluator = trackeval.Evaluator(
        {
            **trackeval.Evaluator.get_default_eval_config(),
            "TIME_PROGRESS": True,
            "PLOT_CURVES": True,
            "OUTPUT_SUMMARY": True,
        }
    )

    dataset = TrackEvalDataset(dataset_config)
    full_result, _ = evaluator.evaluate(
        [dataset],
        metrics_list,
    )

    trackers, _, classes = dataset.get_eval_info()
    tracker = trackers[0]

    tlap_by_seq = np.zeros((len(labels), len(classes)))
    num_gt_by_seq = np.zeros((len(labels), len(classes)))

    for i, seq_id in enumerate(labels.keys()):
        raw_data = dataset.get_raw_seq_data(tracker, seq_id)
        for j, clas in enumerate(classes):
            data = dataset.get_preprocessed_seq_data(raw_data, clas)

            num_gt_ids = len(data["gt_ids"])
            tlap, precisions, recalls = calculate_TempLocAP_merge(data)
            tlap_by_seq[i, j] = tlap
            num_gt_by_seq[i, j] = num_gt_ids

    combined_tlap = np.average(tlap_by_seq, axis=1, weights=num_gt_by_seq)
    referred_tlap = float(
        combined_tlap[np.where(np.array(classes) == "REFERRED_OBJECT")]
    )

    return cast(Dict[str, Any], full_result), referred_tlap


def _tune_score_thresholds(
    labels: Sequences,
    track_predictions: Sequences,
    objective_metric: str,
    classes: List[str],
    num_thresholds: int = 10,
    iou_threshold: float = 0.5,
    match_distance_m: int = 2,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Find the optimal score thresholds to optimize the objective metric.

    Each class is processed independently.

    Args:
        labels: Dictionary of ground truth annotations
        track_predictions: Dictionary of tracks
        objective_metric: Name of the metric to optimize, one of HOTA or MOTA
        classes: List of classes to evaluate
        num_thresholds: Number of score thresholds to try
        iou_threshold: IoU threshold for a True Positive match between a detection to a ground truth bounding box
        match_distance_m: Maximum euclidean distance threshold for a match

    Returns:
        optimal_score_threshold_by_class: Dictionary of class name to optimal score threshold
        optimal_metric_values_by_class: Dictionary of class name to metric value with the optimal score threshold
        mean_metric_values_by_class: Dictionary of class name to metric value averaged over recall levels
    """
    metric_class = SUBMETRIC_TO_METRIC_CLASS_NAME[objective_metric]
    metrics_config = {
        "METRICS": [metric_class],
        "THRESHOLD": iou_threshold,
        "PRINT_CONFIG": False,
    }
    metrics_list = [
        getattr(trackeval.metrics, metric_name)(metrics_config)
        for metric_name in cast(List[str], metrics_config["METRICS"])
    ]
    dataset_config = {
        **TrackEvalDataset.get_default_dataset_config(),
        "GT_TRACKS": {"tracker": labels},
        "PREDICTED_TRACKS": {"tracker": track_predictions},
        "SEQ_IDS_TO_EVAL": list(labels.keys()),
        "CLASSES_TO_EVAL": classes,
        "TRACKERS_TO_EVAL": ["tracker"],
        "OUTPUT_FOLDER": "tmp",
    }
    evaluator = trackeval.Evaluator(
        {
            **trackeval.Evaluator.get_default_eval_config(),
            "PRINT_RESULTS": False,
            "PRINT_CONFIG": False,
            "TIME_PROGRESS": False,
            "OUTPUT_SUMMARY": False,
            "OUTPUT_DETAILED": False,
            "PLOT_CURVES": False,
        }
    )

    score_thresholds_by_class = {}
    sim_func = partial(_xy_center_similarity, zero_distance=match_distance_m)
    for name in classes:
        single_cls_labels = _filter_by_class(labels, name)
        single_cls_predictions = _filter_by_class(track_predictions, name)
        score_thresholds_by_class[name] = _calculate_score_thresholds(
            single_cls_labels,
            single_cls_predictions,
            sim_func,
            num_thresholds=num_thresholds,
        )

    metric_results = []
    for threshold_i in tqdm(
        range(num_thresholds), "calculating optimal track score thresholds"
    ):
        score_threshold_by_class = {
            n: score_thresholds_by_class[n][threshold_i] for n in classes
        }
        filtered_predictions = sm_utils.filter_by_class_thresholds(
            track_predictions, score_threshold_by_class
        )
        with contextlib.redirect_stdout(
            None
        ):  # silence print statements from TrackEval
            result_for_threshold, _ = evaluator.evaluate(
                [
                    TrackEvalDataset(
                        {
                            **dataset_config,
                            "PREDICTED_TRACKS": {"tracker": filtered_predictions},
                        }
                    )
                ],
                metrics_list,
            )
        metric_results.append(
            result_for_threshold["TrackEvalDataset"]["tracker"]["COMBINED_SEQ"]
        )

    optimal_score_threshold_by_class = {}
    optimal_metric_values_by_class = {}
    mean_metric_values_by_class = {}
    for name in classes:
        metric_values = [
            r[name][metric_class][objective_metric] for r in metric_results
        ]
        metric_values = [
            np.mean(v) if isinstance(v, np.ndarray) else v for v in metric_values
        ]
        optimal_threshold = score_thresholds_by_class[name][np.argmax(metric_values)]
        optimal_score_threshold_by_class[name] = optimal_threshold
        optimal_metric_values_by_class[name] = max(0, np.max(metric_values))
        mean_metric_values_by_class[name] = np.nanmean(
            np.array(metric_values).clip(min=0)
        )
    return (
        optimal_score_threshold_by_class,
        optimal_metric_values_by_class,
        mean_metric_values_by_class,
    )


def _filter_by_class(detections: Any, name: str) -> Any:
    return sm_utils.group_frames(
        [
            sm_utils.index_array_values(f, f["name"] == name)
            for f in sm_utils.ungroup_frames(detections)
        ]
    )


def _calculate_score_thresholds(
    labels: Sequences,
    predictions: Sequences,
    sim_func: Callable[[NDArrayFloat, NDArrayFloat], NDArrayFloat],
    num_thresholds: int = 40,
    min_recall: float = 0.1,
) -> NDArrayFloat:
    scores, n_gt = _calculate_matched_scores(labels, predictions, sim_func)
    recall_thresholds = np.linspace(min_recall, 1, num_thresholds).round(12)[::-1]
    if len(scores) == 0:
        return np.zeros_like(recall_thresholds)
    score_thresholds = _recall_to_scores(
        scores, recall_threshold=recall_thresholds, n_gt=n_gt
    )
    score_thresholds = np.nan_to_num(score_thresholds, nan=0)
    return score_thresholds


def _calculate_matched_scores(
    labels: Sequences,
    predictions: Sequences,
    sim_func: Callable[[NDArrayFloat, NDArrayFloat], NDArrayFloat],
) -> Tuple[NDArrayFloat, int]:
    scores = []
    n_gt = 0
    num_tp = 0
    for seq_id in labels:
        for label_frame, prediction_frame in zip(labels[seq_id], predictions[seq_id]):
            sim = sim_func(
                label_frame["translation_m"], prediction_frame["translation_m"]
            )
            match_rows, match_cols = linear_sum_assignment(-sim)
            scores.append(prediction_frame["score"][match_cols])
            n_gt += len(label_frame["translation_m"])
            num_tp += len(match_cols)

    scores_array = np.concatenate(scores)
    return scores_array, n_gt


def _recall_to_scores(
    scores: NDArrayFloat, recall_threshold: NDArrayFloat, n_gt: int
) -> NDArrayFloat:
    # Sort scores.
    scores.sort()
    scores = scores[::-1]

    # Determine thresholds.
    recall_values = np.arange(1, len(scores) + 1) / n_gt
    max_recall_achieved = np.max(recall_values)
    assert max_recall_achieved <= 1
    score_thresholds = np.interp(recall_threshold, recall_values, scores, right=0)

    # Set thresholds for unachieved recall values to nan to penalize AMOTA/AMOTP later.
    if isinstance(recall_threshold, np.ndarray):
        score_thresholds[recall_threshold > max_recall_achieved] = np.nan
    return score_thresholds


def calculate_TempLocAP_merge(
    data: dict[str, Any],
) -> Tuple[float, NDArrayFloat, NDArrayFloat]:
    """Calculates temporal localization average precision."""
    # Return result quickly if tracker or gt sequence is empty
    if data["num_tracker_dets"] == 0:
        if data["num_gt_dets"] == 0:
            precisions = np.array([1, 1])
            recalls = np.array([0, 1])
            TempLocAP = 1.0
            return TempLocAP, precisions, recalls
        else:
            precisions = np.array([0, 0])
            recalls = np.array([0, 1])
            TempLocAP = 0.0
            return TempLocAP, precisions, recalls
    if data["num_gt_dets"] == 0:
        precisions = np.array([0, 0])
        recalls = np.array([0, 1])
        TempLocAP = 0.0
        return TempLocAP, precisions, recalls

    TEMPORAL_IOU_THRESH = 0.5  # iou
    MATCHING_DIST_THRESH = 2.0  # m

    pred_tracks: dict[int, Any] = {}
    gt_tracks: dict[int, Any] = {}

    # Accumulate predicted and ground truth tracks from data
    for t in range(data["num_timesteps"]):
        for i, gt_id in enumerate(data["gt_ids"][t]):
            if gt_id not in gt_tracks:
                gt_tracks[gt_id] = {}
                gt_tracks[gt_id]["xy_pos"] = []
                gt_tracks[gt_id]["timestamps"] = []
                gt_tracks[gt_id]["category"] = data["gt_classes"][t][i]

            gt_tracks[gt_id]["xy_pos"].append(data["gt_dets"][t][i][:2])
            gt_tracks[gt_id]["timestamps"].append(t)

        for i, track_id in enumerate(data["tracker_ids"][t]):
            if track_id not in pred_tracks:
                pred_tracks[track_id] = {}
                pred_tracks[track_id]["confidence"] = data["tracker_confidences"][t][i]
                pred_tracks[track_id]["category"] = data["tracker_classes"][t][i]
                pred_tracks[track_id]["xy_pos"] = []
                pred_tracks[track_id]["timestamps"] = []

            pred_tracks[track_id]["xy_pos"].append(data["tracker_dets"][t][i][:2])
            pred_tracks[track_id]["timestamps"].append(t)

    # 1 to 1 match of predicted and ground truth tracks
    pred_ids_by_conf = sorted(
        pred_tracks.keys(),
        key=lambda key: pred_tracks[key]["confidence"],
        reverse=True,
    )

    # keys are gt_id, values are list of corresponding pred_ids
    matched_ids = {}

    # keys are track_ids, values are the iou of the timestamps of the matched predicted and ground truth timestamps
    unmatched_track_ids = []

    for track_id in pred_ids_by_conf:
        track_stats = pred_tracks[track_id]
        track_confidence = track_stats["confidence"]
        track_traj = track_stats["xy_pos"]
        track_timestamps = track_stats["timestamps"]

        max_similarity = 0
        best_match = None
        for gt_id, gt_stats in gt_tracks.items():
            if gt_stats["category"] != track_stats["category"]:
                continue

            gt_traj = gt_stats["xy_pos"]
            gt_timestamps = gt_stats["timestamps"]

            intersection = len(set(gt_timestamps).intersection((set(track_timestamps))))
            iol = intersection / len(track_timestamps)

            if iol < TEMPORAL_IOU_THRESH:
                continue

            distances = []
            for timestamp in track_timestamps:
                if timestamp in gt_timestamps:
                    distances.append(
                        np.linalg.norm(
                            track_traj[track_timestamps.index(timestamp)]
                            - gt_traj[gt_timestamps.index(timestamp)]
                        )
                    )

            similarity_score = iol * np.maximum(
                0, 1 - (np.mean(np.array(distances)) / MATCHING_DIST_THRESH)
            )

            if similarity_score > max_similarity:
                max_similarity = similarity_score
                best_match = gt_id

        if max_similarity > 0:
            if best_match not in matched_ids:
                matched_ids[best_match] = [track_id]
            else:
                matched_ids[best_match].append(track_id)
        else:
            unmatched_track_ids.append(track_id)

    merged_predictions = {}
    for gt_id, pred_ids in matched_ids.items():
        merged_traj = []
        merged_timestamps: list[int] = []
        merged_confidences = []
        merged_catetory = None

        for pred_id in pred_ids:
            track_timestamps = pred_tracks[pred_id]["timestamps"]
            track_trajectory = pred_tracks[pred_id]["xy_pos"]
            track_confidence = pred_tracks[pred_id]["confidence"]
            track_category = pred_tracks[pred_id]["category"]

            if len(merged_timestamps) == 0:
                merged_timestamps.extend(track_timestamps)
                merged_traj.extend(track_trajectory)
                merged_catetory = track_category
                merged_confidences.extend([track_confidence] * len(track_timestamps))
                continue

            for i, timestamp in enumerate(track_timestamps):
                if timestamp not in merged_timestamps:
                    insertion_index = 0
                    for merge_timestamp in merged_timestamps:
                        if merge_timestamp > timestamp:
                            insertion_index += 1

                    merged_timestamps.insert(insertion_index, timestamp)
                    merged_traj.insert(insertion_index, track_trajectory[i])
                    merged_confidences.insert(insertion_index, track_confidence)
                else:
                    insertion_index = merged_timestamps.index(timestamp)
                    if track_confidence > merged_confidences[insertion_index]:
                        merged_confidences[insertion_index] = track_confidence
                        merged_traj[insertion_index] = track_trajectory[i]

        merged_predictions[-gt_id - 1] = {
            "xy_pos": merged_traj,
            "timestamps": merged_timestamps,
            "confidence": np.mean(np.array(merged_confidences)),
            "category": merged_catetory,
        }

    for unmatched_track_id in unmatched_track_ids:
        merged_predictions.update({unmatched_track_id: pred_tracks[unmatched_track_id]})

    merged_ids_by_conf = sorted(
        merged_predictions.keys(),
        key=lambda key: merged_predictions[key]["confidence"],
        reverse=True,
    )
    matched_gt_ids = []

    # Compute precision and recall at all confidence thresholds.
    tp = np.zeros(len(merged_ids_by_conf))
    fp = np.zeros(len(merged_ids_by_conf))

    for i, track_id in enumerate(merged_ids_by_conf):
        track_stats = merged_predictions[track_id]
        track_confidence = track_stats["confidence"]
        track_traj = track_stats["xy_pos"]
        track_timestamps = track_stats["timestamps"]

        max_similarity = 0
        best_match = None
        for gt_id, gt_stats in gt_tracks.items():
            if (
                gt_id in matched_gt_ids
                or gt_stats["category"] != track_stats["category"]
            ):
                continue

            gt_traj = gt_stats["xy_pos"]
            gt_timestamps = gt_stats["timestamps"]

            intersection = len(set(gt_timestamps).intersection((set(track_timestamps))))
            union = len(set(gt_timestamps).union((set(track_timestamps))))
            iou = intersection / union

            if iou < TEMPORAL_IOU_THRESH:
                continue

            distances = []
            for timestamp in track_timestamps:
                if timestamp in gt_timestamps:
                    distances.append(
                        np.linalg.norm(
                            track_traj[track_timestamps.index(timestamp)]
                            - gt_traj[gt_timestamps.index(timestamp)]
                        )
                    )

            similarity_score = iou * np.maximum(
                0, 1 - (np.mean(np.array(distances)) / MATCHING_DIST_THRESH)
            )

            if similarity_score > max_similarity:
                max_similarity = similarity_score
                best_match = gt_id

        if max_similarity > 0:
            matched_gt_ids.append(best_match)
            tp[i] = 1
        else:
            fp[i] = 1

    tp = np.cumsum(tp)
    fp = np.cumsum(fp)

    recalls = tp / len(gt_tracks)
    precisions = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

    assert np.all(0 <= precisions) & np.all(precisions <= 1)
    TempLocAP = get_ap(recalls, precisions)

    return TempLocAP, precisions, recalls


def _get_envelope(precisions: NDArrayFloat) -> NDArrayFloat:
    """Compute the precision envelope.

    Args:
        precisions: A set of precision values <= 1

    Returns:
        The monotonically non-increasing precision values

    """
    for i in range(precisions.size - 1, 0, -1):
        precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
    return precisions


def get_ap(recalls: NDArrayFloat, precisions: NDArrayFloat) -> float:
    """Calculate average precision.

    Args:
        recalls: Array of recall values
        precisions: Array of precision values

    Returns:
        float: average precision.
    """
    # first append sentinel values at the end
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))

    # get envelope (maximum precision for each recall value)
    precisions = _get_envelope(precisions)

    # to calculate area under PR curve, look for points where X axis (recall) changes value
    i = np.where(recalls[1:] != recalls[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = float(np.sum((recalls[i + 1] - recalls[i]) * precisions[i + 1]))

    return ap


def plot_precision_recall_curve(
    recalls_list: list[NDArrayFloat],
    precisions_list: list[NDArrayFloat],
    ap_values: Union[list[float], None] = None,
    labels: Union[list[str], None] = None,
    colors: list[str] = ["blue", "green"],
    save_path: Union[str, None] = None,
) -> None:
    """Plot precision-recall curves for one or two sets of data.

    Args:
        recalls_list: List of recall arrays to plot
        precisions_list: List of precision arrays to plot
        ap_values: Optional list of AP values to display in the title
        labels: Optional list of labels for the legend
        colors: List of colors for the plots (default: blue and green)
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(8, 6))

    if not isinstance(recalls_list, list):
        recalls_list = [recalls_list]
    if not isinstance(precisions_list, list):
        precisions_list = [precisions_list]

    if labels is None:
        labels = [f"Curve {i+1}" for i in range(len(recalls_list))]

    for i, (recalls, precisions) in enumerate(zip(recalls_list, precisions_list)):
        color = colors[i % len(colors)]

        # Prepare data for plotting (add sentinel values)
        plot_recalls = np.concatenate(([0.0], recalls, [1.0]))
        plot_precisions = np.concatenate(([0.0], precisions, [0.0]))
        plot_precisions = _get_envelope(plot_precisions.copy())

        # Plot the curve
        plt.plot(
            plot_recalls,
            plot_precisions,
            color=color,
            linestyle="-",
            linewidth=2,
            label=labels[i],
        )
        plt.fill_between(plot_recalls, 0, plot_precisions, alpha=0.1, color=color)

    # Set title
    if ap_values:
        ap_text = ", ".join(
            [f"{label}: AP = {ap:.3f}" for label, ap in zip(labels, ap_values)]
        )
        plt.title(f"Precision-Recall Curves ({ap_text})", fontsize=16)
    else:
        plt.title("Precision-Recall Curves", fontsize=16)

    # Set labels and limits
    plt.xlabel("Recall", fontsize=14)
    plt.ylabel("Precision", fontsize=14)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(True)

    # Add legend if multiple curves
    if len(recalls_list) > 1:
        plt.legend(loc="lower left")

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def _xy_center_similarity(
    centers1: NDArrayFloat, centers2: NDArrayFloat, zero_distance: float
) -> NDArrayFloat:
    if centers1.size == 0 or centers2.size == 0:
        return np.zeros((len(centers1), len(centers2)))
    xy_dist = np.linalg.norm(
        centers1[:, np.newaxis, :2] - centers2[np.newaxis, :, :2], axis=2
    )
    sim = np.maximum(0, 1 - xy_dist / zero_distance)
    return cast(NDArrayFloat, sim)


def filter_max_dist(tracks: Any, max_range_m: int) -> Any:
    """Remove all tracks that are beyond the max_dist.

    Args:
        tracks: Dict[seq_id: List[frame]] Dictionary of tracks
        max_range_m: maximum distance from ego-vehicle

    Returns:
        tracks: Dict[seq_id: List[frame]] Dictionary of tracks.
    """
    frames = sm_utils.ungroup_frames(tracks)
    return sm_utils.group_frames(
        [
            sm_utils.index_array_values(
                frame,
                np.linalg.norm(
                    frame["translation_m"][:, :2]
                    - np.array(frame["ego_translation_m"])[:2],
                    axis=1,
                )
                <= max_range_m,
            )
            for frame in frames
        ]
    )


def load(pkl_path: Path) -> Any:
    """Loads a pkl file as a dict."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    return data


def yaw_to_quaternion3d(yaw: float) -> NDArrayFloat:
    """Convert a rotation angle in the xy plane (i.e. about the z axis) to a quaternion.

    Args:
        yaw: angle to rotate about the z-axis, representing an Euler angle, in radians

    Returns:
        array w/ quaternion coefficients (qw,qx,qy,qz) in scalar-first order, per Argoverse convention.
    """
    qx, qy, qz, qw = Rotation.from_euler(seq="z", angles=yaw, degrees=False).as_quat()
    return np.array([qw, qx, qy, qz])


def filter_drivable_area(tracks: Sequences, dataset_dir: Optional[str]) -> Sequences:
    """Convert the unified label format to a format that is easier to work with for forecasting evaluation.

    Args:
        tracks: Dictionary of tracks
        dataset_dir: Dataset root directory

    Returns:
        tracks: Dictionary of tracks.
    """
    if dataset_dir is None:
        return tracks

    log_prompt_pairs = list(tracks.keys())
    log_ids = [log_prompt_pair[0] for log_prompt_pair in log_prompt_pairs]

    log_id_to_avm, log_id_to_timestamped_poses = load_mapped_avm_and_egoposes(
        log_ids, Path(dataset_dir)
    )

    for i, log_id in enumerate(log_ids):
        avm = log_id_to_avm[log_id]

        for frame in tracks[log_prompt_pairs[i]]:
            timestamp_ns = frame["timestamp_ns"]
            city_SE3_ego = log_id_to_timestamped_poses[log_id][int(timestamp_ns)]
            translation_m = frame["translation_m"] - frame["ego_translation_m"]
            size = frame["size"]
            quat = np.array([yaw_to_quaternion3d(yaw) for yaw in frame["yaw"]])
            score = np.ones((translation_m.shape[0], 1))
            boxes = np.concatenate([translation_m, size, quat, score], axis=1)

            is_evaluated = compute_objects_in_roi_mask(boxes, city_SE3_ego, avm)

            frame["translation_m"] = frame["translation_m"][is_evaluated]
            frame["size"] = frame["size"][is_evaluated]
            frame["yaw"] = frame["yaw"][is_evaluated]
            frame["velocity_m_per_s"] = frame["velocity_m_per_s"][is_evaluated]
            frame["label"] = frame["label"][is_evaluated]
            frame["name"] = frame["name"][is_evaluated]
            frame["track_id"] = frame["track_id"][is_evaluated]

            if "score" in frame:
                frame["score"] = frame["score"][is_evaluated]

            if "detection_score" in frame:
                frame["detection_score"] = frame["detection_score"][is_evaluated]

            if "xy" in frame:
                frame["xy"] = frame["xy"][is_evaluated]

            if "xy_velocity" in frame:
                frame["xy_velocity"] = frame["xy_velocity"][is_evaluated]

            if "active" in frame:
                frame["active"] = frame["active"][is_evaluated]

            if "age" in frame:
                frame["age"] = frame["age"][is_evaluated]

    return tracks


def referred_full_tracks(sequences: Sequences) -> Sequences:
    """Expands the predicted partial tracks to the whole length of the track_id.

    Reconstructs a mining pkl file by propagating referred object labels across all instances
    of the same track_id and removing all other objects.

    Args:
        sequences: Sequences for either the labels or ground truth

    Returns:
        reconstructed_sequences: Dictionary containing the reconstructed sequences
    """
    reconstructed_sequences = {}

    # Process each sequence
    for seq_name, frames in sequences.items():
        # First pass: identify all track_ids that were ever referred objects
        referred_track_ids = set()
        for frame in frames:
            mask = frame["label"] == 0  # 0 is for REFERRED_OBJECT
            referred_track_ids.update(frame["track_id"][mask])

        # Second pass: reconstruct frames
        new_frames = []
        for frame in frames:
            # Create mask for referred track_ids
            mask = np.isin(frame["track_id"], list(referred_track_ids))

            # Create new frame with only referred objects
            new_frame = {
                "seq_id": frame["seq_id"],
                "timestamp_ns": frame["timestamp_ns"],
                "ego_translation_m": frame["ego_translation_m"],
                "description": frame["description"],
                "translation_m": frame["translation_m"][mask],
                "size": frame["size"][mask],
                "yaw": frame["yaw"][mask],
                "velocity_m_per_s": frame["velocity_m_per_s"][mask],
                "label": np.zeros(
                    mask.sum(), dtype=np.int32
                ),  # All are referred objects
                "name": np.array(["REFERRED_OBJECT"] * mask.sum(), dtype="<U31"),
                "track_id": frame["track_id"][mask],
            }

            # If score exists in the original frame (for predictions), include it
            if "score" in frame:
                new_frame["score"] = frame["score"][mask]

            new_frames.append(new_frame)

        reconstructed_sequences[seq_name] = new_frames

    return reconstructed_sequences


def evaluate_mining(
    track_predictions: Sequences, labels: Sequences, output_dir: str
) -> tuple[float, float]:
    """Calculates the F1 score.

    F1 is a binary classification metric. A true postive when both
    the ground-truth and predictions sequences contains no tracks
    or both contain at least one track corresponding to the prediction.
    """
    gt_class = np.zeros(len(labels), dtype=np.int64)
    pred_class = np.zeros(len(labels), dtype=np.int64)

    for i, description in enumerate(labels.keys()):

        for frame in labels[description]:
            if len(frame["label"]) > 0 and 0 in frame["label"]:
                gt_class[i] = 1
                break

        for frame in track_predictions[description]:
            if len(frame["label"]) > 0 and 0 in frame["label"]:
                pred_class[i] = 1
                break

    tp = np.sum(gt_class & pred_class)
    fp = np.sum(~gt_class & pred_class)
    fn = np.sum(gt_class & ~pred_class)

    f1_score = float(2 * tp / (2 * tp + fp + fn))

    print(f"GT scenario matches: {gt_class}")
    print(f"Predicted scenario matches: {pred_class}")
    print(f"F1: {f1_score}")

    _plot_confusion_matrix(gt_class, pred_class, output_dir)

    num_correct = 0
    for i in range(len(gt_class)):
        if gt_class[i] == pred_class[i]:
            num_correct += 1

    acc = num_correct / len(labels)

    return f1_score, acc


def _relabel_seq_ids(data: Sequences) -> Sequences:
    """Turns the (log_id, prompt) tuple format into a string for HOTA summarization."""
    new_data = {}

    for seq_id, frames in data.items():
        if isinstance(seq_id, tuple):
            new_seq_id = str(seq_id)
            new_data[new_seq_id] = frames
        else:
            new_data[seq_id] = frames

    for seq_id, frames in new_data.items():
        for frame in frames:
            if "seq_id" in frame and isinstance(frame["seq_id"], tuple):
                frame["seq_id"] = str(frame["seq_id"])

    return new_data


def evaluate(
    track_predictions: Sequences,
    labels: Sequences,
    objective_metric: str,
    max_range_m: int,
    dataset_dir: Any,
    out: str,
) -> tuple[float, float, float, float]:
    """Run scenario mining evaluation on the supplied prediction and label pkl files.

    Args:
        track_predictions: Prediction sequences.
        labels: Ground truth sequences.
        objective_metric: Metric to optimize.
        max_range_m: Maximum evaluation range.
        dataset_dir: Path to dataset. Required for ROI pruning.
        out: Output path.

    Returns:
        F1_score: The F1 score for if the scenario matches the description
        full_track_HOTA: The tracking metric for the full track of any objects that the description ever applies to.
        partial_track_HOTA: The tracking metric for the tracks that contain only the timestamps for which the description applies.
        tlap: Temporal localization average precision calculates how well predictions are temporally localized,
            with some built in give for ambiguous annotations
    """
    output_dir = ""
    if out:
        output_dir = out + "/partial_tracks"
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    res, partial_track_metrics, TempLocAP, f1_score = evaluate_scenario_mining(
        track_predictions,
        labels,
        objective_metric=objective_metric,
        max_range_m=max_range_m,
        dataset_dir=dataset_dir,
        out=output_dir,
    )

    full_track_preds = referred_full_tracks(track_predictions)
    full_track_labels = referred_full_tracks(labels)

    output_dir = ""
    if out:
        output_dir = out + "/full_tracks"
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    _, full_track_metrics, _, _ = evaluate_scenario_mining(
        full_track_preds,
        full_track_labels,
        objective_metric=objective_metric,
        max_range_m=max_range_m,
        dataset_dir=dataset_dir,
        out=output_dir,
        full_tracks=True,
    )

    full_track_hota = full_track_metrics["REFERRED_OBJECT"]
    partial_track_hota = partial_track_metrics["REFERRED_OBJECT"]

    print(
        f"F1: {f1_score}, HOTA: {partial_track_hota}, HOTA_full: {full_track_hota}, TLAP: {TempLocAP}"
    )
    return f1_score, full_track_hota, partial_track_hota, TempLocAP


def evaluate_scenario_mining(
    track_predictions: Sequences,
    labels: Sequences,
    objective_metric: str,
    max_range_m: int,
    dataset_dir: Any,
    out: str,
    full_tracks: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any], float, float]:
    """Run evaluation.

    Args:
        track_predictions: Dictionary of tracks.
        labels: Dictionary of labels.
        objective_metric: Metric to optimize.
        max_range_m: Maximum evaluation range.
        dataset_dir: Path to dataset. Required for ROI pruning.
        out: Output path.
        full_tracks: Whether the supplied labels are for the full track of any
        object that ever corresponds to the description or only for the timestamps
        when the description applies.

    Returns:
        Dictionary of per-category metrics.
    """
    classes = list(constants.AV2_CATEGORIES)

    labels = filter_max_dist(labels, max_range_m)
    track_predictions = filter_max_dist(track_predictions, max_range_m)

    if dataset_dir is not None:
        labels = filter_drivable_area(labels, dataset_dir)
        track_predictions = filter_drivable_area(track_predictions, dataset_dir)

    track_predictions = _relabel_seq_ids(track_predictions)
    labels = _relabel_seq_ids(labels)

    score_thresholds, tuned_metric_values, mean_metric_values = _tune_score_thresholds(
        labels,
        track_predictions,
        objective_metric,
        classes,
        num_thresholds=3,
        match_distance_m=2,
    )
    filtered_track_predictions = sm_utils.filter_by_class_thresholds(
        track_predictions, score_thresholds
    )
    res, tlap = evaluate_tracking(
        labels,
        filtered_track_predictions,
        classes,
        tracker_name="TRACKER",
        output_dir=out,
    )

    if not full_tracks:
        f1_score, acc = evaluate_mining(filtered_track_predictions, labels, out)
        return res, tuned_metric_values, tlap, f1_score

    return res, tuned_metric_values, tlap, 0


@click.command()
@click.option("--predictions", required=True, help="Predictions PKL file")
@click.option("--ground_truth", required=True, help="Ground Truth PKL file")
@click.option(
    "--max_range_m",
    default=50,
    type=int,
    help="Evaluate objects within distance of ego vehicle",
)
@click.option(
    "--dataset_dir",
    default=None,
    help="Path to dataset split (e.g. /data/Sensor/val). Required for ROI pruning",
)
@click.option("--objective_metric", default="HOTA", help="Choices: HOTA, MOTA")
@click.option("--out", required=True, help="Output JSON file")
def runner(
    predictions: str,
    ground_truth: str,
    max_range_m: int,
    dataset_dir: Any,
    objective_metric: str,
    out: str,
) -> None:
    """Standalone evaluation function."""
    track_predictions = pickle.load(open(predictions, "rb"))
    labels = pickle.load(open(ground_truth, "rb"))

    _, _, mean_metric_values, _ = evaluate_scenario_mining(
        track_predictions, labels, objective_metric, max_range_m, dataset_dir, out
    )

    pprint(mean_metric_values)

    with open(out, "w") as f:
        json.dump(mean_metric_values, f, indent=4)


if __name__ == "__main__":
    runner()
