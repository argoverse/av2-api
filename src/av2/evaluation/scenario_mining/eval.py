"""Argoverse 2 scenario mining evaluation.

Evaluation Metrics:
    HOTA: see https://arxiv.org/abs/2009.07736
    log-level balanced accuracy
    timestamp-level balanced accuracy
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
from copy import deepcopy
import pickle
import click
from packaging import version
import numpy as np
from collections import defaultdict
import json

# Handling deprecated NumPy types for compatibility with TrackEval
if version.parse(np.__version__) >= version.parse("1.24.0"):
    # Use type annotations to avoid linter errors
    numpy_float = np.float32
    numpy_int = np.int32
    numpy_bool = np.bool_  # Use np.bool_ instead of trying to assign to bool

    # Hotfix numpy namespace for backward compatibility
    setattr(np, "float", numpy_float)
    setattr(np, "int", numpy_int)
    setattr(np, "bool", numpy_bool)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from av2.map.map_api import ArgoverseStaticMap, RasterLayerType
from av2.utils.typing import NDArrayBool
from av2.structures.cuboid import Cuboid, CuboidList
from av2.evaluation.detection.utils import load_mapped_avm_and_egoposes

from av2.evaluation.tracking.eval import (
    yaw_to_quaternion3d,
    filter_max_dist,
    _tune_score_thresholds,
)
from av2.evaluation.tracking.utils import filter_by_class_thresholds

from av2.evaluation.scenario_mining import (
    SEQ_ID_LOG_INDICES,
    SEQ_ID_PROMPT_INDICES,
    SCENARIO_MINING_CATEGORIES,
)
from av2.utils.typing import NDArrayFloat
from av2.evaluation.typing import Sequences

REFERRED_OBJECT = SCENARIO_MINING_CATEGORIES.index("REFERRED_OBJECT")


def _plot_confusion_matrix(
    tp: int,
    fn: int,
    fp: int,
    tn: int,
    title: str,
    output_dir: Union[str, None] = None,
) -> None:
    """Plots the confusion matrix for scenario mining.

    A true label indicates that the scenario matches the prompt/description.
    A false label indicates the scenario does not match the prompt/description.

    Args:
        tp: The number of classification true positives.
        fn: The number of classification false negatives.
        fp: The number of classification false positives.
        tn: The number of classification true negatives.
        title: A string description indicating the granularity of classification.
        output_dir: The directory to save the confusion matrix plot.

    Returns:
        None
    """
    if output_dir is None:
        return

    # Create confusion matrix (2x2 for binary classification)
    cm = np.array([[tp, fn], [fp, tn]])

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
    ax.set_title(f"{title}-level confusion matrix")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Positive", "Negative"])
    ax.set_yticklabels(["Positive", "Negative"])
    plt.setp(ax.get_yticklabels(), rotation=90, ha="right")

    # Display the plot
    if output_dir:
        plt.savefig(output_dir + f"/{title}_level_confusion_matrix.png")
    plt.close()


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
    log_ids = [seq_id[SEQ_ID_LOG_INDICES] for seq_id in tracks.keys()]

    log_id_to_avm, log_id_to_timestamped_poses = load_mapped_avm_and_egoposes(
        log_ids, Path(dataset_dir)
    )

    for i, log_id in enumerate(log_ids):
        avm = log_id_to_avm[log_id]

        for frame in tracks[log_prompt_pairs[i]]:
            translation_m = frame["translation_m"]
            if translation_m.shape[0] == 0:
                continue
            size = frame["size"]
            quat = np.array([yaw_to_quaternion3d(yaw) for yaw in frame["yaw"]])
            score = np.ones((translation_m.shape[0], 1))
            boxes = np.concatenate([translation_m, size, quat, score], axis=1)

            is_evaluated = compute_objects_in_roi_mask(boxes, avm)

            frame["translation_m"] = frame["translation_m"][is_evaluated]
            frame["size"] = frame["size"][is_evaluated]
            frame["yaw"] = frame["yaw"][is_evaluated]
            frame["label"] = frame["label"][is_evaluated]
            frame["name"] = frame["name"][is_evaluated]
            frame["track_id"] = frame["track_id"][is_evaluated]

            if "score" in frame:
                frame["score"] = frame["score"][is_evaluated]
            if "velocity_m_per_s" in frame:
                frame["velocity_m_per_s"] = frame["velocity_m_per_s"][is_evaluated]

    return tracks


def compute_objects_in_roi_mask(
    cuboids_city: NDArrayFloat, avm: ArgoverseStaticMap
) -> NDArrayBool:
    """Compute the evaluated cuboids mask based off whether _any_ of their vertices fall into the ROI.

    Args:
        cuboids_city: (N,10) Array of cuboid parameters corresponding to `ORDERED_CUBOID_COL_NAMES`.
        city_SE3_ego: Egovehicle pose in the city reference frame.
        avm: Argoverse map object.

    Returns:
        (N,) Boolean mask indicating which cuboids will be evaluated.
    """
    is_within_roi: NDArrayBool
    if len(cuboids_city) == 0:
        is_within_roi = np.zeros((0,), dtype=bool)
        return is_within_roi
    cuboid_list_city: CuboidList = CuboidList(
        [Cuboid.from_numpy(params) for params in cuboids_city]
    )
    cuboid_list_vertices_m_city = cuboid_list_city.vertices_m

    is_within_roi = avm.get_raster_layer_points_boolean(
        cuboid_list_vertices_m_city.reshape(-1, 3)[..., :2], RasterLayerType.ROI
    )
    is_within_roi = is_within_roi.reshape(-1, 8)
    is_within_roi = is_within_roi.any(axis=1)
    return is_within_roi


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
            mask = frame["label"] == REFERRED_OBJECT
            referred_track_ids.update(frame["track_id"][mask])

        # Second pass: reconstruct frames
        new_frames = []
        for frame in frames:
            # Create mask for referred track_ids
            mask = np.isin(frame["track_id"], list(referred_track_ids))
            new_frame = deepcopy(frame)

            new_names = np.zeros(len(frame["name"]), dtype="<U31")
            for i in range(len(frame["name"])):
                if mask[i]:
                    new_names[i] = "REFERRED_OBJECT"
                else:
                    new_names[i] = frame["name"][i]

            new_frame["name"] = new_names
            new_frame["label"] = np.where(mask, REFERRED_OBJECT, frame["label"])
            new_frames.append(new_frame)

        reconstructed_sequences[seq_name] = new_frames

    return reconstructed_sequences


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 1.0


def compute_temporal_metrics(
    track_predictions: Sequences, labels: Sequences, output_dir: str
) -> tuple[dict[str, float], dict[str, float]]:
    """Calculates the balanced accuracy for log and timestamp retrival.

    Balanced Accuracy (BA) is a binary classification metric. A true postive when both
    the ground-truth and prediction sequence/timestamp contain no tracks
    or both contain at least one track corresponding to the prompt. Each prompt
    is evaluated as its own class.

    The formula for BA is: (TP/(TP+FP) + TN/(TN+FN))/2

    Only non-ambiguous timestamps are used for calculating temporal metrics. Objects
    can be procedurally annotated out to 200 meters but are only verified within 50 meters.
    Therefore, all timestamps with referred objects exclusively past 50 meters are labeled ambiguous.

    Args:
        track_predictions: Prediction sequences.
        labels: Ground truth sequences.
        output_dir: The directory to save the plotted confusion matrices.

    Returns:
        scenario_ba_by_class: The balanced accuracy where each log prompt pair counts as a prediction to evaluate.
        timestamp_ba_by_class: The balanced accuracy where each timestamp counts as a prediction to evaluate.
    """
    prompts = []

    timestamp_tp: defaultdict[str, int] = defaultdict(int)
    timestamp_fp: defaultdict[str, int] = defaultdict(int)
    timestamp_tn: defaultdict[str, int] = defaultdict(int)
    timestamp_fn: defaultdict[str, int] = defaultdict(int)

    scenario_tp: defaultdict[str, int] = defaultdict(int)
    scenario_fp: defaultdict[str, int] = defaultdict(int)
    scenario_tn: defaultdict[str, int] = defaultdict(int)
    scenario_fn: defaultdict[str, int] = defaultdict(int)

    for seq_id in labels.keys():

        prompt = seq_id[SEQ_ID_PROMPT_INDICES]
        prompts.append(prompt)

        timestamp_gt = np.zeros(len(labels[seq_id]), dtype=bool)
        timestamp_pred = np.zeros(len(labels[seq_id]), dtype=bool)
        ambiguity_mask = np.zeros(len(labels[seq_id]), dtype=bool)

        scenario_gt = False
        scenario_pred = False

        for j, frame in enumerate(labels[seq_id]):
            if "is_positive" in frame:
                if frame["is_positive"] is None:
                    ambiguity_mask[j] = 1
                elif frame["is_positive"]:
                    timestamp_gt[j] = True
                    scenario_gt = True

            # Backward compatibility with previous versions of the dataset
            elif len(frame["label"]) > 0 and REFERRED_OBJECT in frame["label"]:
                timestamp_gt[j] = True
                scenario_gt = True

        for j, frame in enumerate(track_predictions[seq_id]):
            if "is_positive" in frame and frame["is_positive"]:
                timestamp_pred[j] = True
                scenario_pred = True
            elif (
                "label" in frame
                and len(frame["label"]) > 0
                and REFERRED_OBJECT in frame["label"]
            ):
                timestamp_pred[j] = True
                scenario_pred = True

        if np.all(ambiguity_mask):
            continue

        non_ambiguous = ~ambiguity_mask
        timestamp_tp[prompt] += int(
            np.sum(timestamp_gt[non_ambiguous] & timestamp_pred[non_ambiguous])
        )
        timestamp_fp[prompt] += int(
            np.sum(~timestamp_gt[non_ambiguous] & timestamp_pred[non_ambiguous])
        )
        timestamp_fn[prompt] += int(
            np.sum(timestamp_gt[non_ambiguous] & ~timestamp_pred[non_ambiguous])
        )
        timestamp_tn[prompt] += int(
            np.sum(~timestamp_gt[non_ambiguous] & ~timestamp_pred[non_ambiguous])
        )

        if scenario_pred and scenario_gt:
            scenario_tp[prompt] += 1
        elif scenario_pred and not scenario_gt:
            scenario_fp[prompt] += 1
        elif not scenario_pred and scenario_gt:
            scenario_fn[prompt] += 1
        elif not scenario_pred and not scenario_gt:
            scenario_tn[prompt] += 1

    # Balanced Accuracy: https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers
    # (TPR + TNR) / 2
    scenario_ba_by_class = {}
    timestamp_ba_by_class = {}

    for prompt in prompts:

        scenario_tpr = _safe_rate(
            scenario_tp[prompt], scenario_tp[prompt] + scenario_fn[prompt]
        )
        scenario_tnr = _safe_rate(
            scenario_tn[prompt], scenario_tn[prompt] + scenario_fp[prompt]
        )
        timestamp_tpr = _safe_rate(
            timestamp_tp[prompt], timestamp_tp[prompt] + timestamp_fn[prompt]
        )
        timestamp_tnr = _safe_rate(
            timestamp_tn[prompt], timestamp_tn[prompt] + timestamp_fp[prompt]
        )

        scenario_ba_by_class[prompt] = (scenario_tpr + scenario_tnr) / 2
        timestamp_ba_by_class[prompt] = (timestamp_tpr + timestamp_tnr) / 2

    if output_dir:

        _plot_confusion_matrix(
            np.sum(list(scenario_tp.values())),
            np.sum(list(scenario_fn.values())),
            np.sum(list(scenario_fp.values())),
            np.sum(list(scenario_tn.values())),
            title="scenario",
            output_dir=output_dir,
        )
        _plot_confusion_matrix(
            np.sum(list(timestamp_tp.values())),
            np.sum(list(timestamp_fn.values())),
            np.sum(list(timestamp_fp.values())),
            np.sum(list(timestamp_tn.values())),
            title="timestamp",
            output_dir=output_dir,
        )

    return scenario_ba_by_class, timestamp_ba_by_class


def relabel_sequence_ids(sequences: Sequences) -> Sequences:
    """Turns the (log_id, prompt) tuple format into a string for HOTA summarization.

    Args:
        sequences: The 'sequences' where each top level key is a tuple of (log_id, prompt)

    Returns:
        Sequences where each top level key is a string of '(log_id, prompt)'
    """
    relabeled_sequences = {}

    # Turn top level seq_id into string
    for seq_id, frames in sequences.items():
        if isinstance(seq_id, tuple):
            new_seq_id = str(seq_id)
            relabeled_sequences[new_seq_id] = frames
        else:
            relabeled_sequences[seq_id] = frames

    for seq_id_str, frames in relabeled_sequences.items():
        for frame in frames:
            frame["seq_id"] = seq_id_str

    return relabeled_sequences


def classify_referred_objects(sequences: Sequences) -> Sequences:
    """Classifies all referred objects as the prompt in the sequence id.

    Args:
        sequences: The 'sequences' where each top level key is a string containing '(log_id, prompt)'

    Returns:
        Sequences where all referred objects are given the class name corresponding to the prompt
    """
    # Turn frame seq_id into string and only classify referred objects
    # seq_id str is 39 character prefix f'({log_id}, ' + prompt + 1 character suffix ')'
    for seq_id_str, frames in sequences.items():
        prompt = seq_id_str[SEQ_ID_PROMPT_INDICES]
        for frame in frames:

            frame["seq_id"] = seq_id_str
            referred_object_mask = frame["label"] == REFERRED_OBJECT

            frame["name"] = np.array([prompt] * np.sum(referred_object_mask))
            frame["translation_m"] = frame["translation_m"][referred_object_mask]
            frame["size"] = frame["size"][referred_object_mask]
            frame["yaw"] = frame["yaw"][referred_object_mask]
            frame["label"] = frame["label"][referred_object_mask]
            frame["track_id"] = frame["track_id"][referred_object_mask]

            if "score" in frame:
                frame["score"] = frame["score"][referred_object_mask]
            if "velocity_m_per_s" in frame:
                frame["velocity_m_per_s"] = frame["velocity_m_per_s"][
                    referred_object_mask
                ]

    return sequences


def evaluate(
    scenario_predictions: Sequences,
    labels: Sequences,
    objective_metric: str,
    max_range_m: int,
    dataset_dir: str,
    out: str,
) -> tuple[float, float, float, float]:
    """Run scenario mining evaluation on the supplied prediction and label pkl files.

    If tracks are not submitted within the scenario_predictions dictionary, only temporal metrics will be computed.

    Args:
        scenario_predictions: Prediction sequences.
        labels: Ground truth sequences.
        objective_metric: Metric to optimize.
        max_range_m: Maximum evaluation range.
        dataset_dir: Path to dataset. Required for ROI pruning.
        out: Output path. May be None.

    Returns:
        hota_temporal: The tracking metric for the tracks that contain only the timestamps for which the description applies.
        hota_track: The tracking metric for the full track of any objects that the description ever applies to.
        timestamp_ba: A retrieval/classification metric for determining if each timestamp contains any instance of the prompt.
        scenario_ba: A retrieval/classification metric for determining if each data log contains any instance of the prompt.
    """
    # Submissions may have sequence ids of form (log_id, prompt) or f"({log_id}, {prompt})"
    labels = relabel_sequence_ids(labels)
    scenario_predictions = relabel_sequence_ids(scenario_predictions)

    if out:
        Path(out).mkdir(exist_ok=True, parents=True)

    contains_tracking = False
    for frames in scenario_predictions.values():
        for frame in frames:
            if "is_positive" not in frame:
                contains_tracking = True
                break
            elif (
                "track_id" in frame
                and isinstance(frame["track_id"], np.ndarray)
                and len(frame["track_id"] > 0)
            ):
                contains_tracking = True
                break

        if contains_tracking:
            break

    hota_temporal_by_class: dict[str, float] = {}
    hota_track_by_class: dict[str, float] = {}

    if contains_tracking:

        labels = filter_max_dist(labels, max_range_m)
        scenario_predictions = filter_max_dist(scenario_predictions, max_range_m)

        if dataset_dir is not None:
            labels = filter_drivable_area(labels, dataset_dir)
            scenario_predictions = filter_drivable_area(
                scenario_predictions, dataset_dir
            )

        hota_temporal_by_class, scenario_ba_by_class, timestamp_ba_by_class = (
            evaluate_scenario_mining(
                scenario_predictions,
                labels,
                objective_metric=objective_metric,
                out=out,
                full_tracks=False,
            )
        )
        hota_track_by_class, _, _ = evaluate_scenario_mining(
            scenario_predictions,
            labels,
            objective_metric=objective_metric,
            out=out,
            full_tracks=True,
        )

        hota_temporal = float(np.mean(np.array(list(hota_temporal_by_class.values()))))
        hota_track = float(np.mean(np.array(list(hota_track_by_class.values()))))
        timestamp_ba = float(np.mean(np.array(list(timestamp_ba_by_class.values()))))
        scenario_ba = float(np.mean(np.array(list(scenario_ba_by_class.values()))))
    else:

        scenario_predictions = classify_referred_objects(scenario_predictions)
        labels = classify_referred_objects(labels)

        scenario_ba_by_class, timestamp_ba_by_class = compute_temporal_metrics(
            scenario_predictions, labels, out
        )
        scenario_ba = float(np.mean(np.array(list(scenario_ba_by_class.values()))))
        timestamp_ba = float(np.mean(np.array(list(timestamp_ba_by_class.values()))))

        hota_temporal = 0.0
        hota_track = 0.0

    if out:

        temporal_metrics = {
            "timestamp_balanced_accuracy_class_avg": timestamp_ba,
            "scenario_balanced_accuracy_class_avg": scenario_ba,
            "timestamp_balanced_accuracy_by_class": timestamp_ba_by_class,
            "scenario_balanced_accuracy_by_class": scenario_ba_by_class,
        }

        with open(Path(out) / "temporal_metrics.json", "w") as file:
            json.dump(temporal_metrics, file, indent=4)

        if contains_tracking:

            spatiotemporal_metrics = {
                "hota_temporal_class_avg": hota_temporal,
                "hota_track_class_avg": hota_track,
                "hota_temporal_by_class": hota_temporal_by_class,
                "hota_track_by_class": hota_track_by_class,
            }
            with open(Path(out) / "spatiotemporal_metrics.json", "w") as file:
                json.dump(spatiotemporal_metrics, file, indent=4)

    return (
        hota_temporal,
        hota_track,
        timestamp_ba,
        scenario_ba,
    )


def evaluate_scenario_mining(
    track_predictions: Sequences,
    track_labels: Sequences,
    objective_metric: str,
    out: str,
    full_tracks: bool = False,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Run evaluation.

    Args:
        track_predictions: Dictionary of tracks.
        track_labels: Dictionary of labels.
        objective_metric: Metric to optimize.
        out: Output path.
        full_tracks: Whether the supplied labels are for the full track of any
        object that ever corresponds to the description or only for the timestamps
        when the description applies.

    Returns:
        referred_hota_by_class: The HOTA tracking metric applied to all objects with the category REFERRED_OBJECT
        scenario_ba_by_class: A retrieval/classification metric for determining if each data log contains any instance of the prompt.
        timestamp_ba_by_class: A retrieval/classification metric for determining if each timestamp contains any instance of the prompt.
    """
    scenario_predictions = deepcopy(track_predictions)
    labels = deepcopy(track_labels)

    if full_tracks:
        scenario_predictions = referred_full_tracks(scenario_predictions)
        labels = referred_full_tracks(labels)

    classes = list(set([seq_id[SEQ_ID_PROMPT_INDICES] for seq_id in labels.keys()]))
    scenario_predictions = classify_referred_objects(scenario_predictions)
    labels = classify_referred_objects(labels)

    score_thresholds, optimal_metric_by_class, _ = _tune_score_thresholds(
        labels,
        scenario_predictions,
        objective_metric,
        classes,
        num_thresholds=10,
        match_distance_m=2,
    )
    referred_hota_by_class = {
        prompt: float(metric_value)
        for prompt, metric_value in optimal_metric_by_class.items()
    }

    filtered_scenario_predictions = filter_by_class_thresholds(
        scenario_predictions, score_thresholds
    )
    scenario_ba_by_class, timestamp_ba_by_class = compute_temporal_metrics(
        filtered_scenario_predictions, labels, out
    )
    return referred_hota_by_class, scenario_ba_by_class, timestamp_ba_by_class


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
    with open(predictions, "rb") as f:
        track_predictions = pickle.load(f)
    with open(ground_truth, "rb") as f:
        labels = pickle.load(f)

    partial_track_hota, full_track_hota, timestamp_ba, scenario_ba = evaluate(
        track_predictions, labels, objective_metric, max_range_m, dataset_dir, out
    )

    print(f"HOTA-Temporal: {partial_track_hota:.2f}")
    print(f"HOTA-Track: {full_track_hota:.2f}")
    print(f"Timestamp-level Balanced Accuracy: {timestamp_ba:.2f}")
    print(f"Log-level Balanced Accuracy: {scenario_ba:.2f}")


if __name__ == "__main__":
    runner()
