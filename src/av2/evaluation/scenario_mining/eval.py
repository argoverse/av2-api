"""Argoverse 2 scenario mining evaluation.

Evaluation Metrics:
    HOTA: see https://arxiv.org/abs/2009.07736
    scenario-level F1: see https://jivp-eurasipjournals.springeropen.com/articles/10.1155/2008/246309
    timestamp-level F1: see https://arxiv.org/abs/2008.08063
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
from copy import deepcopy
import pickle
import click
from packaging import version
import numpy as np

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

import matplotlib.pyplot as plt

from av2.map.map_api import ArgoverseStaticMap, RasterLayerType
from av2.utils.typing import NDArrayBool
from av2.structures.cuboid import Cuboid, CuboidList
from av2.evaluation.detection.utils import load_mapped_avm_and_egoposes
from av2.evaluation.tracking.eval import (
    yaw_to_quaternion3d,
    filter_max_dist,
    _tune_score_thresholds,
    evaluate_tracking,
)
from av2.evaluation.scenario_mining import AV2_CATEGORIES
from av2.evaluation.tracking import utils as sm_utils
from av2.utils.typing import NDArrayFloat
from av2.evaluation.typing import Sequences


def _plot_confusion_matrix(
    tp: int,
    fn: int,
    fp: int,
    tn: int,
    title: str = "scenario",
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
    ax.set_title(f"Prompt Occurences at {title}-level")
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
    log_ids = [log_prompt_pair[0] for log_prompt_pair in log_prompt_pairs]

    log_id_to_avm, log_id_to_timestamped_poses = load_mapped_avm_and_egoposes(
        log_ids, Path(dataset_dir)
    )

    for i, log_id in enumerate(log_ids):
        avm = log_id_to_avm[log_id]

        for frame in tracks[log_prompt_pairs[i]]:
            translation_m = frame["translation_m"]
            size = frame["size"]
            quat = np.array([yaw_to_quaternion3d(yaw) for yaw in frame["yaw"]])
            score = np.ones((translation_m.shape[0], 1))
            boxes = np.concatenate([translation_m, size, quat, score], axis=1)

            is_evaluated = compute_objects_in_roi_mask(boxes, avm)

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
            mask = frame["label"] == 0  # 0 is for REFERRED_OBJECT
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
            new_frames.append(new_frame)

        reconstructed_sequences[seq_name] = new_frames

    return reconstructed_sequences


def compute_temporal_metrics(
    track_predictions: Sequences, labels: Sequences, output_dir: str
) -> tuple[float, float]:
    """Calculates the F1 score.

    F1 is a binary classification metric. A true postive when both
    the ground-truth and prediction sequence/timestamp contain no tracks
    or both contain at least one track corresponding to the prompt.

    Args:
        track_predictions: Prediction sequences.
        labels: Ground truth sequences.
        output_dir: The directory to save the plotted confusion matrices.

    Returns:
        timestamp_f1: The F1 score where each timestamp counts as a prediction to evaluate.
        scenario_f1: The F1 score where each log-prompt pair counts as a prediction to evaluate.


    """
    scenario_gt = np.zeros(len(labels), dtype=bool)
    scenario_pred = np.zeros(len(labels), dtype=bool)

    timestamp_tp, timestamp_fp, timestamp_fn, timestamp_tn = 0, 0, 0, 0

    for i, description in enumerate(labels.keys()):

        timestamp_gt = np.zeros(len(labels[description]), dtype=bool)
        timestamp_pred = np.zeros(len(labels[description]), dtype=bool)

        for j, frame in enumerate(labels[description]):
            if len(frame["label"]) > 0 and 0 in frame["label"]:
                timestamp_gt[j] = True
                scenario_gt[i] = True

        for j, frame in enumerate(track_predictions[description]):
            if len(frame["label"]) > 0 and 0 in frame["label"]:
                timestamp_pred[j] = True
                scenario_pred[i] = True

        timestamp_tp += np.sum(timestamp_gt & timestamp_pred)
        timestamp_fp += np.sum(~timestamp_gt & timestamp_pred)
        timestamp_fn += np.sum(timestamp_gt & ~timestamp_pred)
        timestamp_tn += np.sum(~timestamp_gt & ~timestamp_pred)

    scenario_tp = np.sum(scenario_gt & scenario_pred)
    scenario_fp = np.sum(~scenario_gt & scenario_pred)
    scenario_fn = np.sum(scenario_gt & ~scenario_pred)
    scenario_tn = np.sum(~scenario_gt & ~scenario_pred)

    scenario_f1 = float(2 * scenario_tp / (2 * scenario_tp + scenario_fp + scenario_fn))
    timestamp_f1 = float(
        2 * timestamp_tp / (2 * timestamp_tp + timestamp_fp + timestamp_fn)
    )

    _plot_confusion_matrix(
        scenario_tp,
        scenario_fn,
        scenario_fp,
        scenario_tn,
        title="scenario",
        output_dir=output_dir,
    )
    _plot_confusion_matrix(
        timestamp_tp,
        timestamp_fn,
        timestamp_fp,
        timestamp_tn,
        title="timestamp",
        output_dir=output_dir,
    )

    return scenario_f1, timestamp_f1


def _relabel_seq_ids(sequences: Sequences) -> Sequences:
    """Turns the (log_id, prompt) tuple format into a string for HOTA summarization.

    Args:
        sequences: The 'sequences' where each top level key is a tuple of (log_id, prompt)

    Returns:
        Sequences where each top level key is a string of '(log_id, prompt)'
    """
    new_data = {}

    for seq_id, frames in sequences.items():
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
        partial_track_HOTA: The tracking metric for the tracks that contain only the timestamps for which the description applies.
        full_track_HOTA: The tracking metric for the full track of any objects that the description ever applies to.
        timestamp_f1: A retrieval/classification metric for determining if each timestamp contains any instance of the prompt.
        scenario_f1: A retrieval/classification metric for determining if each data log contains any instance of the prompt.
    """
    output_dir = out + "/partial_tracks"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    partial_track_hota, timestamp_f1, scenario_f1 = evaluate_scenario_mining(
        track_predictions,
        labels,
        objective_metric=objective_metric,
        max_range_m=max_range_m,
        dataset_dir=dataset_dir,
        out=output_dir,
    )

    full_track_preds = referred_full_tracks(track_predictions)
    full_track_labels = referred_full_tracks(labels)

    output_dir = out + "/full_tracks"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    full_track_hota, _, _ = evaluate_scenario_mining(
        full_track_preds,
        full_track_labels,
        objective_metric=objective_metric,
        max_range_m=max_range_m,
        dataset_dir=dataset_dir,
        out=output_dir,
        full_tracks=True,
    )

    return (
        partial_track_hota,
        full_track_hota,
        timestamp_f1,
        scenario_f1,
    )


def evaluate_scenario_mining(
    track_predictions: Sequences,
    labels: Sequences,
    objective_metric: str,
    max_range_m: int,
    dataset_dir: Any,
    out: str,
    full_tracks: bool = False,
) -> Tuple[float, float, float]:
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
        referred_hota: The HOTA tracking metric applied to all objects with the category REFERRED_OBJECT
        timestamp_f1: A retrieval/classification metric for determining if each timestamp contains any instance of the prompt.
        scenario_f1: A retrieval/classification metric for determining if each data log contains any instance of the prompt.
    """
    classes = list(AV2_CATEGORIES)

    labels = filter_max_dist(labels, max_range_m)
    track_predictions = filter_max_dist(track_predictions, max_range_m)

    if dataset_dir is not None:
        labels = filter_drivable_area(labels, dataset_dir)
        track_predictions = filter_drivable_area(track_predictions, dataset_dir)

    track_predictions = _relabel_seq_ids(track_predictions)
    labels = _relabel_seq_ids(labels)

    score_thresholds, tuned_metric_values, _ = _tune_score_thresholds(
        labels,
        track_predictions,
        objective_metric,
        classes,
        num_thresholds=10,
        match_distance_m=2,
    )
    filtered_track_predictions = sm_utils.filter_by_class_thresholds(
        track_predictions, score_thresholds
    )

    res = evaluate_tracking(
        labels,
        filtered_track_predictions,
        classes,
        tracker_name="TRACKER",
        output_dir=out,
    )

    referrred_hota = tuned_metric_values["REFERRED_OBJECT"]

    if not full_tracks:
        scenario_f1, timestamp_f1 = compute_temporal_metrics(
            filtered_track_predictions, labels, out
        )
    else:
        scenario_f1 = 0
        timestamp_f1 = 0

    return referrred_hota, scenario_f1, timestamp_f1


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

    partial_track_hota, full_track_hota, timestamp_f1, scenario_f1 = evaluate(
        track_predictions, labels, objective_metric, max_range_m, dataset_dir, out
    )

    print(f"Temporally-localized HOTA: {partial_track_hota:.2f}")
    print(f"Full-lifespan HOTA: {full_track_hota:.2f}")
    print(f"Timestamp-level F1: {timestamp_f1:.2f}")
    print(f"Scenario-level F1: {scenario_f1:.2f}")


if __name__ == "__main__":
    runner()
