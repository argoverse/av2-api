"""Argoverse 2 Scene Flow Evaluation."""

from __future__ import annotations

from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any, DefaultDict, Dict, Final, List, Union

import click
import numpy as np
import pandas as pd
from rich.progress import track

import av2.evaluation.scene_flow.constants as constants
from av2.utils.typing import NDArrayBool, NDArrayFloat, NDArrayInt

ACCURACY_RELAX_DISTANCE_THRESHOLD: Final = 0.1
ACCURACY_STRICT_DISTANCE_THRESHOLD: Final = 0.05
EPS: Final = 1e-10


def compute_end_point_error(dts: NDArrayFloat, gts: NDArrayFloat) -> NDArrayFloat:
    """Compute the end-point error between predictions and ground truth.

    Args:
        dts: (N,3) Array containing predicted flows.
        gts: (N,3) Array containing ground truth flows.

    Returns:
        The point-wise end-point error.
    """
    end_point_error: NDArrayFloat = np.linalg.norm(dts - gts, axis=-1).astype(np.float64)
    return end_point_error


def compute_accuracy(dts: NDArrayFloat, gts: NDArrayFloat, distance_threshold: float) -> NDArrayFloat:
    """Compute the percent of inliers for a given threshold for a set of prediction and ground truth vectors.

    Args:
        dts: (N,3) Array containing predicted flows.
        gts: (N,3) Array containing ground truth flows.
        distance_threshold: Distance threshold for classifying inliers.

    Returns:
        The pointwise inlier assignments.
    """
    l2_norm = np.linalg.norm(dts - gts, axis=-1)
    gts_norm = np.linalg.norm(gts, axis=-1)
    relative_error = np.divide(l2_norm, gts_norm + EPS)
    abs_error_inlier = np.less(l2_norm, distance_threshold).astype(bool)
    relative_error_inlier = np.less(relative_error, distance_threshold).astype(bool)
    accuracy: NDArrayFloat = np.logical_or(abs_error_inlier, relative_error_inlier).astype(np.float64)
    return accuracy


def compute_accuracy_strict(dts: NDArrayFloat, gts: NDArrayFloat) -> NDArrayFloat:
    """Compute the acccuracy with a 0.05 threshold.

    Args:
        dts: (N,3) Array containing predicted flows.
        gts: (N,3) Array containing ground truth flows.

    Returns:
        The pointwise inlier assignments at a 0.05 threshold
    """
    return compute_accuracy(dts, gts, ACCURACY_STRICT_DISTANCE_THRESHOLD)


def compute_accuracy_relax(dts: NDArrayFloat, gts: NDArrayFloat) -> NDArrayFloat:
    """Compute the acccuracy with a 0.1 threshold.

    Args:
        dts: (N,3) Array containing predicted flows.
        gts: (N,3) Array containing ground truth flows.

    Returns:
        The pointwise inlier assignments at a 0.1 threshold.
    """
    return compute_accuracy(dts, gts, ACCURACY_RELAX_DISTANCE_THRESHOLD)


def compute_angle_error(dts: NDArrayFloat, gts: NDArrayFloat) -> NDArrayFloat:
    """Compute the angle error in space-time between the prediced and ground truth flow vectors.

    Args:
        dts: (N,3) Array containing predicted flows.
        gts: (N,3) Array containing ground truth flows.

    Returns:
        The pointwise angle errors in space-time.
    """
    # Convert the 3D flow vectors to 4D space-time vectors.
    dts_space_time = np.pad(dts, ((0, 0), (0, 1)), constant_values=constants.SWEEP_PAIR_TIME_DELTA)
    gts_space_time = np.pad(gts, ((0, 0), (0, 1)), constant_values=constants.SWEEP_PAIR_TIME_DELTA)

    dts_space_time_norm = np.linalg.norm(dts_space_time, axis=-1, keepdims=True)
    gts_space_time_norm = np.linalg.norm(gts_space_time, axis=-1, keepdims=True)
    unit_dts = dts_space_time / dts_space_time_norm
    unit_gts = gts_space_time / gts_space_time_norm

    dot_product = np.einsum("bd,bd->b", unit_dts, unit_gts)

    # Floating point errors can cause `dot_product` to be slightly greater than 1 or less than -1.
    clipped_dot_product = np.clip(dot_product, -1.0, 1.0)
    angle_error: NDArrayFloat = np.arccos(clipped_dot_product).astype(np.float64)
    return angle_error


def compute_true_positives(dts: NDArrayBool, gts: NDArrayBool) -> int:
    """Compute true positive count.

    Args:
        dts: (N,) Array containing predicted dynamic segmentation.
        gts: (N,) Array containing ground truth dynamic segmentation.

    Returns:
        The number of true positive classifications.
    """
    return int(np.logical_and(dts, gts).sum())


def compute_true_negatives(dts: NDArrayBool, gts: NDArrayBool) -> int:
    """Compute true negative count.

    Args:
        dts: (N,) Array containing predicted dynamic segmentation.
        gts: (N,) Array containing ground truth dynamic segmentation.

    Returns:
        The number of true negative classifications.
    """
    return int(np.logical_and(~dts, ~gts).sum())


def compute_false_positives(dts: NDArrayBool, gts: NDArrayBool) -> int:
    """Compute false positive count.

    Args:
        dts: (N,) Array containing predicted dynamic segmentation.
        gts: (N,) Array containing ground truth dynamic segmentation.

    Returns:
        The number of false positive classifications.
    """
    return int(np.logical_and(dts, ~gts).sum())


def compute_false_negatives(dts: NDArrayBool, gts: NDArrayBool) -> int:
    """Compute false negative count.

    Args:
        dts: (N,) Array containing predicted dynamic segmentation.
        gts: (N,) Array containing ground truth dynamic segmentation.

    Returns:
        The number of false negative classifications
    """
    return int(np.logical_and(~dts, gts).sum())


def compute_metrics(
    pred_flow: NDArrayFloat,
    pred_dynamic: NDArrayBool,
    gts: NDArrayFloat,
    category_indices: NDArrayInt,
    is_dynamic: NDArrayBool,
    is_close: NDArrayBool,
    is_valid: NDArrayBool,
    metric_categories: Dict[constants.MetricBreakdownCategories, List[int]],
) -> Dict[str, List[Any]]:
    """Compute all the metrics for a given example and package them into a list to be put into a DataFrame.

    Args:
        pred_flow: (N,3) Predicted flow vectors.
        pred_dynamic: (N,) Predicted dynamic labels.
        gts: (N,3) Ground truth flow vectors.
        category_indices: (N,) Integer class labels for each point.
        is_dynamic: (N,) Ground truth dynamic labels.
        is_close: (N,) True for a point if it is within a 70m x 70m box around the AV.
        is_valid: (N,) True for a point if its flow vector was succesfully computed.
        metric_categories: A dictionary mapping segmentation labels to groups of category indices.

    Returns:
        A dictionary of columns to create a long-form DataFrame of the results from.
        One row for each subset in the breakdown.
    """
    pred_flow = pred_flow[is_valid].astype(np.float64)
    pred_dynamic = pred_dynamic[is_valid].astype(bool)
    gts = gts[is_valid].astype(np.float64)
    category_indices = category_indices[is_valid].astype(int)
    is_dynamic = is_dynamic[is_valid].astype(bool)
    is_close = is_close[is_valid].astype(bool)
    flow_metrics = constants.FLOW_METRICS
    seg_metrics = constants.SEGMENTATION_METRICS

    results: DefaultDict[str, List[Any]] = defaultdict(list)

    # Each metric is broken down by point labels on Object Class, Motion, and Distance from the AV.
    # We iterate over all combinations of those three categories and compute average metrics on each subset.
    for cls, category_idxs in metric_categories.items():
        # Compute the union of all masks within the meta-category.
        category_mask = category_indices == category_idxs[0]
        for i in category_idxs[1:]:
            category_mask = np.logical_or(category_mask, (category_indices == i))

        for motion, m_mask in [("Dynamic", is_dynamic), ("Static", ~is_dynamic)]:
            for distance, d_mask in [("Close", is_close), ("Far", ~is_close)]:
                mask = category_mask & m_mask & d_mask
                subset_size = mask.sum().item()
                gts_sub = gts[mask]
                pred_sub = pred_flow[mask]
                results["Class"] += [cls.value]
                results["Motion"] += [motion]
                results["Distance"] += [distance]
                results["Count"] += [subset_size]

                # Check if there are any points in this subset and if so compute all the average metrics.
                if subset_size > 0:
                    for metric, flow_metric_fn in flow_metrics.items():
                        results[metric] += [flow_metric_fn(pred_sub, gts_sub).mean()]
                    for metric, seg_metric_fn in seg_metrics.items():
                        results[metric] += [seg_metric_fn(pred_dynamic[mask], is_dynamic[mask])]
                else:
                    for metric in flow_metrics:
                        results[metric] += [np.nan]
                    for metric in seg_metrics:
                        results[metric] += [0]
    return results


def evaluate_directories(annotations_dir: Path, predictions_dir: Path) -> pd.DataFrame:
    """Run the evaluation on predictions and labels saved to disk.

    Args:
        annotations_dir: Path to the directory containing the annotation files produced by `make_annotation_files.py`.
        predictions_dir: Path to the prediction files in submission format.

    Returns:
        DataFrame containing the average metrics on each subset of each example.
    """
    results: DefaultDict[str, List[Any]] = defaultdict(list)
    annotation_files = list(annotations_dir.rglob("*.feather"))
    for anno_file in track(annotation_files, description="Evaluating..."):
        gts = pd.read_feather(anno_file)
        name: str = str(anno_file.relative_to(annotations_dir))
        pred_file = predictions_dir / name
        if not pred_file.exists():
            print(f"Warning: File {name} is missing!")
            continue
        pred = pd.read_feather(pred_file)
        current_example_results = compute_metrics(
            pred[list(constants.FLOW_COLUMNS)].to_numpy().astype(float),
            pred["is_dynamic"].to_numpy().astype(bool),
            gts[list(constants.FLOW_COLUMNS)].to_numpy().astype(float),
            gts["category_indices"].to_numpy().astype(np.uint8),
            gts["is_dynamic"].to_numpy().astype(bool),
            gts["is_close"].to_numpy().astype(bool),
            gts["is_valid"].to_numpy().astype(bool),
            constants.FOREGROUND_BACKGROUND_BREAKDOWN,
        )
        num_subsets = len(list(current_example_results.values())[0])
        results["Example"] += [name for _ in range(num_subsets)]
        for m in current_example_results:
            results[m] += current_example_results[m]

    return pd.DataFrame(results)


def results_to_dict(frame: pd.DataFrame) -> Dict[str, float]:
    """Convert a results DataFrame to a dictionary of whole dataset metrics.

    Args:
        frame: DataFrame returned by evaluate_directories.

    Returns:
        Dictionary string keys "<Motion/Class/Distance/Metric>" mapped to average metrics on that subset.
    """
    output = {}
    grouped = frame.groupby(["Class", "Motion", "Distance"])

    def weighted_average(x: pd.DataFrame, metric: str) -> float:
        """Weighted average of metric m using the Count column."""
        total = int(x["Count"].sum())
        if total == 0:
            return np.nan
        averages: float = (x[metric] * x.Count).sum() / total
        return averages

    for m in constants.FLOW_METRICS.keys():
        avg = grouped.apply(partial(weighted_average, metric=m))
        for segment in avg.index:
            if segment[0] == "Background" and segment[1] == "Dynamic":
                continue
            name = m + "/" + "/".join([str(i) for i in segment])
            output[name] = avg[segment]
    grouped = frame.groupby(["Class", "Motion"])
    for m in constants.FLOW_METRICS.keys():
        avg = grouped.apply(partial(weighted_average, metric=m))
        for segment in avg.index:
            if segment[0] == "Background" and segment[1] == "Dynamic":
                continue
            name = m + "/" + "/".join([str(i) for i in segment])
            output[name] = avg[segment]
    output["Dynamic IoU"] = frame.TP.sum() / (frame.TP.sum() + frame.FP.sum() + frame.FN.sum())
    output["EPE 3-Way Average"] = (
        output["EPE/Foreground/Dynamic"] + output["EPE/Foreground/Static"] + output["EPE/Background/Static"]
    ) / 3

    return output


def evaluate(annotations_dir: str, predictions_dir: str) -> Dict[str, float]:
    """Evaluate a set of predictions and print the results.

    Args:
        annotations_dir: Path to the directory containing the annotation files produced by `make_annotation_files.py`.
        predictions_dir: Path to the prediction files in submission format.

    Returns:
        The results as a dict of metric names and values.
    """
    results_df = evaluate_directories(Path(annotations_dir), Path(predictions_dir))
    results_dict = results_to_dict(results_df)

    for metric in sorted(results_dict):
        print(f"{metric}: {results_dict[metric]:.3f}")

    return results_dict


@click.command()
@click.argument("annotations_dir", type=str)
@click.argument("predictions_dir", type=str)
def _evaluate_entry(annotations_dir: str, predictions_dir: str) -> Dict[str, float]:
    """Entry point for evaluate."""
    return evaluate(annotations_dir, predictions_dir)


if __name__ == "__main__":
    _evaluate_entry()
