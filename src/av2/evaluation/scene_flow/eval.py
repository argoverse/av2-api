"""Argoverse 2 Scene Flow Evaluation."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Dict, Final, Iterator, List, Optional, Tuple, Union

import click
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
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
    relative_err = l2_norm / (gts_norm + EPS)
    abs_error_inlier = (l2_norm < distance_threshold).astype(bool)
    relative_err_inlier = (relative_err < distance_threshold).astype(bool)
    accuracy: NDArrayFloat = np.logical_or(abs_error_inlier, relative_err_inlier).astype(np.float64)
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
    gts_space_time = np.pad(gts, ((0, 0), (0, 1)), constant_values=constants.SWEEP_PAIR_TIME_DELTA)
    dts_space_time = np.pad(dts, ((0, 0), (0, 1)), constant_values=constants.SWEEP_PAIR_TIME_DELTA)

    unit_gts = gts_space_time / (np.linalg.norm(gts_space_time, axis=-1, keepdims=True))
    unit_dts = dts_space_time / (np.linalg.norm(dts_space_time, axis=-1, keepdims=True))
    # Floating point errors can cause the dp to be slightly greater than 1 or less than -1.
    dot_product = np.clip(np.sum(unit_gts * unit_dts, axis=-1), -1.0, 1.0)
    angle_error: NDArrayFloat = np.arccos(dot_product).astype(np.float64)
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
    metric_classes: Dict[constants.MetricBreakdownCategories, List[int]],
) -> List[List[Union[str, float, int]]]:
    """Compute all the metrics for a given example and package them into a list to be put into a DataFrame.

    Args:
        pred_flow: (N,3) Predicted flow vectors.
        pred_dynamic: (N,) Predicted dynamic labels.
        gts: (N,3) Ground truth flow vectors.
        category_indices: (N,) Integer class labels for each point.
        is_dynamic: (N,) Ground truth dynamic labels.
        is_close: (N,) True for a point if it is within a 70m x 70m box around the AV.
        is_valid: (N,) True for a point if its flow vector was succesfully computed.
        metric_classes: A dictionary mapping segmentation labels to groups of category indices.

    Returns:
        A list of lists where each sublist corresponds to some subset of the point cloud.
        (e.g., Dynamic/Foreground/Close). Each sublist contains the average of each metrics on that subset
        and the size of the subset.
    """
    results = []
    pred_flow = pred_flow[is_valid].astype(np.float64)
    pred_dynamic = pred_dynamic[is_valid].astype(bool)
    gts = gts[is_valid].astype(np.float64)
    category_indices = category_indices[is_valid].astype(int)
    is_dynamic = is_dynamic[is_valid].astype(bool)
    is_close = is_close[is_valid].astype(bool)
    flow_metrics = constants.FLOW_METRICS
    seg_metrics = constants.SEGMENTATION_METRICS

    for cls, category_idxs in metric_classes.items():
        category_mask = category_indices == category_idxs[0]
        for i in category_idxs[1:]:
            category_mask = np.logical_or(category_mask, (category_indices == i))

        for motion, m_mask in [("Dynamic", is_dynamic), ("Static", ~is_dynamic)]:
            for distance, d_mask in [("Close", is_close), ("Far", ~is_close)]:
                mask = category_mask & m_mask & d_mask
                subset_size = mask.sum().item()
                gts_sub = gts[mask]
                pred_sub = pred_flow[mask]
                result = [cls.value, motion, distance, subset_size]
                if subset_size > 0:
                    result += [flow_metrics[m](pred_sub, gts_sub).mean() for m in flow_metrics]
                    result += [seg_metrics[m](pred_dynamic[mask], is_dynamic[mask]) for m in seg_metrics]
                else:
                    result += [np.nan for m in flow_metrics]
                    result += [0 for m in seg_metrics]
                results.append(result)
    return results


def evaluate_directories(annotations_dir: Path, predictions_dir: Path) -> pd.DataFrame:
    """Run the evaluation on predictions and labels saved to disk.

    Args:
        annotations_dir: Path to the directory containing the annotation files produced by `make_annotation_files.py`.
        predictions_dir: Path to the prediction files in submission format.

    Returns:
        DataFrame containing the average metrics on each subset of each example.
    """
    results: List[List[Union[str, float, int]]] = []
    annotation_files = list(annotations_dir.rglob("*.feather"))
    for anno_file in track(annotation_files, description="Evaluating..."):
        gts = pd.read_feather(anno_file)
        name: str = str(anno_file.relative_to(annotations_dir))
        pred_file = predictions_dir / name
        if not pred_file.exists():
            print(f"Warning: {name} missing")
            continue
        pred = pd.read_feather(pred_file)
        loss_breakdown = compute_metrics(
            pred[list(constants.FLOW_COLUMNS)].to_numpy().astype(float),
            pred["is_dynamic"].to_numpy().astype(bool),
            gts[list(constants.FLOW_COLUMNS)].to_numpy().astype(float),
            gts["category_indices"].to_numpy().astype(np.uint8),
            gts["is_dynamic"].to_numpy().astype(bool),
            gts["is_close"].to_numpy().astype(bool),
            gts["is_valid"].to_numpy().astype(bool),
            constants.FOREGROUND_BACKGROUND_BREAKDOWN,
        )

        n: Union[str, float, int] = name  # make the type checker happy
        results.extend([[n] + bd for bd in loss_breakdown])
    df = pd.DataFrame(
        results,
        columns=["Example", "Class", "Motion", "Distance", "Count"]
        + list(constants.FLOW_METRICS)
        + list(constants.SEGMENTATION_METRICS),
    )
    return df


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


@click.command()
@click.argument("annotations_dir", type=str)
@click.argument("predictions_dir", type=str)
def evaluate(annotations_dir: str, predictions_dir: str) -> None:
    """Evaluate a set of predictions and print the results.

    Args:
        annotations_dir: Path to the directory containing the annotation files produced by `make_annotation_files.py`.
        predictions_dir: Path to the prediction files in submission format.
    """
    results_df = evaluate_directories(Path(annotations_dir), Path(predictions_dir))
    results_dict = results_to_dict(results_df)

    for metric in sorted(results_dict):
        print(f"{metric}: {results_dict[metric]:.3f}")


if __name__ == "__main__":
    evaluate()
