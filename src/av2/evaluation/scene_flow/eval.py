"""Argoverse 2 Scene Flow Evaluation."""

import argparse
from functools import partial
from pathlib import Path
from typing import Dict, Final, Iterator, List, Optional, Tuple, Union

import constants
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rich.progress import track

from av2.utils.typing import NDArrayBool, NDArrayFloat, NDArrayInt

EPS: Final = 1e-10


def epe(dts: NDArrayFloat, gts: NDArrayFloat) -> NDArrayFloat:
    """Compute the end-point-error between predictions and ground truth.

    Args:
        dts: (N,3) array containing predicted flows
        gts: (N,3) array containing ground truth flows

    Returns:
        The point-wise end-point-error
    """
    return np.array(np.linalg.norm(dts - gts, axis=-1), dtype=np.float64)


def accuracy(dts: NDArrayFloat, gts: NDArrayFloat, threshold: float) -> NDArrayFloat:
    """Compute the percent of inliers for a given threshold for a set of dtsictions and ground truth vectors.

    Args:
        dts: (N,3) array containing dtsicted flows
        gts: (N,3) array containing ground truth flows
        threshold: the threshold to use for classifying inliers

    Returns:
        The pointwise inlier assignments
    """
    l2_norm = np.linalg.norm(dts - gts, axis=-1)
    gts_norm = np.linalg.norm(gts, axis=-1)
    relative_err = l2_norm / (gts_norm + EPS)
    error_lt_5 = (l2_norm < threshold).astype(bool)
    relative_err_lt_5 = (relative_err < threshold).astype(bool)
    return np.array(np.logical_or(error_lt_5, relative_err_lt_5), dtype=np.float64)


def accuracy_strict(dts: NDArrayFloat, gts: NDArrayFloat) -> NDArrayFloat:
    """Compute the acccuracy with a 0.05 threshold.

    Args:
        dts: (N,3) array containing predicted flows
        gts: (N,3) array containing ground truth flows

    Returns:
        The pointwise inlier assignments at a 0.05 threshold
    """
    return accuracy(dts, gts, 0.05)


def accuracy_relax(dts: NDArrayFloat, gts: NDArrayFloat) -> NDArrayFloat:
    """Compute the acccuracy with a 0.1 threshold.

    Args:
        dts: (N,3) array containing predicted flows
        gts: (N,3) array containing ground truth flows

    Returns:
        The pointwise inlier assignments at a 0.1 threshold
    """
    return accuracy(dts, gts, 0.10)


def angle_error(dts: NDArrayFloat, gts: NDArrayFloat) -> NDArrayFloat:
    """Compute the angle error between dtsicted and ground truth flow vectors.

    Args:
        dts: (N,3) array containing predicted flows
        gts: (N,3) array containing ground truth flows

    Returns:
        The pointwise angle errors
    """
    unit_label = gts / (np.linalg.norm(gts, axis=-1, keepdims=True) + EPS)
    unit_dts = dts / (np.linalg.norm(dts, axis=-1, keepdims=True) + EPS)
    dot_product = np.clip(np.sum(unit_label * unit_dts, axis=-1), a_min=-1 + EPS, a_max=1 - EPS)
    return np.arccos(dot_product, dtype=np.float64)


def compute_true_positives(dts: NDArrayBool, gts: NDArrayBool) -> int:
    """Compute true positive count.

    Args:
        dts: (N,) array containing predicted dynamic segmentation.
        gts: (N,) array containing ground truth dynamic segmentation.

    Returns:
        The number of true positive classifications.
    """
    return int(np.logical_and(dts, gts).sum())


def compute_true_negatives(dts: NDArrayBool, gts: NDArrayBool) -> int:
    """Compute true negative count.

    Args:
        dts: (N,) array containing predicted dynamic segmentation.
        gts: (N,) array containing ground truth dynamic segmentation.

    Returns:
        The number of true negative classifications.
    """
    return int(np.logical_and(~dts, ~gts).sum())


def compute_false_positives(dts: NDArrayBool, gts: NDArrayBool) -> int:
    """Compute false positive count.

    Args:
        dts: (N,) array containing predicted dynamic segmentation.
        gts: (N,) array containing ground truth dynamic segmentation.

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


FLOW_METRICS: Final = {
    "EPE": epe,
    "Accuracy Strict": accuracy_strict,
    "Accuracy Relax": accuracy_relax,
    "Angle Error": angle_error,
}
SEG_METRICS: Final = {
    "TP": compute_false_positives,
    "TN": compute_false_negatives,
    "FP": compute_false_positives,
    "FN": compute_false_negatives,
}


def compute_metrics(
    pred_flow: NDArrayFloat,
    pred_dynamic: NDArrayBool,
    gt: NDArrayFloat,
    classes: NDArrayInt,
    dynamic: NDArrayBool,
    close: NDArrayBool,
    valid: NDArrayBool,
    object_classes: Dict[str, List[int]],
) -> List[List[Union[str, float, int]]]:
    """Compute all the metrics for a given example and package them into a list to be put into a DataFrame.

    Args:
        pred_flow: (N,3) Predicted flow vectors.
        pred_dynamic: (N,) Predicted dynamic labels.
        gt: (N, 3) Ground truth flow vectors.
        classes: (N,) Integer class labels for each point.
        dynamic: (N,) Ground truth dynamic labels.
        close: (N,) true for a point if it is within a 70m x 70m box around the AV
        valid: (N,) true for a point if its flow vector was succesfully computed.
        object_classes: A dictionary mapping class integers to segmentation labels (e.g., FOREGROUND_BACKGROUND).

    Returns:
        A list of lists where each sublist corresponds to some subset of the point cloud.
        (e.g., Dynamic/Foreground/Close). Each sublist contains the average of each metrics on that subset
        and the size of the subset.
    """
    results = []
    pred_flow = pred_flow[valid].astype(np.float64)
    pred_dynamic = pred_dynamic[valid].astype(bool)
    gt = gt[valid].astype(np.float64)
    classes = classes[valid].astype(int)
    dynamic = dynamic[valid].astype(bool)
    close = close[valid].astype(bool)

    for cls, class_idxs in object_classes.items():
        class_mask = classes == class_idxs[0]
        for i in class_idxs[1:]:
            class_mask = np.logical_or(class_mask, (classes == i))

        for motion, m_mask in [("Dynamic", dynamic), ("Static", ~dynamic)]:
            for distance, d_mask in [("Close", close), ("Far", ~close)]:
                mask = class_mask & m_mask & d_mask
                cnt = mask.sum().item()
                gt_sub = gt[mask]
                pred_sub = pred_flow[mask]
                result = [cls, motion, distance, cnt]
                if cnt > 0:
                    result += [FLOW_METRICS[m](pred_sub, gt_sub).mean() for m in FLOW_METRICS]
                    result += [SEG_METRICS[m](pred_dynamic[mask], dynamic[mask]) for m in SEG_METRICS]
                else:
                    result += [np.nan for m in FLOW_METRICS]
                    result += [0 for m in SEG_METRICS]
                results.append(result)
    return results


def evaluate_directories(annotations_root: Path, predictions_root: Path) -> pd.DataFrame:
    """Run the evaluation on predictions and labels saved to disk.

    Args:
        annotations_root: path to the directory containing the annotation files produced by make_annotation_files.py
        predictions_root: path to the prediction files in submission format

    Returns:
        A DataFrame containing the average metrics on each subset of each example.
    """
    results: List[List[Union[str, float, int]]] = []
    annotation_files = list(annotations_root.rglob("*.feather"))
    for anno_file in track(annotation_files, description="Evaluating..."):
        gt = pd.read_feather(anno_file)
        name: str = str(anno_file.relative_to(annotations_root))
        pred_file = predictions_root / name
        if not pred_file.exists():
            print(f"Warning: {name} missing")
            continue
        pred = pd.read_feather(pred_file)
        loss_breakdown = compute_metrics(
            pred[constants.FLOW_COLS].to_numpy(),
            pred["dynamic"].to_numpy(),
            gt[constants.FLOW_COLS].to_numpy(),
            gt["classes"].to_numpy(),
            gt["dynamic"].to_numpy(),
            gt["close"].to_numpy(),
            gt["valid"].to_numpy(),
            constants.FOREGROUND_BACKGROUND,
        )

        n: Union[str, float, int] = name  # make the type checker happy
        results.extend([[n] + bd for bd in loss_breakdown])
    df = pd.DataFrame(
        results, columns=["Example", "Class", "Motion", "Distance", "Count"] + list(FLOW_METRICS) + list(SEG_METRICS)
    )
    return df


def results_to_dict(results_dataframe: pd.DataFrame) -> Dict[str, float]:
    """Convert a results DataFrame to a dictionary of whole dataset metrics.

    Args:
        results_dataframe: DataFrame returned by evaluate_directories

    Returns:
        A dictionary string keys "<Motion/Class/Distance/Metric>" mapped to average metrics on that subset.
    """
    output = {}
    grouped = results_dataframe.groupby(["Class", "Motion", "Distance"])

    def weighted_average(x: pd.DataFrame, metric: str) -> pd.Series[float]:
        """Weighted average of metric m using the Count column."""
        averages: pd.Series[float] = (x[metric] * x.Count).sum() / x.Count.sum()
        return averages

    for m in FLOW_METRICS.keys():
        avg = grouped.apply(partial(weighted_average, metric=m))
        for segment in avg.index:
            if segment[0] == "Background" and segment[1] == "Dynamic":
                continue
            name = m + "/" + "/".join([str(i) for i in segment])
            output[name] = avg[segment]
    grouped = results_dataframe.groupby(["Class", "Motion"])
    for m in FLOW_METRICS.keys():
        avg = grouped.apply(partial(weighted_average, metric=m))
        for segment in avg.index:
            if segment[0] == "Background" and segment[1] == "Dynamic":
                continue
            name = m + "/" + "/".join([str(i) for i in segment])
            output[name] = avg[segment]
    output["Dynamic IoU"] = results_dataframe.TP.sum() / (
        results_dataframe.TP.sum() + results_dataframe.FP.sum() + results_dataframe.FN.sum()
    )
    output["EPE 3-Way Average"] = (
        output["EPE/Foreground/Dynamic"] + output["EPE/Foreground/Static"] + output["EPE/Background/Static"]
    ) / 3
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="eval", description="Run a local scene flow evaluation on the val split")
    parser.add_argument("predictions_root", type=str, help="path/to/predictions/")
    parser.add_argument("annotations_root", type=str, help="path/to/annotation_files/")

    args = parser.parse_args()
    results_df = evaluate_directories(Path(args.annotations_root), Path(args.predictions_root))
    results_dict = results_to_dict(results_df)

    for metric in sorted(results_dict):
        print(f"{metric}: {results_dict[metric]:.3f}")
    breakpoint()
