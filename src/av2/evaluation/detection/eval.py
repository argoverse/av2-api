# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Argoverse 3D object detection evaluation.

Evaluation:

    Precision/Recall

        1. Average Precision: Standard VOC-style average precision calculation
            except a true positive requires a bird's eye view center distance of less
            than a predefined threshold.

    True Positive Errors

        All true positive errors accumulate error solely when an object is a true positive match
        to a ground truth detection. The matching criterion is represented by `tp_thresh` in the DetectionCfg class.
        In our challenge, we use a `tp_thresh` of 2.0 meters.

        1. Average Translation Error: The average Euclidean distance (center-based) between a
            detection and its ground truth assignment.
        2. Average Scale Error: The average intersection over union (IoU) after the prediction
            and assigned ground truth's pose has been aligned.
        3. Average Orientation Error: The average angular distance between the detection and
            the assigned ground truth. We choose the smallest angle between the two different
            headings when calculating the error.

    Composite Scores

        1. Composite Detection Score: The ranking metric for the detection leaderboard. This
            is computed as the product of mAP with the sum of the complements of the true positive
            errors (after normalization), i.e.:
                - Average Translation Measure (ATM): ATE / TP_THRESHOLD; 0 <= 1 - ATE / TP_THRESHOLD <= 1.
                - Average Scaling Measure (ASM): 1 - ASE / 1;  0 <= 1 - ASE / 1 <= 1.
                - Average Orientation Measure (AOM): 1 - AOE / PI; 0 <= 1 - AOE / PI <= 1.

            These (as well as AP) are averaged over each detection class to produce:
                - mAP
                - mATM
                - mASM
                - mAOM

            Lastly, the Composite Detection Score is computed as:
                CDS = mAP * (mATE + mASE + mAOE); 0 <= mAP * (mATE + mASE + mAOE) <= 1.

        ** In the case of no true positives under the specified threshold, the true positive measures
            will assume their upper bounds of 1.0. respectively.

Results:

    The results are represented as a (C + 1, P) table, where C + 1 represents the number of evaluation classes
    in addition to the mean statistics average across all classes, and P refers to the number of included statistics,
    e.g. AP, ATE, ASE, AOE, CDS by default.
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rich.progress import track

from av2.evaluation.detection.constants import NUM_DECIMALS, MetricNames, TruePositiveErrorNames
from av2.evaluation.detection.utils import DetectionCfg, accumulate, compute_average_precision
from av2.map.map_api import ArgoverseStaticMap
from av2.utils.io import TimestampedCitySE3EgoPoses, read_city_SE3_ego
from av2.utils.typing import NDArrayBool, NDArrayFloat

logger = logging.getLogger(__name__)


def evaluate(
    dts: pd.DataFrame,
    gts: pd.DataFrame,
    cfg: DetectionCfg,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate a set of detections against the ground truth annotations.

    Each sweep is processed independently, computing assignment between detections and ground truth annotations.

    Args:
        dts: (N,15) Table of detections.
        gts: (M,15) Table of ground truth annotations.
        cfg: Detection configuration.
        poses: (K,8) Table of city egopose.

    Returns:
        (C+1,K) Table of evaluation metrics where C is the number of classes. Plus a row for their means.
        K refers to the number of evaluation metrics.

    Raises:
        RuntimeError: If parallel processing fails to complete.
    """
    # (log_id, timestamp_ns) uniquely identifies a sweep.
    uuid_columns = ["log_id", "timestamp_ns"]

    # Set index and sort for performance.
    dts = dts.set_index(keys=uuid_columns).sort_index()
    gts = gts.set_index(keys=uuid_columns).sort_index()

    # Find unique list of uuid tuples.
    uuids: List[Tuple[str, int]] = gts.index.unique().tolist()
    log_ids: List[str] = gts.index.get_level_values(level=0).unique()

    log_id_to_avm: Dict[str, ArgoverseStaticMap] = {}
    log_id_to_timestamped_poses: Dict[str, TimestampedCitySE3EgoPoses] = {}
    if cfg.eval_only_roi_instances and cfg.dataset_dir is not None:
        for log_id in log_ids:
            log_dir = cfg.dataset_dir / log_id
            avm_dir = log_dir / "map"
            avm = ArgoverseStaticMap.from_map_dir(avm_dir, build_raster=True)
            log_id_to_avm[log_id] = avm
            log_id_to_timestamped_poses[log_id] = read_city_SE3_ego(log_dir)

    args_list = []
    for (log_id, timestamp_ns) in track(uuids, "Loading maps ..."):
        sweep_dts = dts.loc[(log_id, timestamp_ns)].reset_index().copy()
        sweep_gts = gts.loc[(log_id, timestamp_ns)].reset_index().copy()

        sweep_map = None
        sweep_poses = None
        if log_id in log_id_to_avm:
            sweep_map = log_id_to_avm[log_id]
        if log_id in log_id_to_timestamped_poses:
            if timestamp_ns in log_id_to_timestamped_poses[log_id]:
                sweep_poses = log_id_to_timestamped_poses[log_id][timestamp_ns]
        args_list.append([sweep_dts, sweep_gts, cfg, sweep_map, sweep_poses])

    # Accumulate and gather the processed detections and ground truth annotations.
    results: Optional[List[Tuple[pd.DataFrame, pd.DataFrame]]] = Parallel(
        n_jobs=-1, backend="multiprocessing", verbose=True
    )(delayed(accumulate)(*x) for x in args_list)
    if results is None:
        raise RuntimeError("Accumulation of dts and gts has failed!")

    dts_list, gts_list = zip(*results)

    # Concatenate the detections and ground truth annotations into DataFrames.
    dts_processed: pd.DataFrame = pd.concat(dts_list).reset_index(drop=True)
    gts_processed: pd.DataFrame = pd.concat(gts_list).reset_index()

    # Compute summary metrics.
    metrics = summarize_metrics(dts_processed, gts_processed, cfg)
    metrics.loc["AVERAGE_METRICS"] = metrics.mean()
    metrics = metrics.round(NUM_DECIMALS)
    return dts, gts, metrics


def summarize_metrics(
    dts: pd.DataFrame,
    gts: pd.DataFrame,
    cfg: DetectionCfg,
) -> pd.DataFrame:
    """Calculate and print the 3D object detection metrics.

    Args:
        dts: (N,15) Table of detections.
        gts: (M,15) Table of ground truth annotations.
        cfg: Detection configuration.

    Returns:
        The summary metrics.
    """
    # Sample recall values in the [0, 1] interval.
    recall_interpolated: NDArrayFloat = np.linspace(0, 1, cfg.num_recall_samples)

    # Initialize the summary metrics.
    summary = pd.DataFrame(
        {s.value: cfg.metrics_defaults[i] for i, s in enumerate(tuple(MetricNames))}, index=cfg.categories
    )

    average_precisions = pd.DataFrame({t: 0.0 for t in cfg.affinity_thresholds_m}, index=cfg.categories)
    for category in cfg.categories:
        # Find detections that have the current category.
        is_category_dts = dts["category"] == category

        # Only keep detections if they match the category and have NOT been filtered.
        is_valid_dts = np.logical_and(is_category_dts, dts["is_evaluated"])

        # Get valid detections and sort them in ascending order.
        category_dts = dts.loc[is_valid_dts].sort_values(by="score", ascending=False).reset_index(drop=True)

        # Find annotations that have the current category.
        is_category_gts = gts["category"] == category

        # Compute number of ground truth annotations.
        num_gts = gts.loc[is_category_gts, "is_evaluated"].sum()

        # Cannot evaluate without ground truth information.
        if num_gts == 0:
            continue

        for affinity_threshold_m in cfg.affinity_thresholds_m:
            true_positives: NDArrayBool = category_dts[affinity_threshold_m].to_numpy()

            # Continue if there aren't any true positives.
            if len(true_positives) == 0:
                continue

            # Compute average precision for the current threshold.
            threshold_average_precision, _ = compute_average_precision(true_positives, recall_interpolated, num_gts)

            # Record the average precision.
            average_precisions.loc[category, affinity_threshold_m] = threshold_average_precision

        mean_average_precisions = average_precisions.loc[category].mean()

        # Select only the true positives for each instance.
        middle_idx = len(cfg.affinity_thresholds_m) // 2
        middle_threshold = cfg.affinity_thresholds_m[middle_idx]
        is_tp_t = category_dts[middle_threshold]

        # Initialize true positive metrics.
        tp_errors = cfg.tp_normalization_terms

        # Check whether any true positives exist under the current threshold.
        has_true_positives = np.any(is_tp_t)

        # If true positives exist, compute the metrics.
        if has_true_positives:
            tp_errors = category_dts.loc[is_tp_t, tuple(x.value for x in TruePositiveErrorNames)].mean(axis=0)

        # Convert errors to scores.
        tp_scores = 1 - np.divide(tp_errors, cfg.tp_normalization_terms)

        # Compute Composite Detection Score (CDS).
        cds = mean_average_precisions * np.mean(tp_scores)
        summary.loc[category] = [mean_average_precisions, *tp_errors, cds]

    # Return the summary.
    return summary
