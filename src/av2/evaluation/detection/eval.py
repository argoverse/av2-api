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
from typing import Dict, Final, List, Optional, Tuple

import numpy as np
import pandas as pd
from rich.progress import track

from av2.evaluation.detection.constants import NUM_DECIMALS, MetricNames, TruePositiveErrorNames
from av2.evaluation.detection.utils import (
    DetectionCfg,
    accumulate,
    compute_average_precision,
    load_mapped_avm_and_egoposes,
)
from av2.map.map_api import ArgoverseStaticMap
from av2.utils.io import TimestampedCitySE3EgoPoses, read_city_SE3_ego
from av2.utils.typing import NDArrayBool, NDArrayFloat, NDArrayInt, NDArrayObject

TP_ERROR_COLUMNS: Final[Tuple[str, ...]] = tuple(x.value for x in TruePositiveErrorNames)

logger = logging.getLogger(__name__)


def groupby_mapping(uuids: NDArrayObject, values: NDArrayFloat) -> Dict[Tuple[str, ...], NDArrayFloat]:
    outputs: Tuple[NDArrayInt, NDArrayInt] = np.unique(uuids, axis=0, return_index=True)
    unique_items, unique_items_indices = outputs
    dts_groups: List[NDArrayFloat] = np.split(values, unique_items_indices[1:])
    uuid_to_groups = {tuple(unique_items[i].tolist()): x for i, x in enumerate(dts_groups)}
    return uuid_to_groups


CUBOID_COLS: Final[List[str]] = ["tx_m", "ty_m", "tz_m", "length_m", "width_m", "height_m", "qw", "qx", "qy", "qz"]


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

    Returns:
        (C+1,K) Table of evaluation metrics where C is the number of classes. Plus a row for their means.
        K refers to the number of evaluation metrics.
    """
    uuid_column_names = ("log_id", "timestamp_ns", "category")
    dts = dts.sort_values(uuid_column_names)
    gts = gts.sort_values(uuid_column_names)

    dts_cols = CUBOID_COLS + ["score"]
    gts_cols = CUBOID_COLS
    dts_npy = dts.loc[:, dts_cols].to_numpy()
    gts_npy = gts.loc[:, gts_cols].to_numpy()

    # Convert UUID into fixed bitwidth for unique computation.
    dts_uuids: NDArrayFloat = dts[uuid_column_names].to_numpy().astype("U64")
    gts_uuids: NDArrayFloat = gts[uuid_column_names].to_numpy().astype("U64")

    uuid_to_dts = groupby_mapping(dts_uuids, dts_npy)
    uuid_to_gts = groupby_mapping(gts_uuids, gts_npy)

    dts_list = []
    gts_list = []
    for k, v in track(uuid_to_gts.items()):
        dts_accum, gts_accum = accumulate(uuid_to_dts[k], v, cfg, None, None)
        dts_list.append(dts_accum)
        gts_list.append(gts_accum)

    # log_id_to_avm: Optional[Dict[str, ArgoverseStaticMap]] = None
    # log_id_to_timestamped_poses: Optional[Dict[str, TimestampedCitySE3EgoPoses]] = None

    # # Load maps and egoposes if roi-pruning is enabled.
    # if cfg.eval_only_roi_instances and cfg.dataset_dir is not None:
    #     log_id_to_avm, log_id_to_timestamped_poses = load_mapped_avm_and_egoposes(gts, cfg.dataset_dir)

    # column_names = ["log_id", "timestamp_ns", "category"]
    # uuid_to_dts = {uuid: x for uuid, x in dts.groupby(column_names, sort=False)}
    # uuid_to_gts = {uuid: x for uuid, x in gts.groupby(column_names, sort=False)}

    # dts_list = []
    # gts_list = []
    # for uuid, sweep_gts in track(uuid_to_gts.items()):
    #     args = uuid_to_dts[uuid], sweep_gts, cfg, None, None
    #     if log_id_to_avm is not None and log_id_to_timestamped_poses is not None:
    #         avm = log_id_to_avm[uuid[0]]
    #         city_SE3_ego = log_id_to_timestamped_poses[uuid[0]][uuid[1]]
    #         args = uuid_to_dts[uuid], sweep_gts, cfg, avm, city_SE3_ego
    #     dts_accum, gts_accum = accumulate(*args)
    #     dts_list.append(dts_accum)
    #     gts_list.append(gts_accum)
    dts_npy: NDArrayFloat = np.concatenate(dts_list)
    gts_npy: NDArrayFloat = np.concatenate(gts_list)

    COLS = cfg.affinity_thresholds_m + TP_ERROR_COLUMNS + ("is_evaluated",)
    dts.loc[:, COLS] = dts_npy
    gts.loc[:, COLS] = gts_npy

    # Compute summary metrics.
    metrics = summarize_metrics(dts, gts, cfg)
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

        # Get valid detections and sort them in descending order.
        category_dts = dts.loc[is_valid_dts].sort_values(by="score", ascending=False).reset_index(drop=True)

        # Find annotations that have the current category.
        is_category_gts = gts["category"] == category

        # Compute number of ground truth annotations.
        num_gts = gts.loc[is_category_gts, "is_evaluated"].sum()

        # Cannot evaluate without ground truth information.
        if num_gts == 0:
            continue

        for affinity_threshold_m in cfg.affinity_thresholds_m:
            true_positives: NDArrayBool = category_dts[affinity_threshold_m].astype(bool).to_numpy()

            # Continue if there aren't any true positives.
            if len(true_positives) == 0:
                continue

            # Compute average precision for the current threshold.
            threshold_average_precision, _ = compute_average_precision(true_positives, recall_interpolated, num_gts)

            # Record the average precision.
            average_precisions.loc[category, affinity_threshold_m] = threshold_average_precision

        mean_average_precisions: NDArrayFloat = average_precisions.loc[category].to_numpy().mean()

        # Select only the true positives for each instance.
        middle_idx = len(cfg.affinity_thresholds_m) // 2
        middle_threshold = cfg.affinity_thresholds_m[middle_idx]
        is_tp_t = category_dts[middle_threshold].to_numpy().astype(bool)

        # Initialize true positive metrics.
        tp_errors = np.array(cfg.tp_normalization_terms)

        # Check whether any true positives exist under the current threshold.
        has_true_positives = np.any(is_tp_t)

        # If true positives exist, compute the metrics.
        if has_true_positives:
            tp_error_cols = [str(x.value) for x in TruePositiveErrorNames]
            tp_errors: NDArrayFloat = category_dts.loc[is_tp_t, tp_error_cols].to_numpy().mean(axis=0)

        # Convert errors to scores.
        tp_scores = 1 - np.divide(tp_errors, cfg.tp_normalization_terms)

        # Compute Composite Detection Score (CDS).
        cds = mean_average_precisions * np.mean(tp_scores)
        summary.loc[category] = np.array([mean_average_precisions, *tp_errors, cds])

    # Return the summary.
    return summary
