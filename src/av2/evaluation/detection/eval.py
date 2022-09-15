# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Argoverse 3D object detection evaluation.

Evaluation:

    Precision/Recall

        1. Average Precision: Standard VOC-style average precision calculation
            except a true positive requires a 3D Euclidean center distance of less
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
import itertools
import multiprocessing as mp
import logging
import warnings
from statistics import mean
from typing import Dict, Final, List, Optional, Tuple
import polars as pl

import numpy as np
import pandas as pd

from av2.evaluation.detection.constants import NUM_DECIMALS, MetricNames, TruePositiveErrorNames
from av2.evaluation.detection.utils import (
    DetectionCfg,
    accumulate,
    compute_average_precision,
    load_mapped_avm_and_egoposes,
)
from av2.geometry.se3 import SE3
from av2.map.map_api import ArgoverseStaticMap
from av2.structures.cuboid import ORDERED_CUBOID_COL_NAMES
from av2.utils.io import TimestampedCitySE3EgoPoses
from av2.utils.typing import NDArrayBool, NDArrayFloat
from joblib import Parallel, delayed

warnings.filterwarnings("ignore", module="google")

TP_ERROR_COLUMNS: Final[Tuple[str, ...]] = tuple(x.value for x in TruePositiveErrorNames)
DTS_COLUMN_NAMES: Final[Tuple[str, ...]] = tuple(ORDERED_CUBOID_COL_NAMES) + ("score",)
GTS_COLUMN_NAMES: Final[Tuple[str, ...]] = tuple(ORDERED_CUBOID_COL_NAMES) + ("num_interior_pts",)
UUID_COLUMN_NAMES: Final[Tuple[str, ...]] = (
    "log_id",
    "timestamp_ns",
    "category",
)
RANGE_BINS: Final[List[Tuple[(float, float)]]] = [(0.0, 150.0), (0.0, 50.0), (50.0, 100.0), (100.0, 150.0)]

logger = logging.getLogger(__name__)


def evaluate(
    dts: pd.DataFrame,
    gts: pd.DataFrame,
    cfg: DetectionCfg,
    n_jobs: int = 8,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate a set of detections against the ground truth annotations.

    Each sweep is processed independently, computing assignment between detections and ground truth annotations.

    Args:
        dts: (N,14) Table of detections.
        gts: (M,15) Table of ground truth annotations.
        cfg: Detection configuration.
        n_jobs: Number of jobs running concurrently during evaluation.

    Returns:
        (C+1,K) Table of evaluation metrics where C is the number of classes. Plus a row for their means.
        K refers to the number of evaluation metrics.

    Raises:
        RuntimeError: If accumulation fails.
        ValueError: If ROI pruning is enabled but a dataset directory is not specified.
    """
    if cfg.eval_only_roi_instances and cfg.dataset_dir is None:
        raise ValueError(
            "ROI pruning has been enabled, but the dataset directory has not be specified. "
            "Please set `dataset_directory` to the split root, e.g. av2/sensor/val."
        )

    dts: pl.DataFrame = pl.from_pandas(dts)
    gts: pl.DataFrame = pl.from_pandas(gts)

    # Sort both the detections and annotations by lexicographic order for grouping.
    dts = dts.sort(list(UUID_COLUMN_NAMES))
    gts = gts.sort(list(UUID_COLUMN_NAMES))

    # We merge the unique identifier -- the tuple of ("log_id", "timestamp_ns", "category")
    # into a single string to optimize the subsequent grouping operation.
    # `groupby_mapping` produces a mapping from the uuid to the group of detections / annotations
    # which fall into that group.
    uuid_to_dts = dts.partition_by(list(UUID_COLUMN_NAMES), as_dict=True)
    uuid_to_gts = gts.partition_by(list(UUID_COLUMN_NAMES), as_dict=True)

    uuid_to_dts = {k: v[list(DTS_COLUMN_NAMES)].to_numpy() for k, v in uuid_to_dts.items()}
    uuid_to_gts = {k: v[list(GTS_COLUMN_NAMES)].to_numpy() for k, v in uuid_to_gts.items()}

    log_id_to_avm: Optional[Dict[str, ArgoverseStaticMap]] = None
    log_id_to_timestamped_poses: Optional[Dict[str, TimestampedCitySE3EgoPoses]] = None

    dts = dts.to_pandas()
    gts = gts.to_pandas()

    # Load maps and egoposes if roi-pruning is enabled.
    if cfg.eval_only_roi_instances and cfg.dataset_dir is not None:
        logger.info("Loading maps and egoposes ...")
        log_ids: List[str] = gts.loc[:, "log_id"].unique().tolist()
        log_id_to_avm, log_id_to_timestamped_poses = load_mapped_avm_and_egoposes(log_ids, cfg.dataset_dir, n_jobs=n_jobs)

    args_list: List[Tuple[NDArrayFloat, NDArrayFloat, DetectionCfg, Optional[ArgoverseStaticMap], Optional[SE3]]] = []
    uuids = sorted(uuid_to_dts.keys() | uuid_to_gts.keys())
    for uuid in uuids:
        log_id, timestamp_ns, _ = uuid
        args: Tuple[NDArrayFloat, NDArrayFloat, DetectionCfg, Optional[ArgoverseStaticMap], Optional[SE3]]

        sweep_dts: NDArrayFloat = np.zeros((0, 10))
        sweep_gts: NDArrayFloat = np.zeros((0, 10))
        if uuid in uuid_to_dts:
            sweep_dts = uuid_to_dts[uuid]
        if uuid in uuid_to_gts:
            sweep_gts = uuid_to_gts[uuid]

        args = sweep_dts, sweep_gts, cfg, None, None
        if log_id_to_avm is not None and log_id_to_timestamped_poses is not None:
            avm = log_id_to_avm[log_id]
            city_SE3_ego = log_id_to_timestamped_poses[log_id][int(timestamp_ns)]
            args = sweep_dts, sweep_gts, cfg, avm, city_SE3_ego
        args_list.append(args)

    logger.info("Starting evaluation ...")
    with mp.get_context("forkserver").Pool(processes=n_jobs) as p:
        outputs: Optional[List[Tuple[NDArrayFloat, NDArrayFloat]]] = p.starmap(accumulate, args_list)

    if outputs is None:
        raise RuntimeError("Accumulation has failed! Please check the integrity of your detections and annotations.")
    dts_list, gts_list = zip(*outputs)

    METRIC_COLUMN_NAMES = cfg.affinity_thresholds_m + TP_ERROR_COLUMNS + ("is_evaluated",)
    dts_metrics: NDArrayFloat = np.concatenate(dts_list)
    gts_metrics: NDArrayFloat = np.concatenate(gts_list)

    dts.loc[:, METRIC_COLUMN_NAMES] = dts_metrics
    gts.loc[:, METRIC_COLUMN_NAMES] = gts_metrics

    # Compute summary metrics.
    metrics = summarize_metrics(dts, gts, cfg)
    metrics = metrics.round(NUM_DECIMALS)
    return dts, gts, metrics


def summarize_metrics(
    dts: pd.DataFrame,
    gts: pd.DataFrame,
    cfg: DetectionCfg,
) -> pd.DataFrame:
    """Calculate and print the 3D object detection metrics.

    Args:
        dts: (N,14) Table of detections.
        gts: (M,15) Table of ground truth annotations.
        cfg: Detection configuration.

    Returns:
        The summary metrics.
    """
    # Sample recall values in the [0, 1] interval.
    recall_interpolated: NDArrayFloat = np.linspace(0, 1, cfg.num_recall_samples, endpoint=True)

    # Initialize the summary metrics.
    metric_default_mapping = {
        metric_name.value: metric_default_value
        for metric_name, metric_default_value in zip(tuple(MetricNames), cfg.metrics_defaults)
    }
    keys = itertools.product(cfg.categories, metric_default_mapping.keys(), RANGE_BINS)
    summary = {key: [metric_default_mapping[key[-2]]] for key in keys}
    for category in cfg.categories:
        # Find detections that have the current category.
        is_category_dts = dts["category"] == category

        # Only keep detections if they match the category and have NOT been filtered.
        is_valid_dts = np.logical_and(is_category_dts, dts["is_evaluated"])

        # Get valid detections and sort them in descending order.
        category_dts = dts.loc[is_valid_dts].sort_values(by="score", ascending=False).reset_index(drop=True)

        # Find annotations that have the current category.
        is_category_gts = gts["category"] == category
        is_valid_gts = np.logical_and(is_category_gts, gts["is_evaluated"])

        # Compute number of ground truth annotations.
        category_gts = gts.loc[is_valid_gts]
        num_gts = category_gts["is_evaluated"].sum()

        # Cannot evaluate without ground truth information.
        if num_gts == 0:
            continue

        for (min_range, max_range) in RANGE_BINS:
            categories_dts_dists = np.linalg.norm(category_dts[["tx_m", "ty_m", "tz_m"]].to_numpy(), axis=-1)
            expr = (categories_dts_dists >= min_range) & (categories_dts_dists < max_range)
            category_dts_range = category_dts.iloc[expr]

            categories_gts_dists = np.linalg.norm(category_gts[["tx_m", "ty_m", "tz_m"]].to_numpy(), axis=-1)
            expr = (categories_gts_dists >= min_range) & (categories_gts_dists < max_range)
            category_gts_range = category_gts.iloc[expr]
            num_gts = len(category_gts_range)

            metrics = {
                (category, "AP", (min_range, max_range), affinity_threshold_m): 0.0
                for affinity_threshold_m in cfg.affinity_thresholds_m
            }
            for affinity_threshold_m in cfg.affinity_thresholds_m:
                true_positives: NDArrayBool = category_dts_range[affinity_threshold_m].astype(bool).to_numpy()

                # Continue if there aren't any true positives.
                if len(true_positives) == 0:
                    continue

                # Cannot evaluate without ground truth information.
                if num_gts == 0:
                    continue

                # Compute average precision for the current threshold.
                threshold_average_precision, _ = compute_average_precision(true_positives, recall_interpolated, num_gts)

                # Record the average precision.
                key = (category, "AP", (min_range, max_range), affinity_threshold_m)
                metrics[key] = threshold_average_precision

            mean_average_precisions = mean(
                [
                    metrics[(category, "AP", (min_range, max_range), affinity_threshold_m)]
                    for affinity_threshold_m in cfg.affinity_thresholds_m
                ]
            )

            # Select only the true positives for each instance.
            middle_idx = len(cfg.affinity_thresholds_m) // 2
            middle_threshold = cfg.affinity_thresholds_m[middle_idx]
            is_tp_t = category_dts_range[middle_threshold].to_numpy().astype(bool)

            # Initialize true positive metrics.
            tp_errors: NDArrayFloat = np.array(cfg.tp_normalization_terms)

            # Check whether any true positives exist under the current threshold.
            has_true_positives = np.any(is_tp_t)

            # If true positives exist, compute the metrics.
            if has_true_positives:
                tp_error_cols = [str(x.value) for x in TruePositiveErrorNames]
                tp_errors = category_dts_range.loc[is_tp_t, tp_error_cols].to_numpy().mean(axis=0)

            # Convert errors to scores.
            tp_scores = 1 - np.divide(tp_errors, cfg.tp_normalization_terms)

            # Compute Composite Detection Score (CDS).
            cds = mean_average_precisions * np.mean(tp_scores)
            summary[(category, "AP", (min_range, max_range))] = [mean_average_precisions]
            summary[(category, "ATE", (min_range, max_range))] = [tp_errors[0]]
            summary[(category, "ASE", (min_range, max_range))] = [tp_errors[1]]
            summary[(category, "AOE", (min_range, max_range))] = [tp_errors[2]]
            summary[(category, "CDS", (min_range, max_range))] = [cds]
    # Return the summary.
    return pd.DataFrame(summary)
