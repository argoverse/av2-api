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
import logging
import multiprocessing as mp
import warnings
from typing import Any, Dict, Final, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm

from av2.evaluation.detection.constants import (
    HIERARCHY,
    LCA,
    LCA_COLUMNS,
    NUM_DECIMALS,
    MetricNames,
    TruePositiveErrorNames,
)
from av2.evaluation.detection.utils import (
    DetectionCfg,
    accumulate,
    accumulate_hierarchy,
    compute_average_precision,
    is_evaluated,
    load_mapped_avm_and_egoposes,
)
from av2.geometry.se3 import SE3
from av2.map.map_api import ArgoverseStaticMap
from av2.structures.cuboid import ORDERED_CUBOID_COL_NAMES
from av2.utils.io import TimestampedCitySE3EgoPoses
from av2.utils.typing import NDArrayBool, NDArrayFloat, NDArrayObject

warnings.filterwarnings("ignore", module="google")

TP_ERROR_COLUMNS: Final = tuple(x.value for x in TruePositiveErrorNames)
DTS_COLUMNS: Final = tuple(ORDERED_CUBOID_COL_NAMES) + ("score",)
GTS_COLUMNS: Final = tuple(ORDERED_CUBOID_COL_NAMES) + ("num_interior_pts",)

UUID_COLUMNS: Final = ("log_id", "timestamp_ns")
CATEGORY_COLUMN: Final = ("category",)
DETECTION_UUID_COLUMNS: Final = UUID_COLUMNS + CATEGORY_COLUMN

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
        ValueError: If ROI pruning is enabled but a dataset directory is not specified.
    """
    if cfg.eval_only_roi_instances and cfg.dataset_dir is None:
        raise ValueError(
            "ROI pruning has been enabled, but the dataset directory has not be specified. "
            "Please set `dataset_directory` to the split root, e.g. av2/sensor/val."
        )

    # Sort both the detections and annotations by lexicographic order for grouping.
    dts = dts.sort_values(list(DETECTION_UUID_COLUMNS))
    gts = gts.sort_values(list(DETECTION_UUID_COLUMNS))

    dts_pl = pl.from_pandas(dts)
    gts_pl = pl.from_pandas(gts)

    uuid_to_dts = {
        k: v[list(DTS_COLUMNS)].to_numpy().astype(float)
        for k, v in dts_pl.partition_by(
            DETECTION_UUID_COLUMNS, maintain_order=True, as_dict=True
        ).items()
    }
    uuid_to_gts = {
        k: v[list(GTS_COLUMNS)].to_numpy().astype(float)
        for k, v in gts_pl.partition_by(
            DETECTION_UUID_COLUMNS, maintain_order=True, as_dict=True
        ).items()
    }

    log_id_to_avm: Optional[Dict[str, ArgoverseStaticMap]] = None
    log_id_to_timestamped_poses: Optional[Dict[str, TimestampedCitySE3EgoPoses]] = None

    # Load maps and egoposes if roi-pruning is enabled.
    if cfg.eval_only_roi_instances and cfg.dataset_dir is not None:
        logger.info("Loading maps and egoposes ...")
        log_ids: List[str] = gts.loc[:, "log_id"].unique().tolist()
        log_id_to_avm, log_id_to_timestamped_poses = load_mapped_avm_and_egoposes(
            log_ids, cfg.dataset_dir
        )

    accumulate_args_list: List[
        Tuple[
            NDArrayFloat,
            NDArrayFloat,
            DetectionCfg,
            Optional[ArgoverseStaticMap],
            Optional[SE3],
        ]
    ] = []
    uuids = sorted(uuid_to_dts.keys() | uuid_to_gts.keys())
    for uuid in uuids:
        log_id, timestamp_ns, _ = uuid
        args: Tuple[
            NDArrayFloat,
            NDArrayFloat,
            DetectionCfg,
            Optional[ArgoverseStaticMap],
            Optional[SE3],
        ]

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
        accumulate_args_list.append(args)

    logger.info("Starting evaluation ...")
    with mp.get_context("spawn").Pool(processes=n_jobs) as p:
        outputs: List[Tuple[NDArrayFloat, NDArrayFloat]] = p.starmap(
            accumulate, accumulate_args_list
        )

    dts_list, gts_list = zip(*outputs)

    METRIC_COLUMN_NAMES = (
        cfg.affinity_thresholds_m + TP_ERROR_COLUMNS + ("is_evaluated",)
    )
    dts_metrics: NDArrayFloat = np.concatenate(dts_list)
    gts_metrics: NDArrayFloat = np.concatenate(gts_list)
    dts.loc[:, METRIC_COLUMN_NAMES] = dts_metrics
    gts.loc[:, METRIC_COLUMN_NAMES] = gts_metrics

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
        dts: (N,14) Table of detections.
        gts: (M,15) Table of ground truth annotations.
        cfg: Detection configuration.

    Returns:
        The summary metrics.
    """
    # Sample recall values in the [0, 1] interval.
    recall_interpolated: NDArrayFloat = np.linspace(
        0, 1, cfg.num_recall_samples, endpoint=True
    )

    # Initialize the summary metrics.
    summary = pd.DataFrame(
        {s.value: cfg.metrics_defaults[i] for i, s in enumerate(tuple(MetricNames))},
        index=cfg.categories,
    )

    average_precisions = pd.DataFrame(
        {t: 0.0 for t in cfg.affinity_thresholds_m}, index=cfg.categories
    )
    for category in cfg.categories:
        # Find detections that have the current category.
        is_category_dts = dts["category"] == category

        # Only keep detections if they match the category and have NOT been filtered.
        is_valid_dts = np.logical_and(is_category_dts, dts["is_evaluated"])

        # Get valid detections and sort them in descending order.
        category_dts = (
            dts.loc[is_valid_dts]
            .sort_values(by="score", ascending=False)
            .reset_index(drop=True)
        )

        # Find annotations that have the current category.
        is_category_gts = gts["category"] == category

        # Compute number of ground truth annotations.
        num_gts = gts.loc[is_category_gts, "is_evaluated"].sum()

        # Cannot evaluate without ground truth information.
        if num_gts == 0:
            continue

        for affinity_threshold_m in cfg.affinity_thresholds_m:
            true_positives: NDArrayBool = (
                category_dts[affinity_threshold_m].astype(bool).to_numpy()
            )

            # Continue if there aren't any true positives.
            if len(true_positives) == 0:
                continue

            # Compute average precision for the current threshold.
            threshold_average_precision, _ = compute_average_precision(
                true_positives, recall_interpolated, num_gts
            )

            # Record the average precision.
            average_precisions.loc[
                category, affinity_threshold_m
            ] = threshold_average_precision

        mean_average_precisions: NDArrayFloat = (
            average_precisions.loc[category].to_numpy().mean()
        )

        # Select only the true positives for each instance.
        middle_idx = len(cfg.affinity_thresholds_m) // 2
        middle_threshold = cfg.affinity_thresholds_m[middle_idx]
        is_tp_t = category_dts[middle_threshold].to_numpy().astype(bool)

        # Initialize true positive metrics.
        tp_errors: NDArrayFloat = np.array(cfg.tp_normalization_terms)

        # Check whether any true positives exist under the current threshold.
        has_true_positives = np.any(is_tp_t)

        # If true positives exist, compute the metrics.
        if has_true_positives:
            tp_error_cols = [str(x.value) for x in TruePositiveErrorNames]
            tp_errors = category_dts.loc[is_tp_t, tp_error_cols].to_numpy().mean(axis=0)

        # Convert errors to scores.
        tp_scores = 1 - np.divide(tp_errors, cfg.tp_normalization_terms)

        # Compute Composite Detection Score (CDS).
        cds = mean_average_precisions * np.mean(tp_scores)
        summary.loc[category] = np.array([mean_average_precisions, *tp_errors, cds])

    # Return the summary.
    return summary


def evaluate_hierarchy(
    dts: pd.DataFrame,
    gts: pd.DataFrame,
    cfg: DetectionCfg,
    n_jobs: int = 8,
) -> pd.DataFrame:
    """Evaluate a set of detections against the ground truth annotations.

    Each sweep is processed independently, computing assignment between detections and ground truth annotations.

    Args:
        dts: (N,14) Table of detections.
        gts: (M,15) Table of ground truth annotations.
        cfg: Detection configuration.
        n_jobs: Number of jobs running concurrently during evaluation.

    Returns:
        (C+1,3) Pandas DataFrame for the Hierarchical AP at LCA = 0, 1, 2 for all classes. The last row reports the average over all classes.

    Raises:
        RuntimeError: If accumulation fails.
        ValueError: If ROI pruning is enabled but a dataset directory is not specified.
    """
    if cfg.eval_only_roi_instances and cfg.dataset_dir is None:
        raise ValueError(
            "ROI pruning has been enabled, but the dataset directory has not be specified. "
            "Please set `dataset_directory` to the split root, e.g. av2/sensor/val."
        )

    # Sort both the detections and annotations by lexicographic order for grouping.
    dts = dts.sort_values(list(UUID_COLUMNS))
    gts = gts.sort_values(list(UUID_COLUMNS))

    dts_categories = dts[list(CATEGORY_COLUMN)].to_numpy().astype(str)
    gts_categories = gts[list(CATEGORY_COLUMN)].to_numpy().astype(str)

    dts_pl = pl.from_pandas(dts)
    gts_pl = pl.from_pandas(gts)

    uuid_to_dts = {
        cast(Tuple[str, int], k): v[list(DTS_COLUMNS)].to_numpy().astype(np.float64)
        for k, v in dts_pl.partition_by(
            UUID_COLUMNS, maintain_order=True, as_dict=True
        ).items()
    }
    uuid_to_gts = {
        cast(Tuple[str, int], k): v[list(GTS_COLUMNS)].to_numpy().astype(np.float64)
        for k, v in gts_pl.partition_by(
            UUID_COLUMNS, maintain_order=True, as_dict=True
        ).items()
    }

    uuid_to_dts_cats = {
        cast(Tuple[str, int], k): v[list(CATEGORY_COLUMN)].to_numpy().astype(np.object_)
        for k, v in dts_pl.partition_by(
            UUID_COLUMNS, maintain_order=True, as_dict=True
        ).items()
    }

    uuid_to_gts_cats = {
        cast(Tuple[str, int], k): v[list(CATEGORY_COLUMN)].to_numpy().astype(np.object_)
        for k, v in gts_pl.partition_by(
            UUID_COLUMNS, maintain_order=True, as_dict=True
        ).items()
    }

    log_id_to_avm: Optional[Dict[str, ArgoverseStaticMap]] = None
    log_id_to_timestamped_poses: Optional[Dict[str, TimestampedCitySE3EgoPoses]] = None

    # Load maps and egoposes if roi-pruning is enabled.
    if cfg.eval_only_roi_instances and cfg.dataset_dir is not None:
        logger.info("Loading maps and egoposes ...")
        log_ids: List[str] = gts.loc[:, "log_id"].unique().tolist()
        log_id_to_avm, log_id_to_timestamped_poses = load_mapped_avm_and_egoposes(
            log_ids, cfg.dataset_dir
        )

    is_evaluated_args_list: List[
        Tuple[
            NDArrayFloat,
            NDArrayFloat,
            NDArrayObject,
            NDArrayObject,
            Tuple[str, int],
            DetectionCfg,
            Optional[ArgoverseStaticMap],
            Optional[SE3],
        ]
    ] = []
    uuids: List[Tuple[str, int]] = sorted(uuid_to_dts.keys() | uuid_to_gts.keys())
    for uuid in uuids:
        log_id, timestamp_ns = uuid
        sweep_dts: NDArrayFloat = np.zeros((0, 10))
        sweep_gts: NDArrayFloat = np.zeros((0, 10))

        sweep_dts_categories = np.zeros((0, 1), dtype=np.object_)
        sweep_gts_categories = np.zeros((0, 1), dtype=np.object_)

        if uuid in uuid_to_dts:
            sweep_dts = uuid_to_dts[uuid]
            sweep_dts_categories = uuid_to_dts_cats[uuid]

        if uuid in uuid_to_gts:
            sweep_gts = uuid_to_gts[uuid]
            sweep_gts_categories = uuid_to_gts_cats[uuid]

        args: Tuple[
            NDArrayFloat,
            NDArrayFloat,
            NDArrayObject,
            NDArrayObject,
            Tuple[str, int],
            DetectionCfg,
            Optional[ArgoverseStaticMap],
            Optional[SE3],
        ] = (
            sweep_dts,
            sweep_gts,
            sweep_dts_categories,
            sweep_gts_categories,
            uuid,
            cfg,
            None,
            None,
        )
        if log_id_to_avm is not None and log_id_to_timestamped_poses is not None:
            avm = log_id_to_avm[log_id]
            city_SE3_ego = log_id_to_timestamped_poses[log_id][int(timestamp_ns)]
            args = (
                sweep_dts,
                sweep_gts,
                sweep_dts_categories,
                sweep_gts_categories,
                uuid,
                cfg,
                avm,
                city_SE3_ego,
            )
        is_evaluated_args_list.append(args)

    logger.info("Starting evaluation ...")
    with mp.get_context("spawn").Pool(processes=n_jobs) as p:
        outputs: List[
            Tuple[
                NDArrayFloat,
                NDArrayFloat,
                NDArrayObject,
                NDArrayObject,
                Tuple[str, int],
            ]
        ] = p.starmap(is_evaluated, is_evaluated_args_list)

    dts_list: List[NDArrayFloat] = []
    gts_list: List[NDArrayFloat] = []
    dts_uuids_list: List[Tuple[str, int]] = []
    gts_uuids_list: List[Tuple[str, int]] = []
    dts_categories_list, gts_categories_list = [], []
    for (
        sweep_dts,
        sweep_gts,
        sweep_dts_categories,
        sweep_gts_categories,
        uuid,
    ) in outputs:
        dts_list.append(sweep_dts)
        gts_list.append(sweep_gts)
        dts_categories_list.append(sweep_dts_categories)
        gts_categories_list.append(sweep_gts_categories)

        num_dts = len(sweep_dts)
        num_gts = len(sweep_gts)
        dts_uuids_list.extend(num_dts * [uuid])
        gts_uuids_list.extend(num_gts * [uuid])

    dts_npy = np.concatenate(dts_list).astype(np.float64)
    gts_npy = np.concatenate(gts_list).astype(np.float64)
    dts_categories_npy = np.concatenate(dts_categories_list).astype(np.object_)
    gts_categories_npy = np.concatenate(gts_categories_list).astype(np.object_)
    dts_uuids_npy = np.array(dts_uuids_list)
    gts_uuids_npy = np.array(gts_uuids_list)

    accumulate_hierarchy_args_list: List[
        Tuple[
            NDArrayFloat,
            NDArrayFloat,
            NDArrayObject,
            NDArrayObject,
            NDArrayObject,
            NDArrayObject,
            str,
            Tuple[str, ...],
            str,
            DetectionCfg,
        ]
    ] = []
    for category in cfg.categories:
        index = HIERARCHY["FINEGRAIN"].index(category)
        for super_category, categories in HIERARCHY.items():
            lca_category = LCA[categories[index]]
            accumulate_hierarchy_args_list.append(
                (
                    dts_npy,
                    gts_npy,
                    dts_categories_npy,
                    gts_categories_npy,
                    dts_uuids_npy,
                    gts_uuids_npy,
                    category,
                    lca_category,
                    super_category,
                    cfg,
                )
            )

    logger.info("Starting evaluation ...")
    accumulate_outputs = []
    for accumulate_args in tqdm(accumulate_hierarchy_args_list):
        accumulate_outputs.append(accumulate_hierarchy(*accumulate_args))

    super_categories = list(HIERARCHY.keys())
    metrics = np.zeros((len(cfg.categories), len(HIERARCHY.keys())))
    for ap, category, super_category in accumulate_outputs:
        category_index = cfg.categories.index(category)
        super_category_index = super_categories.index(super_category)
        metrics[category_index][super_category_index] = round(ap, NUM_DECIMALS)

    metrics = pd.DataFrame(metrics, columns=LCA_COLUMNS, index=cfg.categories)
    return metrics
