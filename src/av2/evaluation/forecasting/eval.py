"""Argoverse Forecasting evaluation.

Evaluation Metrics:
    mAP: see https://arxiv.org/abs/2203.16297
"""

import itertools
import json
import pickle
from collections import defaultdict
from pathlib import Path
from pprint import pprint
from typing import Any, Dict, List, Optional, Tuple, cast

import click
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from av2.evaluation import NUM_RECALL_SAMPLES
from av2.evaluation.detection.utils import (
    compute_objects_in_roi_mask,
    load_mapped_avm_and_egoposes,
)
from av2.evaluation.forecasting import constants, utils
from av2.evaluation.forecasting.constants import (
    AV2_CATEGORIES,
    CATEGORY_TO_VELOCITY_M_PER_S,
    DISTANCE_THRESHOLDS_M,
    MAX_DISPLACEMENT_M,
    VELOCITY_TYPES,
)
from av2.utils.typing import NDArrayFloat

from ..typing import ForecastSequences, Sequences


def evaluate(
    predictions: ForecastSequences,
    raw_ground_truth: Sequences,
    top_k: int,
    max_range_m: int,
    dataset_dir: Optional[str],
) -> Dict[str, Any]:
    """Compute Mean Forecasting AP, ADE, and FDE given predicted and ground truth forecasts.

    Args:
        predictions: All predicted trajectories for each log_id and timestep.
        raw_ground_truth: All ground truth trajectories for each log_id and timestep.
        top_k: Top K evaluation.
        max_range_m: Maximum evaluation range.
        dataset_dir: Path to dataset. Required for ROI pruning.

    Returns:
        Dictionary of evaluation results.
    """
    ground_truth: ForecastSequences = convert_forecast_labels(raw_ground_truth)
    ground_truth = filter_max_dist(ground_truth, max_range_m)

    utils.annotate_frame_metadata(predictions, ground_truth, ["ego_translation_m"])
    predictions = filter_max_dist(predictions, max_range_m)

    if dataset_dir is not None:
        ground_truth = filter_drivable_area(ground_truth, dataset_dir)
        predictions = filter_drivable_area(predictions, dataset_dir)

    results: Dict[str, Any] = {}
    for velocity_type in VELOCITY_TYPES:
        results[velocity_type] = {}
        for category in AV2_CATEGORIES:
            results[velocity_type][category] = {"mAP_F": [], "ADE": [], "FDE": []}

    gt_agents = []
    pred_agents = []
    for seq_id in tqdm(ground_truth.keys()):
        for timestamp_ns in ground_truth[seq_id].keys():
            gt = [agent for agent in ground_truth[seq_id][timestamp_ns]]

            if seq_id in predictions and timestamp_ns in predictions[seq_id]:
                pred = [agent for agent in predictions[seq_id][timestamp_ns]]
            else:
                pred = []

            for agent in gt:
                if agent["future_translation_m"].shape[0] < 1:
                    continue

                agent["seq_id"] = seq_id
                agent["timestamp_ns"] = timestamp_ns
                agent["velocity_m_per_s"] = utils.agent_velocity_m_per_s(agent)
                agent["trajectory_type"] = utils.trajectory_type(
                    agent, CATEGORY_TO_VELOCITY_M_PER_S
                )

                gt_agents.append(agent)

            for agent in pred:
                agent["seq_id"] = seq_id
                agent["timestamp_ns"] = timestamp_ns
                agent["velocity_m_per_s"] = utils.agent_velocity_m_per_s(agent)
                agent["trajectory_type"] = utils.trajectory_type(
                    agent, CATEGORY_TO_VELOCITY_M_PER_S
                )

                pred_agents.append(agent)

    outputs = []
    for category, velocity_type, th in tqdm(
        list(itertools.product(AV2_CATEGORIES, VELOCITY_TYPES, DISTANCE_THRESHOLDS_M))
    ):
        category_velocity_m_per_s = CATEGORY_TO_VELOCITY_M_PER_S[category]
        outputs.append(
            accumulate(
                pred_agents,
                gt_agents,
                top_k,
                category,
                velocity_type,
                category_velocity_m_per_s,
                th,
            )
        )

    for apf, ade, fde, category, velocity_type, threshold in outputs:
        results[velocity_type][category]["mAP_F"].append(apf)

        if threshold == constants.TP_THRESHOLD_M:
            results[velocity_type][category]["ADE"].append(ade)
            results[velocity_type][category]["FDE"].append(fde)

    for category in AV2_CATEGORIES:
        for velocity_type in VELOCITY_TYPES:
            results[velocity_type][category]["mAP_F"] = round(
                np.mean(results[velocity_type][category]["mAP_F"]),
                constants.NUM_DECIMALS,
            )
            results[velocity_type][category]["ADE"] = round(
                np.mean(results[velocity_type][category]["ADE"]), constants.NUM_DECIMALS
            )
            results[velocity_type][category]["FDE"] = round(
                np.mean(results[velocity_type][category]["FDE"]), constants.NUM_DECIMALS
            )

    return results


def accumulate(
    pred_agents: List[Dict[str, Any]],
    gt_agents: List[Dict[str, Any]],
    top_k: int,
    class_name: str,
    profile: str,
    velocity: float,
    threshold: float,
) -> Tuple[Any, Any, Any, str, str, float]:
    """Perform matching between predicted and ground truth trajectories.

    Args:
        pred_agents: List of predicted trajectories for a given log_id and timestamp_ns.
        gt_agents: List of ground truth trajectories for a given log_id and timestamp_ns.
        top_k: Number of future trajectories to consider when evaluating Forecastin AP, ADE and FDE (K=5 by default).
        class_name: Match class name (e.g. car, pedestrian, bicycle) to determine if a trajectory is included
            in evaluation.
        profile: Match profile (e.g. static/linear/non-linear) to determine if a trajectory is included in evaluation.
        velocity: Average velocity for each class. This determines whether a trajectory is static/linear/non-linear
        threshold: Match threshold to determine true positives and false positives.

    Returns:
        apf: Forecasting AP.
        ade: Average displacement error.
        fde: Final displacement error.
        class_name: Match class name (e.g. car, pedestrian, bicycle) to determine if a trajectory is included
            in evaluation.
        profile: Match profile (e.g. static/linear/non-linear) to determine if a trajectory is included in evaluation.
    """

    def match(gt: str, pred: str, profile: str) -> bool:
        return gt == profile or gt == "ignore" and pred == profile

    pred = [agent for agent in pred_agents if agent["name"] == class_name]
    gt = [
        agent
        for agent in gt_agents
        if agent["name"] == class_name and agent["trajectory_type"] == profile
    ]
    conf = [agent["detection_score"] for agent in pred]
    sortind = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(conf))][::-1]
    gt_agents_by_frame = defaultdict(list)
    for agent in gt:
        gt_agents_by_frame[(agent["seq_id"], agent["timestamp_ns"])].append(agent)

    npos = len(gt)
    # ---------------------------------------------
    # Match and accumulate match data.
    # ---------------------------------------------

    gt_profiles, pred_profiles, agent_ade, agent_fde, tp, fp = [], [], [], [], [], []
    taken = set()  # Initially no gt bounding box is matched.
    for ind in sortind:
        pred_agent = pred[ind]
        min_dist = np.inf
        match_gt_idx = None

        gt_agents_in_frame = gt_agents_by_frame[
            (pred_agent["seq_id"], pred_agent["timestamp_ns"])
        ]
        for gt_idx, gt_agent in enumerate(gt_agents_in_frame):
            if not (pred_agent["seq_id"], pred_agent["timestamp_ns"], gt_idx) in taken:
                # Find closest match among ground truth boxes
                this_distance = utils.center_distance(
                    gt_agent["current_translation_m"],
                    pred_agent["current_translation_m"],
                )
                if this_distance < min_dist:
                    min_dist = this_distance
                    match_gt_idx = gt_idx

        # If the closest match is close enough according to threshold we have a match!
        is_match = min_dist < threshold

        if is_match and match_gt_idx is not None:
            taken.add((pred_agent["seq_id"], pred_agent["timestamp_ns"], match_gt_idx))
            gt_match_agent = gt_agents_in_frame[match_gt_idx]

            gt_len = gt_match_agent["future_translation_m"].shape[0]
            forecast_match_th = [
                threshold + constants.FORECAST_SCALAR[i] * velocity
                for i in range(gt_len + 1)
            ]

            if top_k == 1:
                ind = cast(int, np.argmax(pred_agent["score"]))
                forecast_dist = [
                    utils.center_distance(
                        gt_match_agent["future_translation_m"][i],
                        pred_agent["prediction_m"][ind][i],
                    )
                    for i in range(gt_len)
                ]
                forecast_match = [
                    dist < th for dist, th in zip(forecast_dist, forecast_match_th[1:])
                ]

                ade = cast(float, np.mean(forecast_dist))
                fde = forecast_dist[-1]

            elif top_k == 5:
                forecast_dist, forecast_match = None, [False]
                ade, fde = np.inf, np.inf

                for ind in range(top_k):
                    curr_forecast_dist = [
                        utils.center_distance(
                            gt_match_agent["future_translation_m"][i],
                            pred_agent["prediction_m"][ind][i],
                        )
                        for i in range(gt_len)
                    ]
                    curr_forecast_match = [
                        dist < th
                        for dist, th in zip(curr_forecast_dist, forecast_match_th[1:])
                    ]

                    curr_ade = cast(float, np.mean(curr_forecast_dist))
                    curr_fde = curr_forecast_dist[-1]

                    if curr_ade < ade:
                        forecast_dist = curr_forecast_dist
                        forecast_match = curr_forecast_match
                        ade = curr_ade
                        fde = curr_fde

            agent_ade.append(ade)
            agent_fde.append(fde)
            tp.append(forecast_match[-1])
            fp.append(not forecast_match[-1])

            gt_profiles.append(profile)
            pred_profiles.append("ignore")

        else:
            tp.append(False)
            fp.append(True)

            ind = cast(int, np.argmax(pred_agent["score"]))
            gt_profiles.append("ignore")
            pred_profiles.append(pred_agent["trajectory_type"][ind])

    select = [match(gt, pred, profile) for gt, pred in zip(gt_profiles, pred_profiles)]
    tp_array = np.array(tp)[select]
    fp_array = np.array(fp)[select]

    if len(gt) == 0:
        return (np.nan, np.nan, np.nan, class_name, profile, threshold)

    if sum(tp_array) == 0:
        return (
            cast(float, 0),
            cast(float, MAX_DISPLACEMENT_M),
            cast(float, MAX_DISPLACEMENT_M),
            class_name,
            profile,
            threshold,
        )

    tp_array = np.cumsum(tp_array).astype(float)
    fp_array = np.cumsum(fp_array).astype(float)

    prec = tp_array / (fp_array + tp_array)
    rec = tp_array / float(npos)

    rec_interp = np.linspace(
        0, 1, NUM_RECALL_SAMPLES
    )  # 101 steps, from 0% to 100% recall.
    apf = np.mean(np.interp(rec_interp, rec, prec, right=0))

    return (
        cast(float, apf),
        min(cast(float, np.mean(agent_ade)), MAX_DISPLACEMENT_M),
        min(cast(float, np.mean(agent_fde)), MAX_DISPLACEMENT_M),
        class_name,
        profile,
        threshold,
    )


def convert_forecast_labels(labels: Any) -> Any:
    """Convert the unified label format to a format that is easier to work with for forecasting evaluation.

    Args:
        labels: Dictionary of labels.

    Returns:
        forecast_labels, dictionary of labels in the forecasting format.
    """
    forecast_labels = {}
    for seq_id, frames in labels.items():
        frame_dict = {}
        for frame_idx, frame in enumerate(frames):
            forecast_instances = []
            for instance in utils.array_dict_iterator(
                frame, len(frame["translation_m"])
            ):
                future_translations: Any = []
                for future_frame in frames[
                    frame_idx + 1 : frame_idx + 1 + constants.NUM_TIMESTEPS
                ]:
                    if instance["track_id"] not in future_frame["track_id"]:
                        break
                    future_translations.append(
                        future_frame["translation_m"][
                            future_frame["track_id"] == instance["track_id"]
                        ][0]
                    )

                if len(future_translations) == 0:
                    continue

                forecast_instances.append(
                    {
                        "current_translation_m": instance["translation_m"][:2],
                        "ego_translation_m": instance["ego_translation_m"][:2],
                        "future_translation_m": np.array(future_translations)[:, :2],
                        "name": instance["name"],
                        "size": instance["size"],
                        "yaw": instance["yaw"],
                        "velocity_m_per_s": instance["velocity_m_per_s"][:2],
                        "label": instance["label"],
                    }
                )
            if forecast_instances:
                frame_dict[frame["timestamp_ns"]] = forecast_instances

        forecast_labels[seq_id] = frame_dict

    return forecast_labels


def filter_max_dist(
    forecasts: ForecastSequences, max_range_m: int
) -> ForecastSequences:
    """Remove all tracks that are beyond `max_range_m`.

    Args:
        forecasts: Dictionary of tracks.
        max_range_m: maximum distance from ego-vehicle.

    Returns:
        Dictionary of tracks.
    """
    for seq_id in forecasts.keys():
        for timestamp_ns in forecasts[seq_id].keys():
            keep_forecasts = [
                agent
                for agent in forecasts[seq_id][timestamp_ns]
                if "ego_translation_m" in agent
                and np.linalg.norm(
                    agent["current_translation_m"] - agent["ego_translation_m"]
                )
                < max_range_m
            ]
            forecasts[seq_id][timestamp_ns] = keep_forecasts

    return forecasts


def yaw_to_quaternion3d(yaw: float) -> NDArrayFloat:
    """Convert a rotation angle in the xy plane (i.e. about the z axis) to a quaternion.

    Args:
        yaw: angle to rotate about the z-axis, representing an Euler angle, in radians
    Returns:
        array w/ quaternion coefficients (qw,qx,qy,qz) in scalar-first order, per Argoverse convention.
    """
    qx, qy, qz, qw = Rotation.from_euler(seq="z", angles=yaw, degrees=False).as_quat()
    return np.array([qw, qx, qy, qz])


def filter_drivable_area(
    forecasts: ForecastSequences, dataset_dir: str
) -> ForecastSequences:
    """Convert the unified label format to a format that is easier to work with for forecasting evaluation.

    Args:
        forecasts: Dictionary of tracks.
        dataset_dir: Dataset root directory.

    Returns:
        forecasts: Dictionary of tracks.
    """
    log_ids = list(forecasts.keys())
    log_id_to_avm, log_id_to_timestamped_poses = load_mapped_avm_and_egoposes(
        log_ids, Path(dataset_dir)
    )

    for log_id in log_ids:
        avm = log_id_to_avm[log_id]

        for timestamp_ns in forecasts[log_id]:
            city_SE3_ego = log_id_to_timestamped_poses[log_id][int(timestamp_ns)]

            translation_m, size, quat = [], [], []

            if len(forecasts[log_id][timestamp_ns]) == 0:
                continue

            for box in forecasts[log_id][timestamp_ns]:
                translation_m.append(
                    box["current_translation_m"] - box["ego_translation_m"]
                )
                size.append(box["size"])
                quat.append(yaw_to_quaternion3d(box["yaw"]))

            score = np.ones((len(translation_m), 1))
            boxes = np.concatenate(
                [
                    np.array(translation_m),
                    np.array(size),
                    np.array(quat),
                    np.array(score),
                ],
                axis=1,
            )

            is_evaluated = compute_objects_in_roi_mask(boxes, city_SE3_ego, avm)
            forecasts[log_id][timestamp_ns] = list(
                np.array(forecasts[log_id][timestamp_ns])[is_evaluated]
            )

    return forecasts


@click.command()
@click.option("--predictions", required=True, help="Predictions PKL file")
@click.option("--ground_truth", required=True, help="Ground Truth PKL file")
@click.option("--max_range_m", default=50, type=int, help="Max evaluation range")
@click.option("--top_k", default=5, type=int, help="Top K evaluation metric")
@click.option(
    "--dataset_dir",
    default=None,
    help="Path to dataset split (e.g. /data/Sensor/val). Required for ROI pruning",
)
@click.option("--out", required=True, help="Output PKL file")
def runner(
    predictions: str,
    ground_truth: str,
    max_range_m: int,
    dataset_dir: Any,
    top_k: int,
    out: str,
) -> None:
    """Standalone evaluation function."""
    predictions2 = pickle.load(open(predictions, "rb"))
    ground_truth2 = pickle.load(open(ground_truth, "rb"))

    res = evaluate(predictions2, ground_truth2, top_k, max_range_m, dataset_dir)

    mAP_F = np.nanmean(
        [
            metrics["mAP_F"]
            for traj_metrics in res.values()
            for metrics in traj_metrics.values()
        ]
    )
    ADE = np.nanmean(
        [
            metrics["ADE"]
            for traj_metrics in res.values()
            for metrics in traj_metrics.values()
        ]
    )
    FDE = np.nanmean(
        [
            metrics["FDE"]
            for traj_metrics in res.values()
            for metrics in traj_metrics.values()
        ]
    )
    res["mean_mAP_F"] = mAP_F
    res["mean_ADE"] = ADE
    res["mean_FDE"] = FDE
    pprint(res)

    with open(out, "w") as f:
        json.dump(res, f, indent=4)


if __name__ == "__main__":
    runner()
