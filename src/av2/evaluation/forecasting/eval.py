
import argparse 
import pickle
import json
import numpy as np
from tqdm import tqdm 
from collections import defaultdict
import pandas as pd
import utils
from typing import Dict, List, Union, Tuple

def calc_ap(precision : np.ndarray, min_recall : float = 0, min_precision : float = 0) -> float:
    """ Calculated average precision. This is taken directly from the nuScenes evaluation dev-kit
    
    Args:
        precision: Precision at each recall threshold.
        min_recall: Minimum recall to consider.
        min_precision: Minimum precision to consider.
    
    Returns:
        ap:  Average Precision.
    """

    assert 0 <= min_precision < 1
    assert 0 <= min_recall <= 1

    prec = np.copy(precision)
    prec = prec[round(100 * min_recall) + 1:]  # Clip low recalls. +1 to exclude the min recall bin.
    prec -= min_precision  # Clip low precision
    prec[prec < 0] = 0
    return float(np.mean(prec)) / (1.0 - min_precision)

def evaluate(predictions : Dict, ground_truth : Dict, K : int) -> Dict:
    """ Compute Mean Forecasting AP, ADE, and FDE given predicted and ground truth forecasts.
    Args:
        predictions: All predicted trajectories for each log_id and timestep.
        ground_truth: All ground truth trajectories for each log_id and timestep.
        K: Number of future trajectories to consider when evaluating Forecastin AP, ADE and FDE (K=5 by default).
    
    Returns:
        res:  Dictionary of evaluation results.
     """
    class_names = utils.av2_classes
    class_velocity = utils.av2_velocity

    res  = {}
    velocity_profile = utils.velocity_profile

    for profile in velocity_profile:
        res[profile] = {}
        for cname in class_names:
            res[profile][cname] = {"mAP_F" : [], "ADE" : [], "FDE" : []}
    

    gt_agents = []
    pred_agents = []
    for seq_id in tqdm(ground_truth.keys()):
        for timestamp in ground_truth[seq_id].keys():   
            gt = [agent for agent in ground_truth[seq_id][timestamp]]

            if seq_id in predictions and timestamp in predictions[seq_id]:
                pred = [agent for agent in predictions[seq_id][timestamp]]
            else:
                pred = []
            
            for agent in gt:
                if agent["future_translation"].shape[0] < 1:
                    continue 
                
                agent["seq_id"] = seq_id
                agent["timestamp"] = timestamp
                agent["velocity"] = utils.agent_velocity(agent)
                agent["trajectory_type"] = utils.trajectory_type(agent, class_velocity)
                
                gt_agents.append(agent)
                
            for agent in pred:
                agent["seq_id"] = seq_id
                agent["timestamp"] = timestamp
                agent["velocity"] = utils.agent_velocity(agent)
                agent["trajectory_type"] = utils.trajectory_type(agent, class_velocity)
                
                pred_agents.append(agent)
    
        
    args_list = []
    for cname in class_names:
        cvel = class_velocity[cname]
        for profile in velocity_profile:
            mAP_F, mADE, mFDE = [], [], []
            for th in utils.dist_th:
                args = pred_agents, gt_agents, K, cname, profile, cvel, th
                
                args_list.append(args)
        
    outputs = [accumulate(*args) for args in tqdm(args_list)]
    
    for apf, ade, fde, cname, profile in outputs:     
        res[profile][cname]["mAP_F"].append(apf)
        res[profile][cname]["ADE"].append(ade)
        res[profile][cname]["FDE"].append(fde)
        
    for cname in class_names:
        for profile in velocity_profile:
            res[profile][cname]["mAP_F"] = round(np.mean(res[profile][cname]["mAP_F"]), 3)
            res[profile][cname]["ADE"] = round(np.mean(res[profile][cname]["ADE"]), 3)
            res[profile][cname]["FDE"] = round(np.mean(res[profile][cname]["FDE"]), 3)            

    return res
    
def accumulate(pred_agents : List, gt_agents : List, K : int, class_name : str, profile : str, velocity : float, threshold : float) -> Tuple:
    """ Performs matching between predicted and ground truth trajectories.
    
    Args:
        pred_agents: List of predicted trajectories for a given log_id and timestamp.
        gt_agents: List of ground truth trajectories for a given log_id and timestamp.
        K: Number of future trajectories to consider when evaluating Forecastin AP, ADE and FDE (K=5 by default).
        class_name: Match class name (e.g. car, pedestrian, bicycle) to determine if a trajectory is included in evaluation.
        profile: Match profile (e.g. static/linear/non-linear) to determine if a trajectory is included in evaluation.
        velocity: Average velocity for each class. This determines whether a trajectory is static/linear/non-linear 
        threshold: Match threshold to determine true positives and false positives.
    
    Returns:
        apf: Forecasting AP.
        ade: Average displacement error.
        fde: Final displacement error. 
        class_name: Match class name (e.g. car, pedestrian, bicycle) to determine if a trajectory is included in evaluation.
        profile: Match profile (e.g. static/linear/non-linear) to determine if a trajectory is included in evaluation.
    """

    def match(gt, pred, profile):
        if gt == profile:
            return True 

        if gt == "ignore" and pred == profile:
            return True 

        return False 

    pred = [agent for agent in pred_agents if agent["name"] == class_name]
    gt = [agent for agent in gt_agents if agent["name"] == class_name and agent["trajectory_type"] == profile]
    conf = [agent["detection_score"] for agent in pred]
    sortind = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(conf))][::-1]
    gt_agents_by_frame = defaultdict(list)
    for agent in gt:
        gt_agents_by_frame[(agent["seq_id"], agent["timestamp"])].append(agent)

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

        gt_agents_in_frame = gt_agents_by_frame[(pred_agent["seq_id"], pred_agent["timestamp"])]
        for gt_idx, gt_agent in enumerate(gt_agents_in_frame):
            if not (pred_agent["seq_id"], pred_agent["timestamp"], gt_idx) in taken:
            # Find closest match among ground truth boxes
                this_distance = utils.center_distance(gt_agent["current_translation"], pred_agent["current_translation"])
                if this_distance < min_dist:
                    min_dist = this_distance
                    match_gt_idx = gt_idx

        # If the closest match is close enough according to threshold we have a match!
        is_match = min_dist < threshold

        if is_match:
            taken.add((pred_agent["seq_id"], pred_agent["timestamp"], match_gt_idx))
            gt_match_agent = gt_agents_in_frame[match_gt_idx]
            
            gt_len = gt_match_agent["future_translation"].shape[0]
            forecast_match_th = [threshold + utils.forecast_scalar[i] * velocity for i in range(gt_len + 1)]
            
            if K == 1:
                ind = np.argmax(pred_agent["score"])
                forecast_dist = [utils.center_distance(gt_match_agent["future_translation"][i], pred_agent["prediction"][ind][i]) for i in range(gt_len)]
                forecast_match = [dist < th for dist, th in zip(forecast_dist, forecast_match_th[1:])]
                
                ade = np.mean(forecast_dist)
                fde = forecast_dist[-1]
                
            elif K == 5:
                forecast_dist, forecast_match = None, None
                ade, fde = np.inf, np.inf
                
                for ind in range(K):
                    curr_forecast_dist = [utils.center_distance(gt_match_agent["future_translation"][i], pred_agent["prediction"][ind][i]) for i in range(gt_len)]
                    curr_forecast_match = [dist < th for dist, th in zip(curr_forecast_dist, forecast_match_th[1:])]
                    
                    curr_ade = np.mean(curr_forecast_dist)
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
            pred_profiles.append("gent_apf.append(ignore")

        else:
            tp.append(False)
            fp.append(True)
            
            ind = np.argmax(pred_agent["score"])
            gt_profiles.append("ignore")
            pred_profiles.append(pred_agent["trajectory_type"][ind])
                        
    
    select = [match(gt, pred, profile) for gt, pred in zip(gt_profiles, pred_profiles)]
    tp = np.array(tp)[select]
    fp = np.array(fp)[select]
    
    if len(tp) == 0:
        return np.nan, np.nan, np.nan, class_name, profile

    true_pos = np.sum(tp)
    tp = np.cumsum(tp).astype(float)
    fp = np.cumsum(fp).astype(float)
    
    prec = tp / (fp + tp)
    rec = tp / float(npos)

    rec_interp = np.linspace(0, 1, utils.nelem)  # 101 steps, from 0% to 100% recall.
    prec = np.interp(rec_interp, rec, prec, right=0)

    apf = calc_ap(prec)
        
    return (apf, np.mean(agent_ade), np.mean(agent_fde), class_name, profile)
    
def convert_forecast_labels(labels : Dict) -> Dict:
    """ Converts the unified label format to a format that is easier to work with for forecasting evaluation. 
    
    Args:
        labels: Dictionary of labels.
    
    Returns:
        forecast_labels: Dictionary of labels in the forecasting format
    """

    def index_array_values(array_dict, index):
        return {
            k: v[index] if isinstance(v, np.ndarray) else v for k, v in array_dict.items()
        }

    def array_dict_iterator(array_dict, length):
        return (index_array_values(array_dict, i) for i in range(length))

    forecast_labels = {}
    for seq_id, frames in labels.items():
        frame_dict = {}
        for frame_idx, frame in enumerate(frames):
            forecast_instances = []
            for instance in array_dict_iterator(frame, len(frame["translation"])):
                future_translations = []
                for future_frame in frames[frame_idx + 1 : frame_idx + 1 + utils.num_timesteps]:
                    if instance["track_id"] not in future_frame["track_id"]:
                        break
                    future_translations.append(
                        future_frame["translation"][
                            future_frame["track_id"] == instance["track_id"]
                        ][0]
                    )
                if len(future_translations) == 0:
                    continue

                forecast_instances.append(
                    {
                        "current_translation": instance["translation"][:2],
                        "future_translation": np.array(future_translations)[:, :2],
                        "name": instance["name"],
                        "size": instance["size"],
                        "yaw": instance["yaw"],
                        "velocity": instance["velocity"][:2],
                        "label": instance["label"],
                    }
                )
            if forecast_instances:
                frame_dict[frame["timestamp_ns"]] = forecast_instances

        forecast_labels[seq_id] = frame_dict

    return forecast_labels 

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--predictions", default="sample/constant_velocity.pkl")
    argparser.add_argument("--ground_truth", default="sample/labels.pkl")
    argparser.add_argument("--out", default="sample/output.json")
    argparser.add_argument("--K", default=5)

    args = argparser.parse_args()
    predictions = pickle.load(open(args.predictions, "rb"))
    ground_truth = convert_forecast_labels(pickle.load(open(args.ground_truth, "rb")))

    res = evaluate(predictions, ground_truth, args.K)
    mAP_F = np.nanmean([metrics["mAP_F"] for traj_metrics in res.values() for metrics in traj_metrics.values()])
    ADE = np.nanmean([metrics["ADE"] for traj_metrics in res.values() for metrics in traj_metrics.values()])
    FDE = np.nanmean([metrics["FDE"] for traj_metrics in res.values() for metrics in traj_metrics.values()])
    res["mean_mAP_F"] = mAP_F
    res["mean_ADE"] = ADE
    res["mean_FDE"] = FDE
    print(res)

    with open(args.out, "w") as f:
        json.dump(res, f, indent=4)
