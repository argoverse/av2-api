"""Argoverse 2.0 Scene Flow Evaluation
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Union, Dict, List, Iterator, Tuple, Optional
from pathlib import Path
import argparse
from rich.progress import track


CATEGORY_MAP = {"ANIMAL":0, "ARTICULATED_BUS":1, "BICYCLE":2, "BICYCLIST":3, "BOLLARD":4,
                "BOX_TRUCK":5, "BUS":6, "CONSTRUCTION_BARREL":7, "CONSTRUCTION_CONE":8, "DOG":9,
                "LARGE_VEHICLE":10, "MESSAGE_BOARD_TRAILER":11, "MOBILE_PEDESTRIAN_CROSSING_SIGN":12,
                "MOTORCYCLE":13, "MOTORCYCLIST":14, "OFFICIAL_SIGNALER":15, "PEDESTRIAN":16,
                "RAILED_VEHICLE":17, "REGULAR_VEHICLE":18, "SCHOOL_BUS":19, "SIGN":20,
                "STOP_SIGN":21, "STROLLER":22, "TRAFFIC_LIGHT_TRAILER":23, "TRUCK":24,
                "TRUCK_CAB":25, "VEHICULAR_TRAILER":26, "WHEELCHAIR":27, "WHEELED_DEVICE":28,
                "WHEELED_RIDER":29, "NONE": -1}

BACKGROUND_CATEGORIES = ['BOLLARD', 'CONSTRUCTION_BARREL', 'CONSTRUCTION_CONE', 
                         'MOBILE_PEDESTRIAN_CROSSING_SIGN', 'SIGN', 'STOP_SIGN']
PEDESTRIAN_CATEGORIES = ['PEDESTRIAN', 'STROLLER', 'WHEELCHAIR', 'OFFICIAL_SIGNALER']
SMALL_VEHICLE_CATEGORIES = ['BICYCLE', 'BICYCLIST', 'MOTORCYCLE', 'MOTORCYCLIST',
                            'WHEELED_DEVICE', 'WHEELED_RIDER']
VEHICLE_CATEGORIES = ['ARTICULATED_BUS', 'BOX_TRUCK', 'BUS', 'LARGE_VEHICLE', 'RAILED_VEHICLE',
                      'REGULAR_VEHICLE', 'SCHOOL_BUS', 'TRUCK', 'TRUCK_CAB',
                      'VEHICULAR_TRAILER', 'TRAFFIC_LIGHT_TRAILER', 'MESSAGE_BOARD_TRAILER']
ANIMAL_CATEGORIES = ['ANIMAL', 'DOG']

NO_CLASSES = {'All': [k for k in range(-1, 30)]}
FOREGROUND_BACKGROUND = {'Background': [-1],
                         'Foreground': [CATEGORY_MAP[k] for k in (BACKGROUND_CATEGORIES +
                                                                  PEDESTRIAN_CATEGORIES +
                                                                  SMALL_VEHICLE_CATEGORIES +
                                                                  VEHICLE_CATEGORIES +
                                                                  ANIMAL_CATEGORIES)]}
PED_CYC_VEH_ANI = {'Background': [-1],
                   'Object': [CATEGORY_MAP[k] for k in BACKGROUND_CATEGORIES],
                   'Pedestrian': [CATEGORY_MAP[k] for k in PEDESTRIAN_CATEGORIES],
                   'Small Vehicle': [CATEGORY_MAP[k] for k in SMALL_VEHICLE_CATEGORIES],
                   'Vehicle': [CATEGORY_MAP[k] for k in VEHICLE_CATEGORIES],
                   'Animal': [CATEGORY_MAP[k] for k in ANIMAL_CATEGORIES]}

FLOW_COLS = ['flow_tx_m', 'flow_ty_m', 'flow_ty_m']
def epe(pred, gt):
    return np.sqrt(np.sum((pred - gt) ** 2, axis=-1))

def accuracy(pred, gt, threshold):
    l2_norm = np.sqrt(np.sum((pred - gt) ** 2, axis=-1))
    gt_norm = np.sqrt(np.sum(gt * gt, axis=-1))
    relative_err = l2_norm / (gt_norm + 1e-7)
    error_lt_5 = (l2_norm < threshold).astype(bool)
    relative_err_lt_5 = (relative_err < threshold).astype(bool)
    return  (error_lt_5 | relative_err_lt_5).astype(np.float64)


def accuracy_strict(pred, gt):
    return accuracy(pred, gt, 0.05)


def accuracy_relax(pred, gt):
    return accuracy(pred, gt, 0.10)


def angle_error(pred, gt):
    unit_label = gt / (np.linalg.norm(gt, axis=-1, keepdims=True) + 1e-7)
    unit_pred = pred / (np.linalg.norm(pred, axis=-1, keepdims=True) + 1e-7)
    eps = 1e-7
    dot_product = np.clip(np.sum(unit_label * unit_pred, axis=-1), a_min=-1+eps, a_max=1-eps)
    dot_product[dot_product != dot_product] = 0  # Remove NaNs
    return np.arccos(dot_product)

def coutn(pred, gt):
    return torch.ones(len(pred))

def tp(pred, gt):
    return (pred & gt).sum()

def tn(pred, gt):
    return (~pred & ~gt).sum()

def fp(pred, gt):
    return (pred & ~gt).sum()

def fn(pred, gt):
    return (~pred & gt).sum()


FLOW_METRICS = {'EPE': epe, 'Accuracy Strict': accuracy_strict, 'Accuracy Relax': accuracy_relax,
                'Angle Error': angle_error}
SEG_METRICS = {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}


def metrics(pred_flow, pred_dynamic, gt, classes, dynamic, close, valid, object_classes):
    results = []
    pred_flow = pred_flow[valid].astype(np.float64)
    pred_dynamic = pred_dynamic[valid].astype(bool)
    gt = gt[valid].astype(np.float64)
    classes = classes[valid].astype(int) - 1
    dynamic = dynamic[valid].astype(bool)
    close = close[valid].astype(bool)
    
    for cls, class_idxs in object_classes.items():
        class_mask = classes == class_idxs[0]
        for i in class_idxs[1:]:
            class_mask = class_mask | (classes == i)

        for motion, m_mask in [('Dynamic', dynamic), ('Static', ~dynamic)]:
            for distance, d_mask in [('Close', close), ('Far', ~close)]:
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


def evaluate_directories(annotations_root: Path, predictions_root: Path):
    results = []
    annotation_files = list(annotations_root.rglob('*.feather'))
    for anno_file in track(annotation_files, description='Evaluating...'):
        gt = pd.read_feather(anno_file)
        name = anno_file.relative_to(annotations_root)
        pred_file = predictions_root / name
        if not pred_file.exists():
            print(f'Warning: {str(name)} missing')
            continue
        pred = pd.read_feather(pred_file)
        loss_breakdown = metrics(pred[FLOW_COLS].to_numpy(),
                                 pred['dynamic'].to_numpy(),
                                 gt[FLOW_COLS].to_numpy(),
                                 gt['classes'].to_numpy(),
                                 gt['dynamic'].to_numpy(),
                                 gt['close'].to_numpy(),
                                 gt['valid'].to_numpy(),
                                 FOREGROUND_BACKGROUND)
        results.extend([[str(name)] + bd for bd in loss_breakdown])
    df = pd.DataFrame(results, columns=['Example', 'Class', 'Motion', 'Distance', 'Count']
                      + list(FLOW_METRICS) + list(SEG_METRICS))
    return df
        

def results_to_dict(results_dataframe):
    output = {}
    grouped = results_dataframe.groupby(['Class', 'Motion', 'Distance'])
    for m in FLOW_METRICS.keys():
        avg = grouped.apply(lambda x: (x[m] * x.Count).sum() / x.Count.sum())
        for segment in avg.index:
            if segment[0] == 'Background' and segment[1] == 'Dynamic':
                continue
            name = m + '/' + '/'.join([str(i) for i in segment])
            output[name] = avg[segment]
    grouped = results_dataframe.groupby(['Class', 'Motion'])
    for m in FLOW_METRICS.keys():
        avg = grouped.apply(lambda x: (x[m] * x.Count).sum() / x.Count.sum())
        for segment in avg.index:
            if segment[0] == 'Background' and segment[1] == 'Dynamic':
                continue
            name = m + '/' + '/'.join([str(i) for i in segment])
            output[name] = avg[segment]
    output['Dynamic IoU'] = results_dataframe.TP.sum() / (results_dataframe.TP.sum()
                                                          + results_dataframe.FP.sum()
                                                          + results_dataframe.FN.sum())
    output['EPE 3-Way Average'] = (output['EPE/Foreground/Dynamic'] +
                                   output['EPE/Foreground/Static'] +
                                   output['EPE/Background/Static']) / 3

    
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='eval',
                                     description='Run a local scene flow evaluation on the val split')
    parser.add_argument('predictions_root', type=str,
                        help='path/to/predictions/')
    parser.add_argument('annotations_root', type=str,
                        help='path/to/annotation_files/')

    args = parser.parse_args()
    results_df = evaluate_directories(Path(args.annotations_root), Path(args.predictions_root))
    results_dict = results_to_dict(results_df)

    for metric in sorted(results_dict):
        print(f'{metric}: {results_dict[metric]:.3f}')
    breakpoint()
