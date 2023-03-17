"""Utilities for generating output for the scene flow challenge."""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from av2.torch.dataloaders.scene_flow import SceneFlowDataloader
from av2.torch.dataloaders.utils import Flow, Sweep
from av2.utils.typing import NDArrayBool, NDArrayFloat


def get_eval_subset(dataloader: SceneFlowDataloader) -> List[int]:
    """Return the indicies of the test set used for evaluation on the leaderboard."""
    return list(range(len(dataloader)))[::5]


def get_eval_point_mask(datum: Tuple[Sweep, Sweep, Flow]) -> NDArrayBool:
    """Return for a given sweep, a boolean mask indicating which points are evaluated on."""
    pcl = datum[0].lidar.as_tensor()
    is_close: NDArrayBool = ((pcl[:, 0].abs() <= 50) & (pcl[:, 1].abs() <= 50)).numpy().astype(bool)

    if datum[0].is_ground is None:
        raise ValueError("Must have ground annotations loaded to determine eval mask")

    return is_close & ~datum[0].is_ground.astype(bool)


def write_output_file(flow: NDArrayFloat, dynamic: NDArrayBool, sweep_uuid: Tuple[str, int], output_dir: Path) -> None:
    """Write an output predictions file in the correct format for submission.

    Args:
        flow: (N, 3) Flow predictions
        dynamic: (N,) Dynamic segmentation prediction
        sweep_uuid: The indentifier of the sweep being predicted (log_id, timestamp)
        output_dir: top level directory containing all predictions
    """
    output_log_dir = output_dir / sweep_uuid[0]
    output_log_dir.mkdir(exist_ok=True, parents=True)
    fx = flow[:, 0].astype(np.float16)
    fy = flow[:, 1].astype(np.float16)
    fz = flow[:, 2].astype(np.float16)
    output = pd.DataFrame({"flow_tx_m": fx, "flow_ty_m": fy, "flow_tz_m": fz, "dynamic": dynamic.astype(bool)})
    output.to_feather(output_log_dir / f"{sweep_uuid[1]}.feather")
