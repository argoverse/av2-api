"""Utilities for generating output for the scene flow challenge."""

from pathlib import Path
from typing import Final, List, Optional, Tuple
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch
from kornia.geometry.liegroup import Se3
from torch import BoolTensor

from av2.torch.data_loaders.scene_flow import SceneFlowDataloader
from av2.torch.structures.flow import Flow
from av2.torch.structures.sweep import Sweep
from av2.utils.typing import NDArrayBool, NDArrayFloat

_EVAL_ROOT: Final = Path(__file__).resolve().parent


def get_eval_subset(dataloader: SceneFlowDataloader) -> List[int]:
    """Return the indices of the test set used for evaluation on the leaderboard."""
    return list(range(len(dataloader)))[::5]


def get_eval_point_mask(sweep_uuid: Tuple[str, int], mask_file: Path) -> BoolTensor:
    """Retrieve for a given sweep, a boolean mask indicating which points are evaluated on.

    Args:
        sweep_uuid: The uuid of the first sweep in the pair to retrieve the mask for.
        mask_file: Archive of submission masks.

    Returns:
        The submission mask for that pair.
    """
    with ZipFile(mask_file) as masks:
        log_id, timestamp_ns = sweep_uuid
        mask = (
            pd.read_feather(masks.open(f"{log_id}/{timestamp_ns}.feather"))
            .to_numpy()
            .astype(bool)
        )

    return BoolTensor(torch.from_numpy(mask).squeeze())


def compute_eval_point_mask(
    datum: Tuple[Sweep, Sweep, Se3, Optional[Flow]]
) -> BoolTensor:
    """Compute for a given sweep, a boolean mask indicating which points are evaluated on.

    Note: This should NOT BE USED FOR CREATING SUBMISSIONS use get_eval_point_mask to ensure consistency.

    Args:
        datum: Tuple returned from a `SceneFlowDataloader` to compute the mask for.

    Returns:
        A mask indicating roughly which points will be evauated on.

    Raises:
        ValueError: if datum does not have ground annotations.
    """
    pcl = datum[0].lidar.as_tensor()[:, :3]
    is_close = torch.logical_and(
        (pcl[:, 0].abs() <= 50), (pcl[:, 1].abs() <= 50)
    ).bool()

    if datum[0].is_ground is None:
        raise ValueError("Must have ground annotations loaded to determine eval mask")
    not_ground = torch.logical_not(datum[0].is_ground)
    return BoolTensor(torch.logical_and(is_close, not_ground))


def write_output_file(
    flow: NDArrayFloat,
    is_dynamic: NDArrayBool,
    sweep_uuid: Tuple[str, int],
    output_dir: Path,
) -> None:
    """Write an output predictions file in the correct format for submission.

    Args:
        flow: (N,3) Flow predictions.
        is_dynamic: (N,) Dynamic segmentation prediction.
        sweep_uuid: Identifier of the sweep being predicted (log_id, timestamp_ns).
        output_dir: Top level directory containing all predictions.
    """
    output_log_dir = output_dir / sweep_uuid[0]
    output_log_dir.mkdir(exist_ok=True, parents=True)
    fx_m = flow[:, 0].astype(np.float16)
    fy_m = flow[:, 1].astype(np.float16)
    fz_m = flow[:, 2].astype(np.float16)
    output = pd.DataFrame(
        {
            "flow_tx_m": fx_m,
            "flow_ty_m": fy_m,
            "flow_tz_m": fz_m,
            "is_dynamic": is_dynamic.astype(bool),
        }
    )
    output.to_feather(output_log_dir / f"{sweep_uuid[1]}.feather")
