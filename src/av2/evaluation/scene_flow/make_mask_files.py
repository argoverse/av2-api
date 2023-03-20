"""Utility program for producing submission mask files."""

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from kornia.geometry.liegroup import Se3
from rich.progress import track

from av2.evaluation.scene_flow.utils import compute_eval_point_mask, get_eval_subset
from av2.torch.data_loaders.scene_flow import SceneFlowDataloader
from av2.torch.structures.flow import Flow
from av2.torch.structures.sweep import Sweep
from av2.utils.typing import NDArrayBool, NDArrayFloat, NDArrayInt


def write_mask(
    sweep_0: Sweep,
    sweep_1: Sweep,
    ego: Se3,
    output_root: Path,
) -> None:
    """Write an annotation file.

    Args:
        sweep_0: The first sweep of the pair.
        sweep_1: The second sweep of the pair.
        ego: The relative ego-motion between the two sweeps.
        output_root: The top levevel directory to store the output in.
    """
    mask = compute_eval_point_mask((sweep_0, sweep_1, ego, None))

    output = pd.DataFrame({"mask": mask.numpy().astype(bool)})

    log, ts = sweep_0.sweep_uuid

    output_dir = output_root / log
    output_dir.mkdir(exist_ok=True)
    output_file = (output_dir / str(ts)).with_suffix(".feather")
    output.to_feather(output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "make_mask_files",
        description="Make a directory of feather files storing masks for submission",
    )
    parser.add_argument("output_root", type=str, help="path/to/output/")
    parser.add_argument("data_root", type=str, help="root/path/to/data")
    parser.add_argument(
        "--name", type=str, default="av2", help="the data should be located in <data_root>/<name>/sensor/<split>"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test"],
        help="the data should be located in <data_root>/<name>/sensor/<split>",
    )

    args = parser.parse_args()

    dl = SceneFlowDataloader(args.data_root, args.name, args.split)

    output_root = Path(args.output_root)
    output_root.mkdir(exist_ok=True)

    eval_inds = get_eval_subset(dl)
    for i in track(eval_inds):
        sweep_0, sweep_1, ego, _ = dl[i]
        write_mask(sweep_0, sweep_1, ego, output_root)
