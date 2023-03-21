"""Utility program for producing submission mask files."""

import zipfile
from pathlib import Path
from typing import Tuple

import click
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
    s0: Sweep,
    s1: Sweep,
    s1_SE3_s0: Se3,
    output_root: Path,
) -> None:
    """Write an annotation file.

    Args:
        s0: The first sweep of the pair.
        s1: The second sweep of the pair.
        s1_SE3_s0: The relative ego-motion between the two sweeps.
        output_root: The top levevel directory to store the output in.
    """
    mask = compute_eval_point_mask((s0, s1, s1_SE3_s0, None))

    output = pd.DataFrame({"mask": mask.numpy().astype(bool)})

    log, timestamp_ns = s0.sweep_uuid

    output_dir = output_root / log
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"{timestamp_ns}.feather"
    output.to_feather(output_file)


@click.command()
@click.argument("output_dir", type=str)
@click.argument("data_dir", type=str)
@click.option(
    "--name",
    type=str,
    help="the data should be located in <data_dir>/<name>/sensor/<split>",
    default="av2",
)
@click.option(
    "--split",
    help="the data should be located in <data_dir>/<name>/sensor/<split>",
    default="val",
    type=click.Choice(["test", "val"]),
)
def make_mask_files(output_file: str, data_dir: str, name: str, split: str) -> None:
    """Output the point masks for submission to the leaderboard.

    Args:
        output_file: Path to output files.
        data_dir: Path to input data.
        name: Name of the dataset (e.g. av2).
        split: Split to make masks for.
    """
    data_loader = SceneFlowDataloader(Path(data_dir), name, split)

    output_root = Path(output_dir)
    output_root.mkdir(exist_ok=True)

    eval_inds = get_eval_subset(data_loader)

    with ZipFile(Path(output_file), "w") as maskzip:
        for i in track(eval_inds):
            sweep_0, sweep_1, ego, _ = data_loader[i]
            write_mask(sweep_0, sweep_1, ego, output_root)


if __name__ == "__main__":
    make_mask_files()
