"""Utility program for producing minimnal annotation files used for evaluation on the val and test splits."""

from pathlib import Path
from typing import Final, Tuple

import click
import numpy as np
import pandas as pd
from rich.progress import track

from av2.evaluation.scene_flow.utils import get_eval_point_mask, get_eval_subset
from av2.torch.data_loaders.scene_flow import SceneFlowDataloader
from av2.utils.typing import NDArrayBool, NDArrayFloat, NDArrayInt

CLOSE_DISTANCE_THRESHOLD: Final = 35.0


def write_annotation(
    category_indices: NDArrayInt,
    is_close: NDArrayBool,
    is_dynamic: NDArrayBool,
    is_valid: NDArrayBool,
    flow: NDArrayFloat,
    sweep_uuid: Tuple[str, int],
    output_dir: Path,
) -> None:
    """Write an annotation file.

    Args:
        category_indices: Category label indices.
        is_close: Close (inside 70 meter box) labels.
        is_dynamic: Dynamic labels.
        is_valid: Valid flow labels.
        flow: Flow labels.
        sweep_uuid: Log id and timestamp_ns of the sweep.
        output_dir: Top level directory to store the output in.
    """
    output = pd.DataFrame(
        {
            "category_indices": category_indices.astype(np.uint8),
            "is_close": is_close.astype(bool),
            "is_dynamic": is_dynamic.astype(bool),
            "is_valid": is_valid.astype(bool),
            "flow_tx_m": flow[:, 0].astype(np.float16),
            "flow_ty_m": flow[:, 1].astype(np.float16),
            "flow_tz_m": flow[:, 2].astype(np.float16),
        }
    )

    log_id, timestamp_ns = sweep_uuid

    output_subdir = output_dir / log_id
    output_subdir.mkdir(exist_ok=True)
    output_file = output_subdir / f"{timestamp_ns}.feather"
    output.to_feather(output_file)


def make_annotation_files(
    output_dir: str, mask_file: str, data_dir: str, name: str, split: str
) -> None:
    """Create annotation files for running the evaluation.

    Args:
        output_dir: Path to output directory.
        data_dir: Path to input data.
        mask_file: Archive of submission masks.
        name: Name of the dataset (e.g. av2).
        split: Split to make annotations for.

    Raises:
        ValueError: If the dataset does not have annotations.
    """
    data_loader = SceneFlowDataloader(Path(data_dir), name, "val")

    output_root = Path(output_dir)
    output_root.mkdir(exist_ok=True)

    eval_inds = get_eval_subset(data_loader)
    for i in track(eval_inds):
        sweep_0, _, _, flow_labels = data_loader[i]
        if flow_labels is None:
            raise ValueError("Missing flow annotations!")

        mask = get_eval_point_mask(sweep_0.sweep_uuid, Path(mask_file))

        flow = flow_labels.flow[mask].numpy().astype(np.float16)
        is_valid = flow_labels.is_valid[mask].numpy().astype(bool)
        category_indices = flow_labels.category_indices[mask].numpy().astype(np.uint8)
        is_dynamic = flow_labels.is_dynamic[mask].numpy().astype(bool)

        pc = sweep_0.lidar.as_tensor()[mask, :3].numpy()
        is_close = np.logical_and.reduce(
            np.abs(pc[:, :2]) <= CLOSE_DISTANCE_THRESHOLD, axis=1
        ).astype(bool)

        write_annotation(
            category_indices,
            is_close,
            is_dynamic,
            is_valid,
            flow,
            sweep_0.sweep_uuid,
            output_root,
        )


@click.command()
@click.argument("output_dir", type=str)
@click.argument("data_dir", type=str)
@click.argument("mask_file", type=str)
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
def _make_annotation_files_entry(
    output_dir: str, mask_file: str, data_dir: str, name: str, split: str
) -> None:
    """Entry point for make_annotation_files."""
    make_annotation_files(output_dir, mask_file, data_dir, name, split)


if __name__ == "__main__":
    _make_annotation_files_entry()
