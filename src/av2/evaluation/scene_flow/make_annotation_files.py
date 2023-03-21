"""Utility program for producing minimnal annotation files used for evaluation on the val and test splits."""

import argparse
from pathlib import Path
from typing import Final, Tuple

import numpy as np
import pandas as pd
from rich.progress import track

from av2.evaluation.scene_flow.utils import get_eval_point_mask, get_eval_subset
from av2.torch.data_loaders.scene_flow import SceneFlowDataloader
from av2.utils.typing import NDArrayBool, NDArrayFloat, NDArrayInt

CLOSE_DISTANCE_THRESHOLD: Final = 35


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
        is_close: Close (inside 70m box) labels.
        is_dynamic: Dynamic labels.
        is_valid: Valid flow labels.
        flow: Flow labels.
        sweep_uuid: Log and timestamp of the sweep.
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

    output_dir = output_root / log_id
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"{timestamp_ns}.feather"
    output.to_feather(output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "make_annotation_files",
        description="Make a directory of feather files storing " "just enough info to run a scene flow evaluation",
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
        choices=("val", "test"),
        help="the data should be located in <data_root>/<name>/sensor/<split>",
    )

    args = parser.parse_args()

    data_loader = SceneFlowDataloader(args.data_root, args.name, "val")

    output_root = Path(args.output_root)
    output_root.mkdir(exist_ok=True)

    eval_inds = get_eval_subset(data_loader)
    for i in track(eval_inds):
        datum = data_loader[i]
        if datum[3] is None:
            raise ValueError("Missing flow annotations!")

        mask = get_eval_point_mask(datum[0].sweep_uuid, split=args.split)

        flow = datum[3].flow[mask].numpy().astype(np.float16)
        is_valid = datum[3].is_valid[mask].numpy().astype(bool)
        category_indices = datum[3].category_indices[mask].numpy().astype(np.uint8)
        is_dynamic = datum[3].is_dynamic[mask].numpy().astype(bool)

        pc = datum[0].lidar.as_tensor()[mask, :3].numpy()
        is_close = np.logical_and.reduce(np.abs(pc[:, :2]) <= CLOSE_DISTANCE_THRESHOLD, axis=1).astype(bool)

        write_annotation(category_indices, is_close, is_dynamic, is_valid, flow, datum[0].sweep_uuid, output_root)
