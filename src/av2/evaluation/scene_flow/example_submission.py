"""An example showing how to output flow predictions in the format required for submission."""
from pathlib import Path

import click
import numpy as np
from kornia.geometry.linalg import transform_points
from rich.progress import track

from av2.evaluation.scene_flow.utils import get_eval_point_mask, get_eval_subset, write_output_file
from av2.torch.data_loaders.scene_flow import SceneFlowDataloader


@click.command()
@click.argument("output_dir", type=str)
@click.argument("data_dir", type=str)
@click.option(
    "--name",
    type=str,
    help="the data should be located in <data_dir>/<name>/sensor/<split>",
    default="av2",
)
def example_submission(output_dir: str, data_dir: str, name: str) -> None:
    """Output the point masks for submission to the leaderboard.

    Args:
        output_dir: Path to output directory.
        data_dir: Path to input data.
        name: Name of the dataset (e.g. av2).
    """
    data_loader = SceneFlowDataloader(Path(data_dir), name, "test")

    output_root = Path(output_dir)
    output_root.mkdir(exist_ok=True)

    eval_inds = get_eval_subset(data_loader)
    for i in track(eval_inds, description="Generating outputs..."):
        sweep_0, sweep_1, ego_1_SE3_ego_0, flow = data_loader[i]
        mask = get_eval_point_mask(sweep_0.sweep_uuid)

        pc1 = sweep_0.lidar.as_tensor()[mask, :3]
        pc1_rigid = transform_points(ego_1_SE3_ego_0.matrix(), pc1[None])[0]
        rigid_flow = (pc1_rigid - pc1).detach().numpy()
        is_dynamic = np.zeros(len(rigid_flow), dtype=bool)

        write_output_file(rigid_flow, is_dynamic, sweep_0.sweep_uuid, output_root)


if __name__ == "__main__":
    example_submission()
