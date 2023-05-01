# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Generate an mp4 video for each ring camera, for each AV2 log."""

import logging
import sys
from pathlib import Path
from typing import Final, List

import click
import numpy as np
from joblib import Parallel, delayed

import av2.rendering.video as video_utils
import av2.utils.io as io_utils
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from av2.datasets.sensor.constants import RingCameras
from av2.utils.typing import NDArrayByte

RING_CAMERA_FPS: Final[int] = 20

# for every 50 images, print a status message
PRINT_EVERY: Final[int] = 50

logger = logging.getLogger(__name__)


def generate_per_camera_videos(
    data_root: Path, output_dir: Path, num_workers: int
) -> None:
    """Launch jobs to render ring camera .mp4 videos for all sensor logs available on disk.

    Args:
        data_root: Path to local directory where the Argoverse 2 Sensor Dataset logs are stored.
        output_dir: Path to local directory where videos will be saved.
        num_workers: Number of worker processes to use for rendering.
    """
    loader = AV2SensorDataLoader(data_dir=data_root, labels_dir=data_root)
    log_ids = loader.get_log_ids()

    if num_workers > 1:
        Parallel(n_jobs=num_workers)(
            delayed(render_log_ring_camera_videos)(output_dir, loader, log_id)
            for log_id in log_ids
        )
    else:
        for log_id in log_ids:
            render_log_ring_camera_videos(output_dir, loader, log_id)


def render_log_ring_camera_videos(
    output_dir: Path, loader: AV2SensorDataLoader, log_id: str
) -> None:
    """Render .mp4 videos for all ring cameras of a single log.

    Args:
        output_dir: Path to local directory where videos will be saved.
        loader: data loader for sensor data.
        log_id: unique ID for AV2 scenario/log.
    """
    # The logger must be initialized here, otherwise no output will be emitted to stdout.
    # joblib overrides the logger config, see: https://github.com/joblib/joblib/issues/692
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logger.info("Generating ring camera videos for %s", log_id)
    for camera_name in list(RingCameras):
        video_save_fpath = Path(output_dir) / f"{log_id}_{camera_name}.mp4"
        if video_save_fpath.exists():
            logger.info(
                "Video already exists for %s, %s, so skipping...", log_id, camera_name
            )
            continue

        cam_im_fpaths = loader.get_ordered_log_cam_fpaths(log_id, camera_name)
        num_cam_imgs = len(cam_im_fpaths)

        video_list: List[NDArrayByte] = []
        for i, im_fpath in enumerate(cam_im_fpaths):
            if i % PRINT_EVERY == 0:
                logger.info(
                    f"\tOn file {i}/{num_cam_imgs} of camera {camera_name} of {log_id}"
                )
            img_rgb = io_utils.read_img(im_fpath, channel_order="RGB")
            video_list.append(img_rgb)

        video: NDArrayByte = np.stack(video_list).astype(np.uint8)
        video_utils.write_video(
            video=video,
            dst=video_save_fpath,
            fps=RING_CAMERA_FPS,
            preset="medium",
        )


@click.command(
    help="Generate map visualizations on ego-view imagery from the Argoverse 2 Sensor or TbV Datasets."
)
@click.option(
    "-d",
    "--data-root",
    required=True,
    help="Path to local directory where the Argoverse 2 Sensor Dataset or TbV logs are stored.",
    type=click.Path(exists=True),
)
@click.option(
    "-o",
    "--output-dir",
    required=True,
    help="Path to local directory where videos will be saved.",
    type=str,
)
@click.option(
    "--num-workers",
    required=True,
    help="Number of worker processes to use for rendering.",
    type=int,
)
def run_generate_per_camera_videos(
    data_root: str, output_dir: str, num_workers: int
) -> None:
    """Click entry point for ring camera .mp4 video generation."""
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    generate_per_camera_videos(
        data_root=Path(data_root), output_dir=Path(output_dir), num_workers=num_workers
    )


if __name__ == "__main__":
    run_generate_per_camera_videos()
