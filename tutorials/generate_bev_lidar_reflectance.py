# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Argoverse 2.0 LiDAR Dataset tutorial.

Generates videos visualizing the LiDAR reflectance/intensity over the log.
"""
from __future__ import annotations

import copy
import logging
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final, List, Optional, Tuple, Union

import click
import numpy as np
from rich.progress import track

import av2.rendering.video as video_utils
import av2.utils.dense_grid_interpolation as dense_grid_interpolation
import av2.utils.io as io_utils
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from av2.geometry.se3 import SE3
from av2.geometry.sim2 import Sim2
from av2.structs.aggregated_sweep import AggregatedSweep
from av2.structs.bev_params import BEVParams
from av2.structs.sweep import Sweep
from av2.utils.typing import NDArrayByte, NDArrayFloat

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)


LIDAR_HZ: Final[int] = 10


def render_bev_img(
    bev_params: BEVParams, agg_sweep_ego: Union[AggregatedSweep, Sweep], interpolate: bool = False
) -> NDArrayByte:
    """Render a bird's eye view grayscale image, with intensity coming from equalized LiDAR reflectance values.

    Args:
        bev_params: parameters for rendering,
        agg_sweep_ego: AggregatedSweep representing accumulated points in the egovehicle frame.
        interpolate: whether to densely interpolate the reflectance values.

    Returns:
        Array of shape (H,W) representing grayscale LiDAR intensity image.
    """
    grid_xmin, grid_xmax = bev_params.xlims
    grid_ymin, grid_ymax = bev_params.ylims

    pruned_agg_sweep_ego = agg_sweep_ego.prune_to_2d_bbox(grid_xmin, grid_ymin, grid_xmax, grid_ymax)
    reflectance = pruned_agg_sweep_ego.intensity

    num_lidar_pts = len(pruned_agg_sweep_ego)
    logger.info(f"Rendering {num_lidar_pts/1e6:.1f} million LiDAR points")

    lidar_xy = pruned_agg_sweep_ego.xyz[:, :2]
    img_xy = bev_params.img_Sim2_ego.transform_point_cloud(lidar_xy)
    img_xy = np.round(img_xy).astype(np.int64)

    x = img_xy[:, 0]
    y = img_xy[:, 1]

    img_h, img_w = bev_params.get_image_dims()
    bev_refl_img: NDArrayByte = np.zeros((img_h, img_w), dtype=np.uint8)

    if interpolate:
        bev_refl_img: NDArrayByte = dense_grid_interpolation.interp_dense_grid_from_sparse(
            grid_img=bev_refl_img,
            points=img_xy,
            values=reflectance,
            grid_h=img_h,
            grid_w=img_w,
            interp_method="nearest",  # "linear"
        )
    else:
        bev_refl_img[y, x] = reflectance

    # flip image along y axis, for positive y-axis mirroring
    return np.flipud(bev_refl_img)


def accumulate_all_frames(log_id: str, data_root: Path) -> AggregatedSweep:
    """Aggregated sweeps representing (x,y,z) in city coordinate system, and reflectance.

    Args:
        log_id: unique ID for AV2 scenario/log.
        data_root: path to where the AV2 logs live.

    Returns:
        AggregatedSweep representing aggregated LiDAR returns, placed in the city frame.
    """
    loader = AV2SensorDataLoader(data_dir=data_root, labels_dir=data_root)

    agg_sweep_city = AggregatedSweep()
    lidar_timestamps_ns = loader.get_ordered_log_lidar_timestamps(log_id=log_id)

    if len(lidar_timestamps_ns) == 0:
        raise RuntimeError("No LiDAR sweeps were found.")

    for lidar_timestamp_ns in track(lidar_timestamps_ns, description="Aggregating sweeps..."):
        city_SE3_egovehicle = loader.get_city_SE3_ego(log_id=log_id, timestamp_ns=lidar_timestamp_ns)
        feather_fpath = loader.get_lidar_fpath(log_id, lidar_timestamp_ns)
        sweep_ego = Sweep.from_feather(feather_fpath)
        sweep_city = city_SE3_egovehicle.transform_sweep_from(sweep_ego)
        agg_sweep_city.add_sweep(sweep_city)

    logger.info("Aggregated %d points", len(agg_sweep_city))
    return agg_sweep_city


def render_log_reflectance_video(
    data_root: Path,
    output_dir: Path,
    log_id: str,
    render_north_as_up: bool,
    use_single_sweep: bool,
    interpolate_dense: bool,
    res_meters_per_px: float,
    dump_single_frames: bool,
    bev_params: BEVParams,
) -> None:
    """Render LiDAR reflectance in the BEV for each frame (sweep) of a AV2 log.

    Args:
        data_root: Path to local directory where the Argoverse 2 Sensor Dataset logs are stored.
        output_dir: Path to local directory where renderings will be saved.
        log_id: unique ID for AV2 scenario/log.
        render_north_as_up: whether to render north as up in the rendering.
        use_single_sweep:
        interpolate_dense: whether to densely interpolate the reflectance values.
        res_meters_per_px: resolution of rendering, expressed in meters/pixel.
        dump_single_frames: Whether to save to disk individual RGB frames of the rendering,
            in addition to generating the mp4 file.
        bev_params: bird's eye view rendering parameters.
    """
    dataset_name = str(bev_params)
    dataset_save_dir = output_dir / dataset_name / log_id
    dataset_save_dir.mkdir(parents=True, exist_ok=True)

    loader = AV2SensorDataLoader(data_dir=data_root, labels_dir=data_root)
    if not use_single_sweep:
        agg_sweep_city = accumulate_all_frames(log_id, data_root=data_root)
        agg_sweep_city.equalize_intensity_distribution()

    video_list: List[NDArrayByte] = []

    lidar_timestamps_ns = loader.get_ordered_log_lidar_timestamps(log_id=log_id)
    # render at each of the ego-poses
    for fr_idx, lidar_timestamp_ns in enumerate(lidar_timestamps_ns):

        city_SE3_egovehicle = loader.get_city_SE3_ego(log_id=log_id, timestamp_ns=lidar_timestamp_ns)
        egovehicle_SE3_city = city_SE3_egovehicle.inverse()

        # whether or not to accumulate multiple sweeps
        if use_single_sweep:
            feather_fpath = loader.get_lidar_fpath(log_id, lidar_timestamp_ns)
            sweep_ego = Sweep.from_feather(feather_fpath)
            sweep_ego.equalize_intensity_distribution()
        else:
            # accumulate sweeps
            agg_sweep_ego = egovehicle_SE3_city.transform_sweep_from(copy.deepcopy(agg_sweep_city))

        # allow option to keep North as +y in image frame
        if render_north_as_up:
            city_SO3_ego = SE3(rotation=city_SE3_egovehicle.rotation, translation=np.zeros(3))
            # centered at egovehicle, but now w/ the city's orientation.
            agg_sweep_ego = city_SO3_ego.transform_sweep_from(agg_sweep_ego)

        bev_refl_img = render_bev_img(bev_params, agg_sweep_ego, interpolate=interpolate_dense)
        if dump_single_frames:
            refl_img_fpath = Path(dataset_save_dir) / f"refl__fr_{fr_idx}.png"
            io_utils.write_img(refl_img_fpath, bev_refl_img)

        # copy channels of grayscale, to simulate RGB
        h, w = bev_refl_img.shape
        frame_rgb: NDArrayByte = np.tile(bev_refl_img.reshape(h, w, 1), (1, 1, 3))
        video_list.append(frame_rgb)

    video: NDArrayByte = np.stack(video_list).astype(np.uint8)
    video_output_dir = os.path.join(output_dir, "videos")
    video_utils.write_video(
        video=video,
        dst=Path(video_output_dir) / f"{log_id}_lidar_intensity.mp4",
        fps=LIDAR_HZ,
        preset="medium",
    )


@click.command(help="Generate LiDAR + map visualizations from the Argoverse 2 Sensor Dataset.")
@click.option(
    "-d",
    "--data_root",
    required=True,
    help="Path to local directory where the Argoverse 2 Sensor Dataset logs are stored.",
    type=click.Path(exists=True),
)
@click.option(
    "-o",
    "--output_dir",
    required=True,
    help="Path to local directory where renderings will be saved.",
    type=str,
)
@click.option(
    "-l",
    "--log_id",
    default="00a6ffc1-6ce9-3bc3-a060-6006e9893a1a",
    help="unique log identifier.",
    type=str,
)
@click.option(
    "--render_north_as_up",
    default=True,
    help="Boolean flag whether to render north as up in the rendering"
    "(as opposed to using the +y axis of the egovehicle frame -- left of the AV -- as up).",
    type=bool,
)
@click.option(
    "--use_single_sweep",
    default=False,
    help="defaults to false. whether to use a single sweep vs. aggregating points from all sweeps"
    " using ego-motion compensation",
    type=bool,
)
@click.option(
    "--interpolate_dense",
    default=False,
    help="whether to densely interpolate the reflectance values."
    "(defaults to false, as very computationally expensive and slow).",
    type=bool,
)
@click.option(
    "--res_meters_per_px", type=float, default=0.075, help="resolution of rendering, expressed in meters/pixel."
)
@click.option(
    "--range_m",
    type=float,
    default=30,
    help="Maximum spatial range to include in rendering (distance to 3d points by infinity norm).",
)
@click.option(
    "--dump_single_frames",
    default=False,
    help="Whether to save to disk individual RGB frames of the rendering, in addition to generating the mp4 file"
    "(defaults to False). Note: can quickly generate 100s of MBs, for 200 KB frames.",
    type=bool,
)
def run_render_log_reflectance_video(
    data_root: str,
    output_dir: str,
    log_id: str,
    render_north_as_up: bool,
    use_single_sweep: bool,
    interpolate_dense: bool,
    res_meters_per_px: float,
    range_m: float,
    dump_single_frames: bool,
) -> None:
    """Click entry point for LiDAR reflectance video generation."""
    logging.info(
        "data_root: %s, output_dir: %s, log_id: %s, render_north_as_up: %s, use_single_sweep: %s"
        "interpolate_dense: %s, res_meters_per_px: %f, range_m: %f, dump_single_frames: %s",
        data_root,
        output_dir,
        log_id,
        render_north_as_up,
        use_single_sweep,
        interpolate_dense,
        res_meters_per_px,
        range_m,
        dump_single_frames,
    )
    bev_params = BEVParams(range_m=range_m, res_meters_per_px=res_meters_per_px, accumulate_sweeps=not use_single_sweep)

    if dump_single_frames and interpolate_dense:
        raise ValueError("Invalid args: single LiDAR frames contain insufficient information for interpolation.")

    logger.info("Generate imagery w/ params:")
    logger.info(bev_params)

    render_log_reflectance_video(
        data_root=Path(data_root),
        output_dir=Path(output_dir),
        log_id=log_id,
        render_north_as_up=render_north_as_up,
        use_single_sweep=use_single_sweep,
        interpolate_dense=interpolate_dense,
        res_meters_per_px=res_meters_per_px,
        dump_single_frames=dump_single_frames,
        bev_params=bev_params,
    )


if __name__ == "__main__":
    run_render_log_reflectance_video()
    