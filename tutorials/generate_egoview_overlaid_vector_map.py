# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Generate MP4 videos with map entities rendered on top of sensor imagery, for all cameras, for a single log.

We use a inferred depth map from LiDAR to render only visible map entities (lanes and pedestrian crossings).
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Final, List, Tuple

import click
import numpy as np

import av2.geometry.interpolate as interp_utils
import av2.rendering.video as video_utils
import av2.utils.io as io_utils
import av2.utils.raster as raster_utils
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from av2.datasets.sensor.constants import RingCameras
from av2.map.map_api import ArgoverseStaticMap
from av2.rendering.color import BLUE_BGR
from av2.rendering.map import EgoViewMapRenderer
from av2.utils.typing import NDArrayByte

RING_CAMERA_FPS: Final[int] = 20

logger = logging.getLogger(__name__)


def generate_egoview_overlaid_map(
    data_root: Path,
    output_dir: Path,
    log_id: str,
    max_range_m: float,
    use_depth_map_for_occlusion: bool,
    dump_single_frames: bool,
    cam_names: List[RingCameras],
) -> None:
    """Render the map from a particular camera's viewpoint for each camera frame.

    Args:
        data_root: path to where the AV2 logs live.
        output_dir: path to directory where renderings will be saved.
        log_id: unique ID for AV2 scenario/log.
        max_range_m: maximum range of map entities from egovehicle to consider for rendering (by l-infinity norm).
        use_depth_map_for_occlusion: whether to use an inferred depth map for rendering occluded elements.
        dump_single_frames: Whether to save to disk individual RGB frames of the rendering, in addition to generating
            the mp4 file.
        cam_names: list of camera names. For each camera, its viewport will be used to render the map.
    """
    loader = AV2SensorDataLoader(data_dir=data_root, labels_dir=data_root)

    log_map_dirpath = data_root / log_id / "map"
    avm = ArgoverseStaticMap.from_map_dir(log_map_dirpath, build_raster=True)

    for _, cam_enum in enumerate(cam_names):
        cam_name = cam_enum.value
        pinhole_cam = loader.get_log_pinhole_camera(log_id, cam_name)

        cam_im_fpaths = loader.get_ordered_log_cam_fpaths(log_id, cam_name)
        num_cam_imgs = len(cam_im_fpaths)

        video_list = []
        for i, img_fpath in enumerate(cam_im_fpaths):
            if i % 50 == 0:
                logging.info(
                    f"\tOn file {i}/{num_cam_imgs} of camera {cam_name} of {log_id}"
                )

            cam_timestamp_ns = int(img_fpath.stem)
            city_SE3_ego = loader.get_city_SE3_ego(log_id, cam_timestamp_ns)
            if city_SE3_ego is None:
                logger.info("missing LiDAR pose")
                continue

            # load feather file path, e.g. '315978406032859416.feather"
            lidar_fpath = loader.get_closest_lidar_fpath(log_id, cam_timestamp_ns)
            if lidar_fpath is None:
                # without depth map, can't do this accurately
                continue

            lidar_points = io_utils.read_lidar_sweep(lidar_fpath, attrib_spec="xyz")
            lidar_timestamp_ns = int(lidar_fpath.stem)

            if use_depth_map_for_occlusion:
                depth_map = loader.get_depth_map_from_lidar(
                    lidar_points=lidar_points,
                    cam_name=cam_name,
                    log_id=log_id,
                    cam_timestamp_ns=cam_timestamp_ns,
                    lidar_timestamp_ns=lidar_timestamp_ns,
                )
            else:
                depth_map = None

            egoview_renderer = EgoViewMapRenderer(
                depth_map=depth_map,
                city_SE3_ego=city_SE3_ego,
                pinhole_cam=pinhole_cam,
                avm=avm,
            )
            frame_rgb = render_egoview(
                output_dir=output_dir,
                img_fpath=img_fpath,
                egoview_renderer=egoview_renderer,
                cam_timestamp_ns=cam_timestamp_ns,
                log_id=log_id,
                max_range_m=max_range_m,
                dump_single_frames=dump_single_frames,
            )
            video_list.append(frame_rgb)

        video: NDArrayByte = np.stack(video_list).astype(np.uint8)
        video_output_dir = output_dir / "videos"
        video_utils.write_video(
            video=video,
            dst=video_output_dir / f"{log_id}_{cam_name}.mp4",
            fps=RING_CAMERA_FPS,
            preset="medium",
        )


def render_egoview(
    output_dir: Path,
    img_fpath: Path,
    egoview_renderer: EgoViewMapRenderer,
    cam_timestamp_ns: int,
    log_id: str,
    max_range_m: float,
    dump_single_frames: bool,
) -> NDArrayByte:
    """Synthetically manipulate a vector map, render the map in the ego-view, and save rendering to disk.

    Args:
        output_dir: path to directory where renderings will be saved.
        img_fpath: path to RGB image, from one of the ring or stereo cameras.
        egoview_renderer: rendering engine for map elements in the ego-view.
        cam_timestamp_ns: nanosecond camera timestamp when image was captured.
        log_id: unique ID for AV2 scenario/log.
        max_range_m: maximum range of map entities from egovehicle to consider for rendering (by l-infinity norm).
        dump_single_frames: Whether to save to disk individual RGB frames of the rendering, in addition to generating
            the mp4 file.

    Returns:
        array of shape (H,W,3) and type uint8 representing a RGB image.
    """
    save_dir = output_dir / log_id
    if dump_single_frames:
        # we only create log-specific directories, if dumping individual frames.
        save_dir.mkdir(exist_ok=True, parents=True)

    img_fname = (
        f"{egoview_renderer.pinhole_cam.cam_name}_{cam_timestamp_ns}_vectormap.jpg"
    )
    save_fpath = save_dir / img_fname

    if save_fpath.exists():
        logger.info("Rendered image already exists, skipping")
        img: NDArrayByte = io_utils.read_img(save_fpath)
        return img

    start = time.time()

    img_rgb: NDArrayByte = io_utils.read_img(img_fpath)

    # to prevent washing out, can pass in black image, and get just mask back, or can overlay directly.
    img_h, img_w, _ = img_rgb.shape
    img_empty: NDArrayByte = np.full(
        (img_h, img_w, 3), fill_value=128, dtype=np.uint8
    )  # pure white polylines will disappear @ 255

    img_empty = render_egoview_with_occlusion_checks(
        img_canvas=img_empty,
        egoview_renderer=egoview_renderer,
        max_range_m=max_range_m,
    )
    end = time.time()
    duration = end - start
    logger.info(f"Rendering single image took {duration:.2f} sec.")

    frame_rgb = raster_utils.blend_images(img_rgb, img_empty, alpha=0.45)

    if dump_single_frames:
        io_utils.write_img(save_fpath, frame_rgb, channel_order="RGB")
    return frame_rgb


def render_egoview_with_occlusion_checks(
    img_canvas: NDArrayByte,
    egoview_renderer: EgoViewMapRenderer,
    max_range_m: float,
    line_width_px: int = 10,
) -> NDArrayByte:
    """Render pedestrian crossings and lane segments in the ego-view.

    Pedestrian crossings (crosswalks) will be rendered in blue, and lane markings will be colored according to their
    marking color, or otherwise red, if markings are implicit.

    Args:
        img_canvas: array of shape (H,W,3) representing BGR canvas to rasterize map elements onto.
        egoview_renderer: rendering engine for map elements in the ego-view.
        max_range_m: maximum range of map entities from egovehicle to consider for rendering (by l-infinity norm).
        line_width_px: thickness (in pixels) to use for rendering each polyline.

    Returns:
        array of shape (H,W,3) and type uint8 representing a RGB image.
    """
    for ls in egoview_renderer.avm.get_scenario_lane_segments():
        img_canvas = egoview_renderer.render_lane_boundary_egoview(
            img_canvas, ls, "right", line_width_px
        )
        img_canvas = egoview_renderer.render_lane_boundary_egoview(
            img_canvas, ls, "left", line_width_px
        )

    for pc in egoview_renderer.avm.get_scenario_ped_crossings():
        EPS = 1e-5
        crosswalk_color = BLUE_BGR
        # render ped crossings (pc's)
        xwalk_polygon = pc.polygon
        # prevent duplicate first and last coords
        xwalk_polygon[:-1] += EPS
        N_INTERP_PTS = 100
        # For pixel-perfect rendering, querying crosswalk boundary ground height at waypoints throughout
        # the street is much more accurate than 3d linear interpolation using only the 4 annotated corners.
        polygon_city_frame = interp_utils.interp_arc(
            t=N_INTERP_PTS, points=xwalk_polygon[:, :2]
        )
        polygon_city_frame = egoview_renderer.avm.append_height_to_2d_city_pt_cloud(
            points_xy=polygon_city_frame
        )
        egoview_renderer.render_polyline_egoview(
            polygon_city_frame,
            img_canvas,
            crosswalk_color,
            thickness_px=line_width_px,
        )

    # convert BGR to RGB
    img_rgb: NDArrayByte = img_canvas[:, :, ::-1]
    return img_rgb


def parse_camera_enum_types(cam_names: Tuple[str, ...]) -> List[RingCameras]:
    """Convert a list of CLI string types, to enums of type RingCameras, and validate each input.

    Args:
        cam_names: Tuple of camera names to use for rendering the map.

    Returns:
        List of camera enums to use for rendering the map.

    Raises:
        ValueError: If an invalid camera name is provided.
    """
    valid_ring_cams = set([x.value for x in list(RingCameras)])

    cam_enums: List[RingCameras] = []
    for cam_name in list(cam_names):
        if cam_name in valid_ring_cams:
            cam_enums.append(RingCameras(cam_name))
        else:
            raise ValueError("Must provide _valid_ camera names!")
    return cam_enums


@click.command(
    help="Generate map visualizations on ego-view imagery from the Argoverse 2 Sensor or TbV Datasets."
)
@click.option(
    "--data-root",
    required=True,
    help="Path to local directory where the Argoverse 2 Sensor Dataset or TbV logs are stored.",
    type=click.Path(exists=True),
)
@click.option(
    "-o",
    "--output-dir",
    required=True,
    help="Path to local directory where renderings will be saved.",
    type=str,
)
@click.option(
    "-l",
    "--log-id",
    default="00a6ffc1-6ce9-3bc3-a060-6006e9893a1a",
    help="unique log identifier.",
    type=str,
)
@click.option(
    "-r",
    "--max-range-m",
    type=float,
    default=100,
    help="Maximum range of map entities from egovehicle to consider for rendering (by l-infinity norm).",
)
@click.option(
    "-d",
    "--use-depth-map-for_occlusion",
    default=True,
    help="Whether to use an inferred depth map for rendering occluded elements (defaults to True).",
    type=bool,
)
@click.option(
    "-s",
    "--dump-single-frames",
    default=False,
    help="Whether to save to disk individual RGB frames of the rendering, in addition to generating the mp4 file"
    "(defaults to False). Note: can quickly generate 100s of MBs, for 200 KB frames.",
    type=bool,
)
@click.option(
    "-c",
    "--cam-names",
    default=tuple(x.value for x in list(RingCameras)),
    help="List of camera viewpoints to render the map from.",
    multiple=True,
    type=str,
)
def run_generate_egoview_overlaid_map(
    data_root: "os.PathLike[str]",
    output_dir: "os.PathLike[str]",
    log_id: str,
    max_range_m: float,
    use_depth_map_for_occlusion: bool,
    dump_single_frames: bool,
    cam_names: Tuple[str, ...],
) -> None:
    """Click entry point for visualizing map entities rendered on top of sensor imagery."""
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    data_root = Path(data_root)
    output_dir = Path(output_dir)

    logger.info(
        "data_root: %s, output_dir: %s, log_id: %s, max_range_m: %f, "
        "use_depth_map_for_occlusion: %s, dump_single_frames %s",
        data_root,
        output_dir,
        log_id,
        max_range_m,
        use_depth_map_for_occlusion,
        dump_single_frames,
    )
    generate_egoview_overlaid_map(
        data_root=data_root,
        output_dir=output_dir,
        log_id=log_id,
        max_range_m=max_range_m,
        use_depth_map_for_occlusion=use_depth_map_for_occlusion,
        dump_single_frames=dump_single_frames,
        cam_names=parse_camera_enum_types(cam_names),
    )


if __name__ == "__main__":
    run_generate_egoview_overlaid_map()
