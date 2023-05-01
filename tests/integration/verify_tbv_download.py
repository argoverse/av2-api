# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Verifies that all expected logs are present and have been downloaded + extracted correctly from S3."""

import logging
import sys
from pathlib import Path
from typing import Final, Tuple

import click
from rich.progress import track

import av2.utils.io as io_utils
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from av2.datasets.sensor.constants import RingCameras
from av2.datasets.tbv.splits import TEST, TRAIN, VAL
from av2.geometry.se3 import SE3
from av2.map.map_api import ArgoverseStaticMap

logger = logging.getLogger(__name__)

MIN_NUM_SWEEPS_PER_LOG: Final[int] = 40
MIN_NUM_IMAGES_PER_CAMERA: Final[int] = 80

EXPECTED_NUM_TBV_IMAGES: Final[int] = 7837614
EXPECTED_NUM_TBV_SWEEPS: Final[int] = 559440

AV2_CITY_NAMES: Final[Tuple[str, ...]] = ("ATX", "DTW", "MIA", "PAO", "PIT", "WDC")

# every lane segment should have 10 keys only.
EXPECTED_LANE_SEGMENT_ATTRIB_KEYS: Final[Tuple[str, ...]] = (
    "id",
    "is_intersection",
    "lane_type",
    "left_lane_boundary",
    "left_lane_mark_type",
    "right_lane_boundary",
    "right_lane_mark_type",
    "successors",
    "predecessors",
    "right_neighbor_id",
    "left_neighbor_id",
)


def verify_log_contents(data_root: Path, log_id: str, check_image_sizes: bool) -> None:
    """Verify that expected files exist and are loadable, for a single log.

    Args:
        data_root: Path to local directory where the TbV Dataset logs are stored.
        log_id: unique ID of TbV vehicle log.
        check_image_sizes: Whether to verify the size of every image. This check is very
            expensive over millions of images.
    """
    # every log should have a subfolder for each of the 7 ring cameras.
    assert (data_root / log_id / "sensors" / "cameras").exists()

    # every log should have 4*20 images for each camera.
    for camera_enum in list(RingCameras):
        cam_images_dir = data_root / log_id / "sensors" / "cameras" / camera_enum.value
        img_fpaths = list(cam_images_dir.glob("*.jpg"))
        assert (
            len(img_fpaths) >= MIN_NUM_IMAGES_PER_CAMERA
        ), "There should be at last 80 images for each camera, per log."

        # this check is expensive, and can be skipped.
        if not check_image_sizes:
            continue
        # every image should be (H,W) = 2048x1550 (front-center) or 775x1024 for all other cameras.
        for img_fpath in track(
            img_fpaths, description=f"Verifying image sizes for {camera_enum}"
        ):
            img = io_utils.read_img(img_path=img_fpath, channel_order="RGB")
            if camera_enum == RingCameras.RING_FRONT_CENTER:
                assert img.shape == (2048, 1550, 3)
            else:
                assert img.shape == (775, 1024, 3)

    # every log should have 40+ LiDAR sweeps.
    lidar_dirpath = data_root / log_id / "sensors" / "lidar"
    assert lidar_dirpath.exists()
    assert len(list(lidar_dirpath.glob("*.feather"))) >= MIN_NUM_SWEEPS_PER_LOG

    # every log should have a file w/ ego-vehicle poses.
    poses_fpath = data_root / log_id / "city_SE3_egovehicle.feather"
    assert poses_fpath.exists()
    # poses file should be loadable.
    poses_df = io_utils.read_feather(poses_fpath)
    assert list(poses_df.keys()) == [
        "timestamp_ns",
        "qw",
        "qx",
        "qy",
        "qz",
        "tx_m",
        "ty_m",
        "tz_m",
    ]

    # every log should have an extrinsics calibration file.
    extrinsics_fpath = (
        data_root / log_id / "calibration" / "egovehicle_SE3_sensor.feather"
    )
    assert extrinsics_fpath.exists()
    # extrinsics should be loadable.
    extrinsics_df = io_utils.read_feather(extrinsics_fpath)
    assert list(extrinsics_df.keys()) == [
        "sensor_name",
        "qw",
        "qx",
        "qy",
        "qz",
        "tx_m",
        "ty_m",
        "tz_m",
    ]

    # extrinsics should be provided for each camera.
    for camera_enum in list(RingCameras):
        assert camera_enum.value in extrinsics_df["sensor_name"].tolist()

    # extrinsics should be provided for each LiDAR.
    assert "up_lidar" in extrinsics_df["sensor_name"].tolist()
    assert "down_lidar" in extrinsics_df["sensor_name"].tolist()

    # every log should have an intrinsics calibration file.
    intrinsics_fpath = data_root / log_id / "calibration" / "intrinsics.feather"
    assert intrinsics_fpath.exists()
    # intrinsics file should be loadable.
    intrinsics_df = io_utils.read_feather(intrinsics_fpath)
    assert list(intrinsics_df.keys()) == [
        "sensor_name",
        "fx_px",
        "fy_px",
        "cx_px",
        "cy_px",
        "k1",
        "k2",
        "k3",
        "height_px",
        "width_px",
    ]

    # intrinsics should be provided for each camera
    for camera_enum in list(RingCameras):
        assert camera_enum.value in intrinsics_df["sensor_name"].tolist()

    verify_log_map(data_root=data_root, log_id=log_id)


def verify_log_map(data_root: Path, log_id: str) -> None:
    """Verify validity of each log's map files.

    Args:
        data_root: Path to local directory where the TbV Dataset logs are stored.
        log_id: unique ID of TbV vehicle log.
    """
    # every log should have a `map` subfolder.
    log_map_dirpath = data_root / log_id / "map"
    assert log_map_dirpath.exists()

    # every log should have one and only one raster height map. (Note: season is stripped from uuid here).
    ground_height_raster_fpaths = list(
        log_map_dirpath.glob("*_ground_height_surface____*.npy")
    )
    assert len(ground_height_raster_fpaths) == 1

    # every log should have a Sim(2) mapping from raster grid coordinates to city coordinates.
    Sim2_fpaths = list(log_map_dirpath.glob("*___img_Sim2_city.json"))
    assert len(Sim2_fpaths) == 1
    Sim2_fpath = Sim2_fpaths[0]
    assert Path(Sim2_fpath).exists()

    # every log should have a vector map.
    vector_map_fpaths = list(log_map_dirpath.glob("log_map_archive_*____*_city_*.json"))
    # there should only be 1 vector map file per log.
    assert len(vector_map_fpaths) == 1
    vector_map_fpath = vector_map_fpaths[0]

    # every vector map file should have only 3 keys -- "pedestrian_crossings", "lane_segments", "drivable_areas"
    vector_map_json_data = io_utils.read_json_file(vector_map_fpath)
    assert list(vector_map_json_data.keys()) == [
        "pedestrian_crossings",
        "lane_segments",
        "drivable_areas",
    ]

    for _, lane_segment_dict in vector_map_json_data["lane_segments"].items():
        assert tuple(lane_segment_dict.keys()) == EXPECTED_LANE_SEGMENT_ATTRIB_KEYS

    # every map should be loadable from pure JSON.
    avm = ArgoverseStaticMap.from_json(static_map_path=vector_map_fpath)

    # every map should be loadable w/ build_raster=False
    avm = ArgoverseStaticMap.from_map_dir(
        log_map_dirpath=log_map_dirpath, build_raster=False
    )

    # every map should be loadable w/ build_raster=True.
    avm = ArgoverseStaticMap.from_map_dir(
        log_map_dirpath=log_map_dirpath, build_raster=True
    )

    # load every lane segment
    lane_segments = avm.get_scenario_lane_segments()
    for ls in lane_segments:
        right_lane_boundary = ls.right_lane_boundary.xyz
        left_lane_boundary = ls.left_lane_boundary.xyz

        # each lane boundary should have shape (N,3)
        assert right_lane_boundary.ndim == 2 and right_lane_boundary.shape[1] == 3
        assert left_lane_boundary.ndim == 2 and left_lane_boundary.shape[1] == 3

    # load every pedestrian crossing
    avm.get_scenario_ped_crossings()

    # load every drivable area
    avm.get_scenario_vector_drivable_areas()


def verify_logs_using_dataloader(data_root: Path, log_ids: Tuple[str, ...]) -> None:
    """Use a dataloader object to query each log's data, and verify it.

    Args:
        data_root: Path to local directory where the TbV Dataset logs are stored.
        log_ids: unique IDs of TbV vehicle logs.
    """
    loader = AV2SensorDataLoader(data_dir=data_root, labels_dir=data_root)
    for log_id in track(
        log_ids, description="Verify logs using an AV2 dataloader object"
    ):
        logger.info("Verifying log %s", log_id)
        # city abbreviation should be parsable from every vector map file name, and should fall into 1 of 6 cities
        city_name = loader.get_city_name(log_id=log_id)
        assert city_name in AV2_CITY_NAMES

        # pose should be present for every lidar sweep.
        lidar_timestamps_ns = loader.get_ordered_log_lidar_timestamps(log_id=log_id)
        for lidar_timestamp_ns in lidar_timestamps_ns:
            city_SE3_egovehicle = loader.get_city_SE3_ego(
                log_id=log_id, timestamp_ns=lidar_timestamp_ns
            )
            assert isinstance(city_SE3_egovehicle, SE3)


@click.command(help="Verify contents of downloaded + extracted TbV Dataset logs.")
@click.option(
    "-d",
    "--data-root",
    required=True,
    help="Path to local directory where the TbV Dataset logs are stored.",
    type=click.Path(exists=True),
)
@click.option(
    "--check-image-sizes",
    default=False,
    help="Whether to verify the size of every image. This check is very expensive over millions of images.",
    type=bool,
)
def run_verify_all_tbv_logs(data_root: str, check_image_sizes: bool) -> None:
    """Click entry point for TbV file verification."""
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    log_ids = TRAIN + VAL + TEST
    num_logs = len(log_ids)
    for i in range(num_logs):
        log_id = log_ids[i]
        logger.info("Verifying log %d: %s", i, log_id)
        verify_log_contents(
            data_root=Path(data_root),
            log_id=log_id,
            check_image_sizes=check_image_sizes,
        )

    verify_logs_using_dataloader(data_root=Path(data_root), log_ids=log_ids)

    # verify the total number of images found on disk.
    img_fpaths = list(Path(data_root).glob("*/sensors/cameras/*/*.jpg"))
    assert len(img_fpaths) == EXPECTED_NUM_TBV_IMAGES

    # verify the total number of LiDAR sweeps found on disk.
    lidar_fpaths = list(Path(data_root).glob("*/sensors/lidar/*.feather"))
    assert len(lidar_fpaths) == EXPECTED_NUM_TBV_SWEEPS


if __name__ == "__main__":
    run_verify_all_tbv_logs()
