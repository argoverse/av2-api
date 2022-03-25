# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Visualize a AV2 Sensor Dataset / LiDAR Dataset / TbV scene, using Open3d."""

from pathlib import Path
from typing import Final, List, Tuple

import click
import numpy as np
import open3d

import av2.geometry.mesh_grid as mesh_grid_utils
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from av2.geometry.se3 import SE3
from av2.map.map_api import ArgoverseStaticMap
from av2.structures.sweep import Sweep
from av2.utils.typing import NDArrayByte, NDArrayFloat

RED: Final[Tuple[int, int, int]] = (1, 0, 0)
GREEN: Final[Tuple[int, int, int]] = (0, 1, 0)
BLUE: Final[Tuple[int, int, int]] = (0, 0, 1)


def create_colored_point_cloud_open3d(point_cloud: NDArrayFloat, rgb: NDArrayByte) -> open3d.geometry.PointCloud:
    """Render a point cloud as individual colored points, using Open3d.

    Reference: https://github.com/borglab/gtsfm/blob/master/gtsfm/visualization/open3d_vis_utils.py

    Args:
        point_cloud: array of shape (N,3) representing 3d points.
        rgb: uint8 array of shape (N,3) representing colors in RGB order, in the range [0,255].

    Returns:
        pcd: Open3d geometry object representing a colored 3d point cloud.

    Raises:
        ValueError: if input point cloud is not a Numpy array, or does not have shape (N,3)
    """
    if not isinstance(point_cloud, np.ndarray):
        raise ValueError("Input point cloud must be a Numpy n-d array.")

    if point_cloud.shape[1] != 3:
        raise ValueError("Input point cloud must have shape (N,3).")

    colors = rgb.astype(np.float64) / 255  # type: ignore

    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(point_cloud)
    pcd.colors = open3d.utility.Vector3dVector(colors)

    return pcd


def get_vector_map_geometries(avm: ArgoverseStaticMap, city_SE3_egovehicle: SE3) -> List[open3d.geometry.LineSet]:
    """Get line sets representing lane segment boundaries.

    Args:
        avm: Argoverse 2.0 vector + raster map.
        city_SE3_egovehicle: egovehicle's pose in the city coordinate frame.

    Returns:
        List of line set objects, with each line set representing a lane boundary polyline.
    """
    lane_segments = avm.get_scenario_lane_segments()
    avm.get_scenario_vector_drivable_areas()
    avm.get_scenario_ped_crossings()

    BLUE = [0, 0, 1]
    lane_boundary_color = BLUE

    line_sets = []
    for lane_segment in lane_segments:
        for polyline_city in [lane_segment.right_lane_boundary.xyz, lane_segment.left_lane_boundary.xyz]:
            polyline_ego = city_SE3_egovehicle.inverse().transform_from(polyline_city)
            l = np.arange(len(polyline_ego))
            lines = list(zip(l, l[1:]))

            # color is in range [0,1]
            # color = tuple(colormap[i].tolist())
            colors = [lane_boundary_color for i in range(len(lines))]

            line_set = open3d.geometry.LineSet(
                points=open3d.utility.Vector3dVector(polyline_ego),
                lines=open3d.utility.Vector2iVector(lines),
            )
            line_set.colors = open3d.utility.Vector3dVector(colors)
            line_sets.append(line_set)

    return line_sets


def get_ground_surface_geometry(avm: ArgoverseStaticMap, city_SE3_egovehicle: SE3) -> open3d.geometry.PointCloud:
    """Get a point cloud representing the ground surface.

    Alternatively, a triangle mesh could be rendered.

    Args:
        avm: Argoverse 2.0 vector + raster map.
        city_SE3_egovehicle: egovehicle's pose in the city coordinate frame.

    Returns:
        Open3D point cloud object.
    """
    range_m = 100

    mesh_pts_egovehicle_frame = mesh_grid_utils.get_mesh_grid_as_point_cloud(
        min_x=-range_m, max_x=range_m, min_y=-range_m, max_y=range_m, downsample_factor=0.5
    )
    n_mesh_pts = mesh_pts_egovehicle_frame.shape[0]
    mesh_pts_egovehicle_frame = np.hstack([mesh_pts_egovehicle_frame, np.zeros((n_mesh_pts, 1))])
    # z unknown here
    mesh_pts_city_frame = city_SE3_egovehicle.transform_from(mesh_pts_egovehicle_frame)
    mesh_pts_city_frame = avm.append_height_to_2d_city_pt_cloud(points_xy=mesh_pts_city_frame[:, :2])
    mesh_pts_egovehicle_frame = city_SE3_egovehicle.inverse().transform_from(mesh_pts_city_frame)

    # remove points with NaN-valued heights
    valid_idxs = ~np.isnan(mesh_pts_egovehicle_frame[:, 2])
    valid_mesh_pts_egovehicle_frame = mesh_pts_egovehicle_frame[valid_idxs]

    rgb: NDArrayByte = np.zeros((n_mesh_pts, 3), dtype=np.uint8)
    # color as red.
    rgb[:, 0] = 255
    return create_colored_point_cloud_open3d(point_cloud=valid_mesh_pts_egovehicle_frame, rgb=rgb)


def draw_coordinate_frame(wTc: SE3, axis_length: float = 1.0) -> List[open3d.geometry.LineSet]:
    """Draw 3 orthogonal axes representing a coordinate frame.

    Note: x,y,z axes correspond to red, green, blue colors.
    Reference: https://github.com/borglab/gtsfm/blob/master/gtsfm/visualization/open3d_vis_utils.py

    Args:
        wTc: pose in a world coordinate frame.
        axis_length: length to use for line segments (representing coordinate frame axes).

    Returns:
        line_sets: list of Open3D LineSet objects, representing 3 axes (a coordinate frame).
    """
    axis_colors = (RED, GREEN, BLUE)

    # line segment on each axis will connect just 2 vertices.
    lines = [[0, 1]]

    line_sets = []
    for axis, color in zip([0, 1, 2], axis_colors):
        # one point at optical center, other point along specified axis.
        verts_camfr = np.zeros((2, 3))
        verts_camfr[0, axis] = axis_length

        verts_worldfr = wTc.transform_from(verts_camfr)

        line_set = open3d.geometry.LineSet(
            points=open3d.utility.Vector3dVector(verts_worldfr),
            lines=open3d.utility.Vector2iVector(lines),
        )
        line_set.colors = open3d.utility.Vector3dVector(np.array(color).reshape(1, 3))
        line_sets.append(line_set)

    return line_sets


def visualize_scene(data_root: Path, log_id: str, color_sweeps: bool = True) -> None:
    """Visualize LiDAR point cloud, lane segment boundaries, and ground surface height using Open3d.

    Args:
        data_root: Path to local directory where the Argoverse 2 Sensor Dataset logs are stored.
        log_id: unique log identifier.
        color_sweeps: whether to color LiDAR returns with corresponding synchronized RGB imagery.
    """
    loader = AV2SensorDataLoader(data_dir=data_root, labels_dir=data_root)

    log_map_dirpath = loader.get_log_map_dirpath(log_id=log_id)
    avm = ArgoverseStaticMap.from_map_dir(log_map_dirpath, build_raster=True)

    lidar_fpaths = loader.get_ordered_log_lidar_fpaths(log_id=log_id)
    for frame_idx, lidar_fpath in enumerate(lidar_fpaths):

        # full stream sometimes not available until a few frames in.
        if frame_idx < 5:
            continue

        geometries = []

        lidar_timestamp_ns = int(lidar_fpath.stem)
        city_SE3_egovehicle = loader.get_city_SE3_ego(log_id=log_id, timestamp_ns=lidar_timestamp_ns)

        # plot the ground surface as triangle mesh or point cloud.
        mesh_geometry = get_ground_surface_geometry(avm, city_SE3_egovehicle)
        geometries.append(mesh_geometry)

        vector_map_geometries = get_vector_map_geometries(avm, city_SE3_egovehicle)
        geometries.extend(vector_map_geometries)

        # put the point cloud into the frame
        # color LiDAR points by their RGB values
        sweep = Sweep.from_feather(lidar_fpath)

        if color_sweeps:
            rgb: NDArrayByte = loader.get_colored_sweep(log_id=log_id, lidar_timestamp_ns=lidar_timestamp_ns)
        else:
            # default to black.
            rgb: NDArrayByte = np.zeros((len(sweep), 3), dtype=np.uint8)  # type: ignore
        pcd = create_colored_point_cloud_open3d(point_cloud=sweep.xyz, rgb=rgb)
        geometries.append(pcd)

        coord_frame = draw_coordinate_frame(wTc=city_SE3_egovehicle, axis_length=0.3)
        geometries.extend(coord_frame)
        open3d.visualization.draw_geometries(geometries)


@click.command(help="Generate 3d visualizations of scenes from the Argoverse 2 Sensor / LiDAR / TbV Datasets.")
@click.option(
    "-d",
    "--data-root",
    required=True,
    help="Path to local directory where the Argoverse 2 Sensor Dataset logs are stored.",
    type=click.Path(exists=True),
)
@click.option(
    "-l",
    "--log-id",
    required=True,
    help="unique log identifier.",
    type=str,
)
def run_visualize_scene(data_root: str, log_id: str) -> None:
    """Click entry point for Open3d scene visualization."""
    visualize_scene(Path(data_root), log_id)


if __name__ == "__main__":
    run_visualize_scene()
