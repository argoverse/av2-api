# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Methods to construct Vedo objects for rendering."""

from typing import Dict, Final, List

import cv2
import numpy as np
from vedo import Line, Mesh, Picture, Plotter, Points, Sphere

from av2.geometry.se3 import SE3
from av2.map.map_api import ArgoverseStaticMap
from av2.structures.cuboid import CuboidList
from av2.structures.sweep import Sweep
from av2.structures.timestamped_image import TimestampedImage
from av2.utils.typing import NDArrayByte, NDArrayFloat

MESH_EDGES: Final[List[List[int]]] = [
    [0, 1, 2],
    [0, 2, 3],
    [0, 4, 5],
    [0, 5, 1],
    [0, 3, 7],
    [0, 7, 4],
    [1, 6, 2],
    [1, 5, 6],
    [4, 5, 6],
    [4, 6, 7],
]

MESH_CUBOID_COLORS: NDArrayByte = np.array(
    [[128, 128, 128], [128, 128, 128]] + [[0, 0, 255] for _ in MESH_EDGES[2:]],
    dtype=np.uint8,
)


def plotter() -> Plotter:
    """Return a plotter object for rendering."""
    return Plotter(
        title="AV2",
        interactive=False,
        offscreen=False,
        bg=[32, 32, 32],
        sharecam=False,
        resetcam=False,
        backend="k3d",
    )


def imagery(imagery: Dict[str, TimestampedImage]) -> List[Picture]:
    """Return ring camera imagery for rendering."""
    imgs: List[NDArrayByte] = [cv2.cvtColor(v.img, cv2.COLOR_BGR2RGB) for v in imagery.values()]
    return [Picture(v) for v in imgs]


def egovehicle() -> Sphere:
    """Return a sphere representing the egovehicle."""
    return Sphere(r=0.5, c="white")


def point_cloud(sweep: Sweep, colors: List[List[int]]) -> Points:
    """Return a rendered point cloud object.

    Args:
        sweep: Lidar sweep.
        colors: List of RGB colors.

    Returns:
        Rendered points object.
    """
    points: Points = Points(inputobj=sweep.xyz[..., :3].tolist(), r=4, c=colors).lighting("off")
    return points


def cuboids(cuboid_list: CuboidList) -> List[Mesh]:
    """Return meshes representing ground truth annotations or detections.

    Args:
        cuboid_list: List of cuboids.

    Returns:
        List of mesh objects.
    """
    vertices = cuboid_list.vertices_m
    meshes: List[Mesh] = [
        Mesh(
            [x, MESH_EDGES],
            alpha=0.4,
        ).lighting("off")
        for x in vertices
    ]
    for mesh in meshes:
        mesh.celldata["mycolors"] = MESH_CUBOID_COLORS
        mesh.celldata.select("mycolors")
    return meshes


def lanes(avm: ArgoverseStaticMap, city_SE3_ego: SE3, alpha: float = 0.5) -> List[Line]:
    """Return lanes represented as rendered lines.

    Args:
        avm: AV2 static maps for a particular log.
        city_SE3_ego: Egopose the city reference frame.
        alpha: Opacity for the rendered lines.

    Returns:
        List of rendered lines.
    """
    lanes: List[Line] = []
    lane_segments = avm.get_nearby_lane_segments(city_SE3_ego.translation[:2], search_radius_m=75.0)
    for lane in lane_segments:
        left_lane: NDArrayFloat = np.array([[point.x, point.y, point.z] for point in lane.left_lane_boundary.waypoints])
        right_lane: NDArrayFloat = np.array(
            [[point.x, point.y, point.z] for point in lane.right_lane_boundary.waypoints]
        )

        left_lane = city_SE3_ego.inverse().transform_from(left_lane)
        right_lane = city_SE3_ego.inverse().transform_from(right_lane)
        left_lane = Line(left_lane.tolist(), lw=10, c="grey", alpha=alpha)
        right_lane = Line(right_lane.tolist(), lw=10, c="grey", alpha=alpha)
        lanes.append(left_lane)
        lanes.append(right_lane)
    return lanes
