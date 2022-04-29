# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

from typing import Final, List

import numpy as np
import vedo
from av2.geometry.se3 import SE3
from av2.map.map_api import ArgoverseStaticMap

from av2.structures.cuboid import CuboidList
from av2.structures.sweep import Sweep
from av2.structures.timestamped_image import TimestampedImage
from av2.utils.typing import NDArrayByte, NDArrayFloat
import cv2

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

COLORS: NDArrayByte = np.array(
    [[128, 128, 128], [128, 128, 128]] + [[0, 0, 255] for _ in MESH_EDGES[2:]],
    dtype=np.uint8,
)


def plotter():
    return vedo.Plotter(
        title="AV2",
        # shape="1|7",
        interactive=False,
        offscreen=False,
        bg=[32,32,32],
        sharecam=False,
        resetcam=False,
        # backend="k3d",
    )

def imagery(imagery: List[TimestampedImage]):
    # breakpoint()
    imgs = [cv2.cvtColor(v.img, cv2.COLOR_BGR2RGB) for v in imagery.values()]
    imgs = [vedo.Picture(v) for v in imgs]
    return [vedo.Picture(v) for v in imgs]

def egovehicle():
    return vedo.Sphere(r=0.5, c="white")


def points(sweep: Sweep, colors: List[List[int]]):
    pts = vedo.Points(inputobj=sweep.xyz[..., :3].tolist(), r=4, c=colors).lighting("off")
    # pts = pts.cmap("magma", sweep.xyz[..., -1], vmin=-5, vmax=5)
    return pts


def cuboids(annotations: CuboidList) -> List[vedo.Mesh]:
    vertices = annotations.vertices_m
    meshes: List[vedo.Mesh] = [
        vedo.Mesh(
            [x, MESH_EDGES],
            alpha=0.4,
        ).lighting("off")
        for x in vertices
    ]
    for mesh in meshes:
        mesh.celldata["mycolors"] = COLORS
        mesh.celldata.select("mycolors")
    return meshes


def lanes(avm: ArgoverseStaticMap, city_SE3_ego: SE3, alpha: float = 0.5):
    lanes = []
    lane_segments = avm.get_nearby_lane_segments(city_SE3_ego.translation[:2], search_radius_m=75.0)
    for lane in lane_segments:
        # avm.get_nearby_lane_segments()
        left_lane = np.array([[point.x, point.y, point.z] for point in lane.left_lane_boundary.waypoints])
        right_lane = np.array([[point.x, point.y, point.z] for point in lane.right_lane_boundary.waypoints])

        left_lane = city_SE3_ego.inverse().transform_from(left_lane)
        right_lane = city_SE3_ego.inverse().transform_from(right_lane)
        left_lane = vedo.Line(left_lane.tolist(), lw=10, c="grey", alpha=alpha)
        right_lane = vedo.Line(right_lane.tolist(), lw=10, c="grey", alpha=alpha)
        lanes.append(left_lane)
        lanes.append(right_lane)
    return lanes


def map():
    pass
