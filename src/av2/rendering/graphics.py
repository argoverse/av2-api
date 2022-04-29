# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

from typing import Final, List

import numpy as np
import vedo

from av2.structures.cuboid import CuboidList
from av2.structures.sweep import Sweep
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

COLORS: NDArrayByte = np.array(
    [[255, 255, 255], [255, 255, 255]] + [[0, 0, 255] for _ in MESH_EDGES[2:]],
    dtype=np.uint8,
)


def plotter():
    return vedo.Plotter(
        title="AV2",
        interactive=False,
        offscreen=False,
        bg="black",
        sharecam=False,
        resetcam=False,
        # backend="k3d",
    )


def points(sweep: Sweep):
    pts = vedo.Points(inputobj=sweep.xyz[..., :3].tolist(), r=4, c="grey")
    pts = pts.cmap("Spectral", sweep.xyz[..., -1], vmin=-3, vmax=3)
    return pts


def cuboids(annotations: CuboidList) -> List[vedo.Mesh]:
    vertices = annotations.vertices_m
    meshes: List[vedo.Mesh] = [
        vedo.Mesh(
            [x, MESH_EDGES],
            alpha=0.5,
        ).lighting("off")
        for x in vertices
    ]
    for mesh in meshes:
        mesh.celldata["mycolors"] = COLORS
        mesh.celldata.select("mycolors")
    return meshes


def map():
    pass
