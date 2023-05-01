# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Units tests for cuboids."""

from enum import Enum
from typing import Any, Callable, List, Tuple

import numpy as np
import pytest

from av2.datasets.sensor.constants import AnnotationCategories
from av2.geometry.se3 import SE3
from av2.structures.cuboid import Cuboid, CuboidList
from av2.utils.typing import NDArrayBool, NDArrayFloat


def test_vertices() -> None:
    """Ensure that 8 cuboid vertices are located where we expect.

    Cuboid center is placed at the origin.
    """
    dst_SE3_object = SE3(rotation=np.eye(3), translation=np.zeros(3))
    cuboid = Cuboid(
        dst_SE3_object=dst_SE3_object,
        length_m=3,
        width_m=2,
        height_m=1,
        category=AnnotationCategories.REGULAR_VEHICLE,
        timestamp_ns=0,  # dummy value
    )
    vertices = cuboid.vertices_m

    expected_vertices: NDArrayFloat = np.array(
        [
            [1.5, 1, 0.5],
            [1.5, -1, 0.5],
            [1.5, -1, -0.5],
            [1.5, 1, -0.5],
            [-1.5, 1, 0.5],
            [-1.5, -1, 0.5],
            [-1.5, -1, -0.5],
            [-1.5, 1, -0.5],
        ]
    )
    assert np.array_equal(vertices, expected_vertices)


def test_compute_interior_points() -> None:
    """Ensure that a point cloud is filtered correctly to a cuboid's interior.

    Cuboid center is placed at (2,0,0) in the egovehicle frame here.
    """
    # fmt: off
    points_xyz: NDArrayFloat = np.array(
        [
            [-1, 0, 0],
            [0, 0, 0],  # start
            [1, 0, 0],
            [2, 0, 0],
            [3, 0, 0],
            [4, 0, 0],  # end
            [5, 0, 0]
        ], dtype=float  
    )
    # fmt: on
    expected_is_interior: NDArrayBool = np.array(
        [False, True, True, True, True, True, False]
    )

    dst_SE3_object = SE3(rotation=np.eye(3), translation=np.array([2, 0, 0]))

    cuboid = Cuboid(
        dst_SE3_object=dst_SE3_object,
        length_m=4,
        width_m=2,
        height_m=1,
        category=AnnotationCategories.REGULAR_VEHICLE,
        timestamp_ns=0,  # dummy value
    )
    interior_pts, is_interior = cuboid.compute_interior_points(points_xyz)
    expected_interior_pts: NDArrayFloat = np.array(
        [
            [0, 0, 0],  # start
            [1, 0, 0],
            [2, 0, 0],
            [3, 0, 0],
            [4, 0, 0],  # end
        ],
        dtype=float,
    )
    assert np.array_equal(interior_pts, expected_interior_pts)
    assert np.array_equal(is_interior, expected_is_interior)


def _get_dummy_cuboid_params() -> Tuple[SE3, float, float, float, Enum, int]:
    """Create cuboid parameters to construct a `Cuboid` object."""
    rotation: NDArrayFloat = np.eye(3)
    translation: NDArrayFloat = np.array([1.0, 0.0, 0.0])
    ego_SE3_obj = SE3(rotation=rotation, translation=translation)
    length_m = 3.0
    width_m = 2.0
    height_m = 1.0
    category = AnnotationCategories.BUS
    timestamp_ns = 0

    return ego_SE3_obj, length_m, width_m, height_m, category, timestamp_ns


def _get_dummy_cuboid_list_params(num_cuboids: int) -> List[Cuboid]:
    """Create a cuboid list of length `num_cuboids`."""
    cuboids: List[Cuboid] = []
    for i in range(num_cuboids):
        (
            ego_SE3_object,
            length_m,
            width_m,
            height_m,
            category,
            timestamp_ns,
        ) = _get_dummy_cuboid_params()
        cuboid = Cuboid(
            dst_SE3_object=ego_SE3_object,
            length_m=length_m + i,
            width_m=width_m + i,
            height_m=height_m + i,
            category=category,
            timestamp_ns=timestamp_ns,
        )
        cuboids.append(cuboid)
    return cuboids


def test_cuboid_constructor() -> None:
    """Test initializing a single cuboid."""
    (
        ego_SE3_object,
        length_m,
        width_m,
        height_m,
        category,
        timestamp_ns,
    ) = _get_dummy_cuboid_params()
    cuboid = Cuboid(
        dst_SE3_object=ego_SE3_object,
        length_m=length_m,
        width_m=width_m,
        height_m=height_m,
        category=category,
        timestamp_ns=timestamp_ns,
    )

    rotation_expected = ego_SE3_object.rotation
    translation_expected = ego_SE3_object.translation
    length_m_expected = length_m
    width_m_expected = width_m
    height_m_expected = height_m

    assert isinstance(cuboid, Cuboid)
    assert np.array_equal(cuboid.dst_SE3_object.rotation, rotation_expected)
    assert np.array_equal(cuboid.dst_SE3_object.translation, translation_expected)
    assert cuboid.length_m == length_m_expected
    assert cuboid.width_m == width_m_expected
    assert cuboid.height_m == height_m_expected


def test_cuboid_list_constructor_single() -> None:
    """Test initializing a single cuboid."""
    num_cuboids = 1
    cuboids = _get_dummy_cuboid_list_params(num_cuboids)
    cuboid_list = CuboidList(cuboids)
    assert isinstance(cuboid_list, CuboidList)


def test_cuboid_list_constructor_multiple() -> None:
    """Test initializing a list of cuboids."""
    num_cuboids = 50
    cuboids = _get_dummy_cuboid_list_params(num_cuboids)
    cuboid_list = CuboidList(cuboids)
    assert isinstance(cuboid_list, CuboidList)


def test_getitem() -> None:
    """Ensure __getitem__ works as expected for access operations."""
    num_cuboids = 50
    cuboids = _get_dummy_cuboid_list_params(num_cuboids)
    cuboid_list = CuboidList(cuboids)
    assert isinstance(cuboid_list, CuboidList)

    assert isinstance(cuboid_list[5], Cuboid)

    with pytest.raises(IndexError):
        cuboid_list[-1]

    with pytest.raises(IndexError):
        cuboid_list[50]

    with pytest.raises(IndexError):
        cuboid_list[51]


def test_benchmark_cuboid_list_1k(benchmark: Callable[..., Any]) -> None:
    """Test initializing a list of 1000 cuboids."""
    num_cuboids = 1000
    cuboids = _get_dummy_cuboid_list_params(num_cuboids)
    benchmark(CuboidList, cuboids)


def test_benchmark_cuboid_list_10k(benchmark: Callable[..., Any]) -> None:
    """Test initializing a list of 10,000 cuboids."""
    num_cuboids = 10000
    cuboids = _get_dummy_cuboid_list_params(num_cuboids)
    benchmark(CuboidList, cuboids)


def test_benchmark_cuboid_list_100k(benchmark: Callable[..., Any]) -> None:
    """Test initializing a list of 100,000 cuboids."""
    num_cuboids = 100000
    cuboids = _get_dummy_cuboid_list_params(num_cuboids)
    benchmark(CuboidList, cuboids)


def test_benchmark_transform_for_loop(benchmark: Callable[..., Any]) -> None:
    """Benchmark cuboid transform with a for loop."""

    def benchmark_transform(cuboids: List[Cuboid], target_SE3_ego: SE3) -> List[Cuboid]:
        transformed_cuboids: List[Cuboid] = []
        for cuboid in cuboids:
            target_SE3_object = target_SE3_ego.compose(cuboid.dst_SE3_object)
            transformed_cuboid = Cuboid(
                dst_SE3_object=target_SE3_object,
                length_m=cuboid.length_m,
                width_m=cuboid.width_m,
                height_m=cuboid.height_m,
                category=cuboid.category,
                timestamp_ns=cuboid.timestamp_ns,
            )
            transformed_cuboids.append(transformed_cuboid)
        return transformed_cuboids

    num_cuboids = 1000
    cuboids = _get_dummy_cuboid_list_params(num_cuboids)
    benchmark(benchmark_transform, cuboids, cuboids[0].dst_SE3_object)


def test_benchmark_transform_list_comprehension(benchmark: Callable[..., Any]) -> None:
    """Benchmark cuboid transform with list comprehension."""

    def benchmark_transform_list_comprehension(
        cuboids: List[Cuboid], target_SE3_ego: SE3
    ) -> List[Cuboid]:
        transformed_cuboids: List[Cuboid] = [
            Cuboid(
                dst_SE3_object=target_SE3_ego.compose(cuboid.dst_SE3_object),
                length_m=cuboid.length_m,
                width_m=cuboid.width_m,
                height_m=cuboid.height_m,
                category=cuboid.category,
                timestamp_ns=cuboid.timestamp_ns,
            )
            for cuboid in cuboids
        ]
        return transformed_cuboids

    num_cuboids = 1000
    cuboids = _get_dummy_cuboid_list_params(num_cuboids)
    benchmark(
        benchmark_transform_list_comprehension, cuboids, cuboids[0].dst_SE3_object
    )


if __name__ == "__main__":
    test_getitem()
