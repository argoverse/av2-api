# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Unit tests for drawing ops."""

from typing import Any, Callable

import cv2
import numpy as np

from av2.rendering.ops.draw import (
    alpha_blend_kernel,
    draw_points_kernel,
    gaussian_kernel,
)
from av2.utils.typing import NDArrayByte, NDArrayInt


def _draw_points_cv2(
    img: NDArrayByte,
    points_xy: NDArrayInt,
    colors: NDArrayByte,
    radius: int,
    with_anti_alias: bool = True,
) -> NDArrayByte:
    """Draw points in an image using OpenCV functionality.

    Args:
        img: (H,W,3) BGR image.
        points_xy: (N,2) Array of points.
        colors: (N,3) Array of BGR colors.
        radius: Radius of the circles.
        with_anti_alias: Boolean to enable anti-aliasing.

    Returns:
        The image with points overlaid.
    """
    line_type = cv2.LINE_AA if with_anti_alias else None
    for i, (x, y) in enumerate(points_xy):
        rgb = colors[i]
        rgb = tuple([int(intensity) for intensity in rgb])
        cv2.circle(img, (x, y), radius, (0, 0, 0), -1, line_type)
    return img


def test_alpha_blend_kernel() -> None:
    """Unit test alpha blending."""
    fg: NDArrayByte = np.array([128, 128, 128], dtype=np.uint8)
    bg: NDArrayByte = np.array([64, 64, 64], dtype=np.uint8)

    alpha = 0
    blended_pixel = alpha_blend_kernel(fg, bg, alpha)
    assert blended_pixel == (64, 64, 64)

    alpha = 128
    blended_pixel = alpha_blend_kernel(fg, bg, alpha)
    assert blended_pixel == (96, 96, 96)

    alpha = 255
    blended_pixel = alpha_blend_kernel(fg, bg, alpha)
    assert blended_pixel == (128, 128, 128)


def test_gaussian_kernel() -> None:
    """Unit test for a Gaussian kernel."""
    x = 0
    mu = 0
    sigma = 1
    out = gaussian_kernel(x=x, mu=mu, sigma=sigma)

    assert out == 1.0
    assert gaussian_kernel(x=x - 1, mu=mu, sigma=sigma) < out
    assert gaussian_kernel(x=x + 1, mu=mu, sigma=sigma) < out


def test_draw_points_kernel_3x3_antialiased() -> None:
    """Unit test drawing points on an image."""
    img: NDArrayByte = np.zeros((3, 3, 3), dtype=np.uint8)
    points_xy: NDArrayInt = np.array([[1, 1]], dtype=int)
    colors: NDArrayByte = np.array([[128, 128, 128]], dtype=np.uint8)
    sigma = 1.0
    diameter = 3

    expected_img: NDArrayByte = np.array(
        [
            [[47, 47, 47], [77, 77, 77], [47, 47, 47]],
            [[77, 77, 77], [128, 128, 128], [77, 77, 77]],
            [[47, 47, 47], [77, 77, 77], [47, 47, 47]],
        ],
        dtype=np.uint8,
    )
    img = draw_points_kernel(
        img=img,
        points_uv=points_xy,
        colors=colors,
        diameter=diameter,
        sigma=sigma,
        with_anti_alias=True,
    )
    assert np.array_equal(img, expected_img)


def test_draw_points_kernel_9x9_aliased() -> None:
    """Unit test drawing points on an image."""
    img: NDArrayByte = np.zeros((9, 9, 3), dtype=np.uint8)
    points_xy: NDArrayInt = np.array([[4, 4]], dtype=int)
    colors: NDArrayByte = np.array([[128, 128, 128]], dtype=np.uint8)
    sigma = 1.0
    diameter = 3

    expected_img = np.zeros((9, 9, 3), dtype=np.uint8)

    expected_img[3:6, 3:6] = np.array(
        [
            [[128, 128, 128], [128, 128, 128], [128, 128, 128]],
            [[128, 128, 128], [128, 128, 128], [128, 128, 128]],
            [[128, 128, 128], [128, 128, 128], [128, 128, 128]],
        ],
        dtype=np.uint8,
    )
    img = draw_points_kernel(
        img=img,
        points_uv=points_xy,
        colors=colors,
        diameter=diameter,
        sigma=sigma,
        with_anti_alias=False,
    )
    assert np.array_equal(img, expected_img)


def test_benchmark_draw_points_kernel_aliased(benchmark: Callable[..., Any]) -> None:
    """Benchmark the draw points kernel _without_ anti-aliasing."""
    img: NDArrayByte = np.zeros((2048, 2048, 3), dtype=np.uint8)
    points_xy: NDArrayInt = np.random.randint(low=0, high=2048, size=(60000, 2)).astype(
        np.int64
    )
    colors: NDArrayByte = np.random.randint(low=0, high=255, size=(60000, 3)).astype(
        np.uint8
    )
    diameter = 10
    benchmark(draw_points_kernel, img, points_xy, colors, diameter)


def test_benchmark_draw_points_kernel_anti_aliased(
    benchmark: Callable[..., Any]
) -> None:
    """Benchmark the draw points kernel _with_ anti-aliasing."""
    img: NDArrayByte = np.zeros((2048, 2048, 3), dtype=np.uint8)
    points_xy: NDArrayInt = np.random.randint(low=0, high=2048, size=(60000, 2)).astype(
        np.int64
    )
    colors: NDArrayByte = np.random.randint(low=0, high=255, size=(60000, 3)).astype(
        np.uint8
    )
    diameter = 10
    benchmark(
        draw_points_kernel, img, points_xy, colors, diameter, with_anti_alias=True
    )


def test_benchmark_draw_points_cv2_aliased(benchmark: Callable[..., Any]) -> None:
    """Benchmark the draw points method from OpenCV _without_ anti-aliasing."""
    img: NDArrayByte = np.zeros((2048, 2048, 3), dtype=np.uint8)
    points_xy: NDArrayInt = np.random.randint(low=0, high=2048, size=(60000, 2)).astype(
        np.int64
    )
    colors: NDArrayByte = np.random.randint(low=0, high=255, size=(60000, 3)).astype(
        np.uint8
    )
    radius = 10
    benchmark(_draw_points_cv2, img, points_xy, colors, radius, with_anti_alias=False)


def test_benchmark_draw_points_cv2_anti_aliased(benchmark: Callable[..., Any]) -> None:
    """Benchmark the draw points method from OpenCV _with_ anti-aliasing."""
    img: NDArrayByte = np.zeros((2048, 2048, 3), dtype=np.uint8)
    points_xy: NDArrayInt = np.random.randint(low=0, high=2048, size=(60000, 2)).astype(
        np.int64
    )
    colors: NDArrayByte = np.random.randint(low=0, high=255, size=(60000, 3)).astype(
        np.uint8
    )
    radius = 10
    benchmark(_draw_points_cv2, img, points_xy, colors, radius, with_anti_alias=True)
