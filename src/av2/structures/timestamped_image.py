# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Timestamped image class with camera model for synchronization."""

from dataclasses import dataclass

from av2.geometry.camera.pinhole_camera import PinholeCamera
from av2.utils.typing import NDArrayByte


@dataclass(frozen=True)
class TimestampedImage:
    """Timestamped image with an accompanying camera model.

    Args:
        img: (H,W,C) image.
        camera_model: Pinhole camera model with intrinsics and extrinsics.
        timestamp_ns: Nanosecond timestamp.

    """

    img: NDArrayByte
    camera_model: PinholeCamera
    timestamp_ns: int
