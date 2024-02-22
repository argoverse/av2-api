"""Timestamped image class with camera model for synchronization."""

from dataclasses import dataclass

import torch
from kornia.geometry.camera import PinholeCamera


@dataclass
class TimeStampedImage:
    """Timestamped image with an accompanying camera model.

    Args:
        image: (H,W,C) image.
        camera_model: Pinhole camera model with intrinsics and extrinsics.
        timestamp_ns: Nanosecond timestamp.
    """

    image: torch.Tensor
    camera_model: PinholeCamera
    timestamp_ns: int
