"""Pytorch rigid body transformations."""

import math

import torch
from torch import Tensor


@torch.jit.script  # type: ignore
def quat_to_xyz(quat_wxyz: Tensor, singularity_value: float = math.pi / 2) -> Tensor:
    """Convert scalar first quaternion to Tait-Bryan angles.

    Reference:
        https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Source_code_2

    Args:
        quat_wxyz: (...,4) Scalar first quaternions.
        singularity_value: Value that's set at the singularities.

    Returns:
        (...,3) The Tait-Bryan angles --- roll, pitch, and yaw.
    """
    qw = quat_wxyz[..., 0]
    qx = quat_wxyz[..., 1]
    qy = quat_wxyz[..., 2]
    qz = quat_wxyz[..., 3]

    # roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    pitch = 2 * (qw * qy - qz * qx)
    is_out_of_range = torch.abs(pitch) >= 1
    pitch[is_out_of_range] = torch.copysign(torch.as_tensor(singularity_value), pitch[is_out_of_range])
    pitch[~is_out_of_range] = torch.asin(pitch[~is_out_of_range])

    # yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    xyz = torch.stack([roll, pitch, yaw], dim=-1)
    return xyz


def quat_to_yaw(quat_wxyz: Tensor) -> Tensor:
    """Scalar-first quaternion to rotation around the gravity aligned axis in radians."""
    xyz: Tensor = quat_to_xyz(quat_wxyz)[:, -1]
    return xyz
