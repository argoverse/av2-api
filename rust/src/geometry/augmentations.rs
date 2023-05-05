//! # augmentations
//!
//! Geometric augmentations.

use std::f32::consts::PI;

use ndarray::{Array, ArrayView, Ix2};

use crate::geometry::so3::{quat_to_yaw, yaw_to_quat};

/// Reflect pose across the x-axis.
pub fn reflect_pose_x(quat_wxyz: ArrayView<f32, Ix2>) -> Array<f32, Ix2> {
    let yaw_rad = quat_to_yaw(quat_wxyz);
    let reflected_yaw_rad = -yaw_rad;
    yaw_to_quat(reflected_yaw_rad.view())
}

/// Reflect pose across the y-axis.
pub fn reflect_pose_y(quat_wxyz: ArrayView<f32, Ix2>) -> Array<f32, Ix2> {
    let yaw_rad = quat_to_yaw(quat_wxyz);
    let reflected_yaw_rad = PI - yaw_rad;
    yaw_to_quat(reflected_yaw_rad.view())
}
