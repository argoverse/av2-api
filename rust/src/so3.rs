//! # SO(3)
//!
//! Special Orthogonal Group 3 (SO(3)).

use std::f32::consts::PI;

use ndarray::{par_azip, Array, Array2, ArrayView, Ix1, Ix2};

/// Convert a quaternion in scalar-first format to a 3x3 rotation matrix.
pub fn quat_to_mat3(quat_wxyz: &ArrayView<f32, Ix1>) -> Array<f32, Ix2> {
    let w = quat_wxyz[0];
    let x = quat_wxyz[1];
    let y = quat_wxyz[2];
    let z = quat_wxyz[3];

    let e_00 = 1. - 2. * y.powi(2) - 2. * z.powi(2);
    let e_01: f32 = 2. * x * y - 2. * z * w;
    let e_02: f32 = 2. * x * z + 2. * y * w;

    let e_10 = 2. * x * y + 2. * z * w;
    let e_11 = 1. - 2. * x.powi(2) - 2. * z.powi(2);
    let e_12 = 2. * y * z - 2. * x * w;

    let e_20 = 2. * x * z - 2. * y * w;
    let e_21 = 2. * y * z + 2. * x * w;
    let e_22 = 1. - 2. * x.powi(2) - 2. * y.powi(2);

    // Safety: We will always have nine elements.
    unsafe {
        Array2::from_shape_vec_unchecked(
            [3, 3],
            vec![e_00, e_01, e_02, e_10, e_11, e_12, e_20, e_21, e_22],
        )
    }
}

/// Convert a scalar-first quaternion to yaw.
/// In the Argoverse 2 coordinate system, this is counter-clockwise rotation about the +z axis.
/// Parallelized for batch processing.
pub fn quat_to_yaw(quat_wxyz: ArrayView<f32, Ix2>) -> Array<f32, Ix2> {
    let num_quats = quat_wxyz.shape()[0];
    let mut yaws_rad = Array::<f32, Ix2>::zeros((num_quats, 1));
    par_azip!((mut y in yaws_rad.outer_iter_mut(), q in quat_wxyz.outer_iter()) {
        y[0] = _quat_to_yaw(q);
    });
    yaws_rad
}

/// Convert a scalar-first quaternion to yaw.
/// In the Argoverse 2 coordinate system, this is counter-clockwise rotation about the +z axis.
pub fn _quat_to_yaw(quat_wxyz: ArrayView<f32, Ix1>) -> f32 {
    let (qw, qx, qy, qz) = (quat_wxyz[0], quat_wxyz[1], quat_wxyz[2], quat_wxyz[3]);
    let siny_cosp = 2. * (qw * qz + qx * qy);
    let cosy_cosp = 1. - 2. * (qy * qy + qz * qz);
    siny_cosp.atan2(cosy_cosp)
}

/// Convert a scalar-first quaternion to yaw.
/// In the Argoverse 2 coordinate system, this is counter-clockwise rotation about the +z axis.
/// Parallelized for batch processing.
pub fn yaw_to_quat(yaw_rad: ArrayView<f32, Ix2>) -> Array<f32, Ix2> {
    let num_yaws = yaw_rad.shape()[0];
    let mut quat_wxyz = Array::<f32, Ix2>::zeros((num_yaws, 4));
    par_azip!((mut q in quat_wxyz.outer_iter_mut(), y in yaw_rad.outer_iter()) {
        q.assign(&_yaw_to_quat(y[0]));
    });
    quat_wxyz
}

/// Convert rotation about the z-axis to a scalar-first quaternion.
pub fn _yaw_to_quat(yaw_rad: f32) -> Array<f32, Ix1> {
    let cy = f32::cos(0.5 * yaw_rad);
    let sy = f32::sin(0.5 * yaw_rad);

    // pitch_rad = 0.0
    // cos(0.5 * pitch_rad) = 1.0
    let cp = 1.0;

    // pitch_rad = 0.0
    // sin(0.5 * pitch_rad) = 0.0
    let sp = 0.0;

    // roll_rad = 0.0
    // cos(0.5 * roll_rad) = 1.0
    let cr = 1.0;

    // roll_rad = 0.0
    // sin(0.5 * roll_rad) = 0.0
    let sr = 0.0;

    let qw = cr * cp * cy + sr * sp * sy;
    let qx = sr * cp * cy - cr * sp * sy;
    let qy = cr * sp * cy + sr * cp * sy;
    let qz = cr * cp * sy - sr * sp * cy;
    Array::<f32, Ix1>::from_vec(vec![qw, qx, qy, qz])
}

/// Reflect pose across the x-axis.
pub fn reflect_pose_x(quat_wxyz: ArrayView<f32, Ix2>) -> Array<f32, Ix2> {
    let yaw_rad = quat_to_yaw(quat_wxyz);
    let reflected_yaw_rad = -yaw_rad;
    yaw_to_quat(reflected_yaw_rad.view())
}

/// Reflect pose across the y-axis.
pub fn reflect_pose_y(quat_wxyz: ArrayView<f32, Ix2>) -> Array<f32, Ix2> {
    let yaw_rad = quat_to_yaw(quat_wxyz);
    let reflected_yaw_rad = yaw_rad - PI;
    yaw_to_quat(reflected_yaw_rad.view())
}
