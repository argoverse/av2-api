//! # SO(3)
//!
//! Special Orthogonal Group 3 (SO(3)).

use std::f32::consts::PI;

use ndarray::{par_azip, s, Array, Array2, ArrayView, Ix1, Ix2, Ix3};
use rand_distr::{Distribution, StandardNormal};

/// Convert a quaternion in scalar-first format to a 3x3 rotation matrix.
/// Parallelized for batch processing.
pub fn quat_to_mat3(quat_wxyz: &ArrayView<f32, Ix2>) -> Array<f32, Ix3> {
    let num_quats = quat_wxyz.shape()[0];
    let mut mat3 = Array::<f32, Ix3>::zeros((num_quats, 3, 3));
    par_azip!((mut m in mat3.outer_iter_mut(), q in quat_wxyz.outer_iter()) {
        m.assign(&_quat_to_mat3(&q));
    });
    mat3
}

/// Convert a quaternion in scalar-first format to a 3x3 rotation matrix.
pub fn _quat_to_mat3(quat_wxyz: &ArrayView<f32, Ix1>) -> Array<f32, Ix2> {
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

/// Convert a 3x3 rotation matrix to a scalar-first quaternion.
/// Parallelized for batch processing.
pub fn mat3_to_quat(mat3: &ArrayView<f32, Ix3>) -> Array<f32, Ix2> {
    let num_transformations = mat3.shape()[0];
    let mut quat_wxyz = Array::<f32, Ix2>::zeros((num_transformations, 4));
    par_azip!((mut q in quat_wxyz.outer_iter_mut(), m in mat3.outer_iter()) {
        q.assign(&_mat3_to_quat(&m))
    });
    quat_wxyz
}

/// Convert a 3x3 rotation matrix to a scalar-first quaternion.
pub fn _mat3_to_quat(mat3: &ArrayView<f32, Ix2>) -> Array<f32, Ix1> {
    let trace = mat3[[0, 0]] + mat3[[1, 1]] + mat3[[2, 2]];
    let mut quat_wxyz = if trace > 0.0 {
        let s = 0.5 / f32::sqrt(trace + 1.0);
        let qw = 0.25 / s;
        let qx = (mat3[[2, 1]] - mat3[[1, 2]]) * s;
        let qy = (mat3[[0, 2]] - mat3[[2, 0]]) * s;
        let qz = (mat3[[1, 0]] - mat3[[0, 1]]) * s;
        Array::<f32, Ix1>::from_vec(vec![qw, qx, qy, qz])
    } else if mat3[[0, 0]] > mat3[[1, 1]] && mat3[[0, 0]] > mat3[[2, 2]] {
        let s = 2.0 * f32::sqrt(1.0 + mat3[[0, 0]] - mat3[[1, 1]] - mat3[[2, 2]]);
        let qw = (mat3[[2, 1]] - mat3[[1, 2]]) / s;
        let qx = 0.25 * s;
        let qy = (mat3[[0, 1]] + mat3[[1, 0]]) / s;
        let qz = (mat3[[0, 2]] + mat3[[2, 0]]) / s;
        Array::<f32, Ix1>::from_vec(vec![qw, qx, qy, qz])
    } else if mat3[[1, 1]] > mat3[[2, 2]] {
        let s = 2.0 * f32::sqrt(1.0 + mat3[[1, 1]] - mat3[[0, 0]] - mat3[[2, 2]]);
        let qw = (mat3[[0, 2]] - mat3[[2, 0]]) / s;
        let qx = (mat3[[0, 1]] + mat3[[1, 0]]) / s;
        let qy = 0.25 * s;
        let qz = (mat3[[1, 2]] + mat3[[2, 1]]) / s;
        Array::<f32, Ix1>::from_vec(vec![qw, qx, qy, qz])
    } else {
        let s = 2.0 * f32::sqrt(1.0 + mat3[[2, 2]] - mat3[[0, 0]] - mat3[[1, 1]]);
        let qw = (mat3[[1, 0]] - mat3[[0, 1]]) / s;
        let qx = (mat3[[0, 2]] + mat3[[2, 0]]) / s;
        let qy = (mat3[[1, 2]] + mat3[[2, 1]]) / s;
        let qz = 0.25 * s;
        Array::<f32, Ix1>::from_vec(vec![qw, qx, qy, qz])
    };

    // Canonicalize the quaternion.
    if quat_wxyz[0] < 0.0 {
        quat_wxyz *= -1.0;
    }
    quat_wxyz
}

/// Convert a scalar-first quaternion to yaw.
/// In the Argoverse 2 coordinate system, this is counter-clockwise rotation about the +z axis.
/// Parallelized for batch processing.
pub fn quat_to_yaw(quat_wxyz: &ArrayView<f32, Ix2>) -> Array<f32, Ix2> {
    let num_quats = quat_wxyz.shape()[0];
    let mut yaws_rad = Array::<f32, Ix2>::zeros((num_quats, 1));
    par_azip!((mut y in yaws_rad.outer_iter_mut(), q in quat_wxyz.outer_iter()) {
        y[0] = _quat_to_yaw(&q);
    });
    yaws_rad
}

/// Convert a scalar-first quaternion to yaw.
/// In the Argoverse 2 coordinate system, this is counter-clockwise rotation about the +z axis.
pub fn _quat_to_yaw(quat_wxyz: &ArrayView<f32, Ix1>) -> f32 {
    let (qw, qx, qy, qz) = (quat_wxyz[0], quat_wxyz[1], quat_wxyz[2], quat_wxyz[3]);
    let siny_cosp = 2. * (qw * qz + qx * qy);
    let cosy_cosp = 1. - 2. * (qy * qy + qz * qz);
    siny_cosp.atan2(cosy_cosp)
}

/// Convert a scalar-first quaternion to yaw.
/// In the Argoverse 2 coordinate system, this is counter-clockwise rotation about the +z axis.
/// Parallelized for batch processing.
pub fn yaw_to_quat(yaw_rad: &ArrayView<f32, Ix2>) -> Array<f32, Ix2> {
    let num_yaws = yaw_rad.shape()[0];
    let mut quat_wxyz = Array::<f32, Ix2>::zeros((num_yaws, 4));
    par_azip!((mut q in quat_wxyz.outer_iter_mut(), y in yaw_rad.outer_iter()) {
        q.assign(&_yaw_to_quat(y[0]));
    });
    quat_wxyz
}

/// Convert rotation about the z-axis to a scalar-first quaternion.
pub fn _yaw_to_quat(yaw_rad: f32) -> Array<f32, Ix1> {
    let qw = f32::cos(0.5 * yaw_rad);
    let qz = f32::sin(0.5 * yaw_rad);
    Array::<f32, Ix1>::from_vec(vec![qw, 0.0, 0.0, qz])
}

/// Reflect orientation across the x-axis.
/// (N,4) `quat_wxyz` orientation of `N` rigid objects.
pub fn reflect_orientation_x(quat_wxyz: &ArrayView<f32, Ix2>) -> Array<f32, Ix2> {
    let yaw_rad = quat_to_yaw(quat_wxyz);
    let reflected_yaw_rad = -yaw_rad;
    yaw_to_quat(&reflected_yaw_rad.view())
}

/// Reflect orientation across the y-axis.
/// (N,4) `quat_wxyz` orientation of `N` rigid objects.
pub fn reflect_orientation_y(quat_wxyz: &ArrayView<f32, Ix2>) -> Array<f32, Ix2> {
    let yaw_rad = quat_to_yaw(quat_wxyz);
    let reflected_yaw_rad = PI - yaw_rad;
    yaw_to_quat(&reflected_yaw_rad.view())
}

/// Reflect translation across the x-axis.
pub fn reflect_translation_x(xyz_m: &ArrayView<f32, Ix2>) -> Array<f32, Ix2> {
    let mut augmented_xyz_m = xyz_m.to_owned();
    augmented_xyz_m
        .slice_mut(s![.., 1])
        .par_mapv_inplace(|y| -y);
    augmented_xyz_m
}

/// Reflect translation across the y-axis.
pub fn reflect_translation_y(xyz_m: &ArrayView<f32, Ix2>) -> Array<f32, Ix2> {
    let mut augmented_xyz_m = xyz_m.to_owned();
    augmented_xyz_m
        .slice_mut(s![.., 0])
        .par_mapv_inplace(|x| -x);
    augmented_xyz_m
}

/// Sample a random quaternion.
pub fn sample_random_quat_wxyz() -> Array<f32, Ix1> {
    let distribution = StandardNormal;
    let qw: f32 = distribution.sample(&mut rand::thread_rng());
    let qx: f32 = distribution.sample(&mut rand::thread_rng());
    let qy: f32 = distribution.sample(&mut rand::thread_rng());
    let qz: f32 = distribution.sample(&mut rand::thread_rng());
    let quat_wxyz = Array::<f32, Ix1>::from_vec(vec![qw, qx, qy, qz]);
    let norm = quat_wxyz.dot(&quat_wxyz).sqrt();
    let mut versor_wxyz = quat_wxyz / norm;

    // Canonicalize the quaternion.
    if versor_wxyz[0] < 0.0 {
        versor_wxyz *= -1.0;
    }
    versor_wxyz
}

#[cfg(test)]
mod tests {
    use super::{_mat3_to_quat, _quat_to_mat3, sample_random_quat_wxyz};

    #[test]
    fn test_quat_to_mat3_round_trip() {
        let num_trials = 100000;
        let epsilon = 1e-6;
        for _ in 0..num_trials {
            let quat_wxyz = sample_random_quat_wxyz();
            let mat3 = _quat_to_mat3(&quat_wxyz.view());
            let _quat_wxyz = _mat3_to_quat(&mat3.view());
            assert!(quat_wxyz.abs_diff_eq(&_quat_wxyz, epsilon));
        }
    }
}
