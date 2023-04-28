//! # SO(3)
//!
//! Special Orthogonal Group 3 (SO(3)).

use ndarray::{Array2, ArrayView1};

/// Convert a quaternion in scalar-first format to a 3x3 rotation matrix.
pub fn quat_to_mat3(quat_wxyz: &ArrayView1<f32>) -> Array2<f32> {
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
