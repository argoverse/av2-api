//! # SO3
//!
//! Special Orthogonal Group in 3 Dimensions.

use ndarray::{Array2, ArrayView1};

pub fn quat_to_mat(quat_wxyz: &ArrayView1<f32>) -> Array2<f32> {
    let w = quat_wxyz[0];
    let x = quat_wxyz[1];
    let y = quat_wxyz[2];
    let z = quat_wxyz[3];

    let tx = 2. * x;
    let ty = 2. * y;
    let tz = 2. * z;
    let twx = tx * w;
    let twy = ty * w;
    let twz = tz * w;
    let txx = tx * x;
    let txy = ty * x;
    let txz = tz * x;
    let tyy = ty * y;
    let tyz = tz * y;
    let tzz = tz * z;

    let e_00 = 1. - (tyy + tzz);
    let e_01 = txy - twz;
    let e_02 = txy + twy;
    let e_10 = txy + twz;
    let e_11 = 1. - (txx + tzz);
    let e_12 = tyz - twx;
    let e_20 = txz - twy;
    let e_21 = tyz + twx;
    let e_22 = 1. - (txx + tyy);

    // Safety: We will always have nine elements.
    unsafe {
        Array2::from_shape_vec_unchecked(
            [3, 3],
            vec![e_00, e_01, e_02, e_10, e_11, e_12, e_20, e_21, e_22],
        )
    }
}
