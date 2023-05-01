//! # SO(3)
//!
//! Special Orthogonal Group 3 (SO(3)).

use ndarray::{Array, Array2, ArrayView, Ix1, Ix2, NdFloat};
use num_traits::NumCast;

/// Convert a quaternion in scalar-first format to a 3x3 rotation matrix.
pub fn quat_to_mat3<T>(quat_wxyz: &ArrayView<T, Ix1>) -> Array<T, Ix2>
where
    T: NdFloat,
{
    let w = quat_wxyz[0];
    let x = quat_wxyz[1];
    let y = quat_wxyz[2];
    let z = quat_wxyz[3];

    let one = T::one();
    let two: T = NumCast::from(2.0).unwrap();

    let e_00 = one - two * y.powi(2) - two * z.powi(2);
    let e_01 = two * x * y - two * z * w;
    let e_02 = two * x * z + two * y * w;

    let e_10 = two * x * y + two * z * w;
    let e_11 = one - two * x.powi(2) - two * z.powi(2);
    let e_12 = two * y * z - two * x * w;

    let e_20 = two * x * z - two * y * w;
    let e_21 = two * y * z + two * x * w;
    let e_22 = one - two * x.powi(2) - two * y.powi(2);

    // Safety: We will always have nine elements.
    unsafe {
        Array2::from_shape_vec_unchecked(
            [3, 3],
            vec![e_00, e_01, e_02, e_10, e_11, e_12, e_20, e_21, e_22],
        )
    }
}
