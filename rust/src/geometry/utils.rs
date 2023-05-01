//! # utils
//!
//! Geometric utilities.

use ndarray::{s, Array, Ix2, NdFloat};

/// Convert Cartesian coordinates into Homogeneous coordinates.
/// This function converts a set of points in R^N to its homogeneous representation in R^(N+1).
pub fn cart_to_hom<T>(cart: Array<T, Ix2>) -> Array<T, Ix2>
where
    T: NdFloat,
{
    let num_points = cart.shape()[0];
    let num_dims = cart.shape()[1];
    let mut hom = Array::<T, Ix2>::ones([num_points, num_dims + 1]);
    hom.slice_mut(s![.., ..num_dims]).assign(&cart);
    hom
}
