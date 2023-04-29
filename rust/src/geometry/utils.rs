//! # utils
//!
//! Geometric utilities.

use ndarray::{s, Array, Ix2};

/// Convert Cartesian coordinates into Homogeneous coordinates.
/// This function converts a set of points in R^N to its homogeneous representation in R^(N+1).
pub fn cart_to_hom(cart: Array<f32, Ix2>) -> Array<f32, Ix2> {
    let num_points = cart.shape()[0];
    let num_dims = cart.shape()[1];
    let mut hom = Array::<f32, Ix2>::ones([num_points, num_dims + 1]);
    hom.slice_mut(s![.., ..num_dims]).assign(&cart);
    hom
}
