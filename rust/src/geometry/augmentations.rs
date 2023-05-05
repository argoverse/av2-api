//! # augmentations
//!
//! Geometric augmentations.

use ndarray::{Array, ArrayView, Ix2};
use rand_distr::{Bernoulli, Distribution};

use super::se3::{reflect_pose_x, reflect_pose_y};

/// Reflect all poses across the x-axis with probability `p` (sampled from a Bernoulli distribution).
pub fn sample_global_reflect_pose_x(xyzlwh_qwxyz: ArrayView<f32, Ix2>, p: f64) -> Array<f32, Ix2> {
    let d = Bernoulli::new(p).unwrap();
    if d.sample(&mut rand::thread_rng()) {
        return reflect_pose_x(&xyzlwh_qwxyz.view());
    }
    xyzlwh_qwxyz.to_owned()
}

/// Reflect all poses across the y-axis with probability `p` (sampled from a Bernoulli distribution).
pub fn sample_global_reflect_pose_y(xyzlwh_qwxyz: ArrayView<f32, Ix2>, p: f64) -> Array<f32, Ix2> {
    let d = Bernoulli::new(p).unwrap();
    if d.sample(&mut rand::thread_rng()) {
        return reflect_pose_y(&xyzlwh_qwxyz.view());
    }
    xyzlwh_qwxyz.to_owned()
}
