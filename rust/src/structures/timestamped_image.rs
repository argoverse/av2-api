use ndarray::{Array, Ix3};

use crate::geometry::camera::pinhole_camera::PinholeCamera;

/// Image modeled by `camera_model` occuring at `timestamp_ns`.
#[derive(Clone)]
pub struct TimeStampedImage {
    /// (H,W,C) image.
    pub image: Array<u8, Ix3>,
    /// Pinhole camera model with intrinsics and extrinsics.
    pub camera_model: PinholeCamera,
    /// Nanosecond timestamp.
    pub timestamp_ns: usize,
}
