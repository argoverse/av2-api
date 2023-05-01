use image::{ImageBuffer, Rgba};

use crate::geometry::camera::pinhole_camera::PinholeCamera;

/// Image modeled by `camera_model` occuring at `timestamp_ns`.
#[derive(Clone)]
pub struct TimeStampedImage {
    /// RGBA u8 image buffer.
    pub image: ImageBuffer<Rgba<u8>, Vec<u8>>,
    /// Pinhole camera model with intrinsics and extrinsics.
    pub camera_model: PinholeCamera,
    /// Nanosecond timestamp.
    pub timestamp_ns: usize,
}
