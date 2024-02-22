use image::{ImageBuffer, Rgba};
use ndarray::Ix3;
use nshare::RefNdarray3;
use numpy::{PyArray, ToPyArray};

use crate::geometry::camera::pinhole_camera::PinholeCamera;
use pyo3::prelude::*;

/// Image modeled by `camera_model` occuring at `timestamp_ns`.
// #[pyclass]
#[derive(Clone, Debug)]
#[pyclass]
pub struct TimeStampedImage {
    /// RGBA u8 image buffer.
    pub image: ImageBuffer<Rgba<u8>, Vec<u8>>,
    /// Pinhole camera model with intrinsics and extrinsics.
    #[pyo3(get, set)]
    pub camera_model: PinholeCamera,
    /// Nanosecond timestamp.
    #[pyo3(get, set)]
    pub timestamp_ns: usize,
}

#[pymethods]
impl TimeStampedImage {
    #[getter]
    fn image<'py>(&self, py: Python<'py>) -> &'py PyArray<u8, Ix3> {
        self.image.ref_ndarray3().to_pyarray(py)
    }
}
