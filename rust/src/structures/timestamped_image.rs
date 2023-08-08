use image::{ImageBuffer, Rgba};
use ndarray::Ix3;
use nshare::ToNdarray3;
use numpy::{IntoPyArray, PyArray};
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use crate::geometry::camera::pinhole_camera::PinholeCamera;

/// Image modeled by `camera_model` occuring at `timestamp_ns`.
#[pyclass]
#[derive(Clone)]
pub struct TimeStampedImage {
    /// RGBA u8 image buffer.
    // #[pyo3(get, set)]
    pub image: ImageBuffer<Rgba<u8>, Vec<u8>>,
    /// Pinhole camera model with intrinsics and extrinsics.
    #[pyo3(get, set)]
    pub camera_model: PinholeCamera,
    /// Nanosecond timestamp.
    #[pyo3(get, set)]
    pub timestamp_ns: usize,
    /// Ego-vehicle city pose.
    #[pyo3(get, set)]
    pub city_pose: Option<PyDataFrame>,
}

#[pymethods]
impl TimeStampedImage {
    /// Image buffer (python).
    #[getter(image)]
    pub fn py_image<'py>(&self, py: Python<'py>) -> &'py PyArray<u8, Ix3> {
        self.image.clone().into_ndarray3().into_pyarray(py)
    }
}
