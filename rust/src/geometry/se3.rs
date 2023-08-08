//! # SE(3)
//!
//! Special Euclidean Group 3.

use ndarray::{s, Array, Array1, Array2, Ix1};
use ndarray::{ArrayView2, Ix2};
use numpy::{IntoPyArray, PyArray};
use pyo3::prelude::*;
use pyo3::types::PyType;
use pyo3_polars::PyDataFrame;

use crate::frame_utils::extract_f64_from_frame;

use super::so3::_quat_to_mat3;

/// Special Euclidean Group 3.
/// Rigid transformation parameterized by a rotation and translation in $R^3$.
#[pyclass]
#[derive(Clone, Debug)]
pub struct SE3 {
    /// (3,3) Orthonormal rotation matrix.
    pub rotation: Array2<f64>,
    /// (3,) Translation vector.
    pub translation: Array1<f64>,
}

impl SE3 {
    /// Get the (4,4) homogeneous transformation matrix associated with the rigid transformation.
    pub fn transform_matrix(&self) -> Array2<f64> {
        let mut transform_matrix = Array2::eye(4);
        transform_matrix
            .slice_mut(s![..3, ..3])
            .assign(&self.rotation);
        transform_matrix
            .slice_mut(s![..3, 3])
            .assign(&self.translation);
        transform_matrix
    }

    /// Transform the point cloud from its reference from to the SE(3) destination.
    pub fn transform_from(&self, point_cloud: &ArrayView2<f64>) -> Array2<f64> {
        point_cloud.dot(&self.rotation.t()) + &self.translation
    }

    /// Invert the SE(3) transformation.
    pub fn inverse(&self) -> SE3 {
        let rotation = self.rotation.t().as_standard_layout().to_owned();
        let translation = rotation.dot(&(-&self.translation));
        Self {
            rotation,
            translation,
        }
    }

    /// Compose (right multiply) an SE(3) with another SE(3).
    pub fn compose(&self, right_se3: &SE3) -> SE3 {
        let chained_transform_matrix = self.transform_matrix().dot(&right_se3.transform_matrix());
        SE3 {
            rotation: chained_transform_matrix
                .slice(s![..3, ..3])
                .as_standard_layout()
                .to_owned(),
            translation: chained_transform_matrix
                .slice(s![..3, 3])
                .as_standard_layout()
                .to_owned(),
        }
    }
}

#[pymethods]
impl SE3 {
    /// SE(3) rotation component (python).
    #[getter(rotation)]
    pub fn py_rotation<'py>(&self, py: Python<'py>) -> &'py PyArray<f64, Ix2> {
        self.rotation.clone().into_pyarray(py)
    }

    /// SE(3) translation component (python).
    #[getter(translation)]
    pub fn py_translation<'py>(&self, py: Python<'py>) -> &'py PyArray<f64, Ix1> {
        self.translation.clone().into_pyarray(py)
    }

    /// SE(3) transform matrix component (python).
    #[getter(transform_matrix)]
    pub fn py_transform_matrix<'py>(&self, py: Python<'py>) -> &'py PyArray<f64, Ix2> {
        self.transform_matrix().into_pyarray(py)
    }

    /// Construct SE(3) from a DataFrame row.
    #[classmethod]
    #[pyo3(name = "from_dataframe")]
    pub fn py_from_dataframe<'py>(cls: &PyType, py_extrinsics: PyDataFrame) -> Self {
        let extrinsics = py_extrinsics.0;
        let qw = extract_f64_from_frame(&extrinsics, "qw");
        let qx = extract_f64_from_frame(&extrinsics, "qx");
        let qy = extract_f64_from_frame(&extrinsics, "qy");
        let qz = extract_f64_from_frame(&extrinsics, "qz");
        let tx_m = extract_f64_from_frame(&extrinsics, "tx_m");
        let ty_m = extract_f64_from_frame(&extrinsics, "ty_m");
        let tz_m = extract_f64_from_frame(&extrinsics, "tz_m");

        let quat_wxyz = Array::<f64, Ix1>::from_vec(vec![qw, qx, qy, qz]);
        let rotation = _quat_to_mat3(&quat_wxyz.view());
        let translation = Array::<f64, Ix1>::from_vec(vec![tx_m, ty_m, tz_m]);
        Self {
            rotation,
            translation,
        }
    }
}
