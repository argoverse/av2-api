//! # av2
//!
//! Argoverse 2 Rust library.

#![warn(missing_docs)]
#![warn(missing_doc_code_examples)]

#[cfg(feature = "blas")]
extern crate blas_src;

pub mod constants;
pub mod data_loader;
pub mod geometry;
pub mod io;
pub mod ops;
pub mod path;
pub mod share;
pub mod structures;

use data_loader::{DataLoader, Sweep};
use ndarray::{Dim, Ix1, Ix2};
use numpy::PyReadonlyArray;
use numpy::{IntoPyArray, PyArray};
use pyo3::prelude::*;

use geometry::so3::{_quat_to_mat3, quat_to_yaw, yaw_to_quat};
use numpy::PyReadonlyArray2;

use crate::ops::voxelize;

#[pyfunction]
#[pyo3(name = "voxelize")]
#[allow(clippy::type_complexity)]
fn py_voxelize<'py>(
    py: Python<'py>,
    indices: PyReadonlyArray2<usize>,
    features: PyReadonlyArray2<f32>,
    length: usize,
    width: usize,
    height: usize,
) -> (
    &'py PyArray<usize, Dim<[usize; 2]>>,
    &'py PyArray<f32, Dim<[usize; 2]>>,
    &'py PyArray<f32, Dim<[usize; 2]>>,
) {
    let (indices, values, counts) = voxelize(
        &indices.as_array(),
        &features.as_array(),
        length,
        width,
        height,
    );
    (
        indices.into_pyarray(py),
        values.into_pyarray(py),
        counts.into_pyarray(py),
    )
}

#[pyfunction]
#[pyo3(name = "quat_to_mat3")]
#[allow(clippy::type_complexity)]
fn py_quat_to_mat3<'py>(
    py: Python<'py>,
    quat_wxyz: PyReadonlyArray<f32, Ix1>,
) -> &'py PyArray<f32, Ix2> {
    _quat_to_mat3(&quat_wxyz.as_array().view()).into_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "quat_to_yaw")]
#[allow(clippy::type_complexity)]
fn py_quat_to_yaw<'py>(
    py: Python<'py>,
    quat_wxyz: PyReadonlyArray<f32, Ix2>,
) -> &'py PyArray<f32, Ix2> {
    quat_to_yaw(&quat_wxyz.as_array().view()).into_pyarray(py)
}

#[pyfunction]
#[pyo3(name = "yaw_to_quat")]
#[allow(clippy::type_complexity)]
fn py_yaw_to_quat<'py>(
    py: Python<'py>,
    quat_wxyz: PyReadonlyArray<f32, Ix2>,
) -> &'py PyArray<f32, Ix2> {
    yaw_to_quat(&quat_wxyz.as_array().view()).into_pyarray(py)
}

/// A Python module implemented in Rust.
#[pymodule]
fn _r(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<DataLoader>()?;
    m.add_class::<Sweep>()?;
    m.add_function(wrap_pyfunction!(py_quat_to_mat3, m)?)?;
    m.add_function(wrap_pyfunction!(py_quat_to_yaw, m)?)?;
    m.add_function(wrap_pyfunction!(py_voxelize, m)?)?;
    m.add_function(wrap_pyfunction!(py_yaw_to_quat, m)?)?;
    Ok(())
}
