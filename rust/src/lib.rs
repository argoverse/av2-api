//! # av2
//!
//! Argoverse 2 Rust library.

#![warn(missing_docs)]
#![warn(missing_doc_code_examples)]

pub mod constants;
pub mod dataloader;
pub mod io;
pub mod ops;
pub mod path;
pub mod se3;
pub mod so3;

use dataloader::{Dataloader, Sweep};
use ndarray::Dim;
use numpy::{IntoPyArray, PyArray};
use pyo3::prelude::*;

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

/// A Python module implemented in Rust.
#[pymodule]
fn _r(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Dataloader>()?;
    m.add_class::<Sweep>()?;
    m.add_function(wrap_pyfunction!(py_voxelize, m)?)?;
    Ok(())
}
