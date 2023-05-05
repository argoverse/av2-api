//! # share
//!
//! Conversion methods between different libraries.

use ndarray::{Array, Ix2};
use polars::{prelude::NamedFrom, series::Series};

/// Convert the columns of an `ndarray::Array` into a vector of `polars::series::Series`.
pub fn ndarray_to_series_vec(arr: Array<f32, Ix2>, column_names: Vec<&str>) -> Vec<Series> {
    let num_dims = arr.shape()[1];
    if num_dims != column_names.len() {
        panic!("Number of array columns and column names must match.");
    }

    let mut series_vec = vec![];
    for (column, column_name) in arr.columns().into_iter().zip(column_names) {
        let series = Series::new(
            column_name,
            column.as_standard_layout().to_owned().into_raw_vec(),
        );
        series_vec.push(series);
    }
    series_vec
}
