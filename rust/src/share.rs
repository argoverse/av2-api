//! # share
//!
//! Conversion methods between different libraries.

use ndarray::{Array, Ix2};
use polars::{
    lazy::dsl::{cols, lit, Expr},
    prelude::{DataFrame, Float32Type, IndexOrder, IntoLazy, NamedFrom},
    series::Series,
};

/// Convert the columns of an `ndarray::Array` into a vector of `polars` expressions.
pub fn ndarray_to_expr_vec(arr: Array<f32, Ix2>, column_names: Vec<&str>) -> Vec<Expr> {
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
        series_vec.push(lit(series));
    }
    series_vec
}

/// Convert a data frame to an `ndarray::Array::<f32, Ix2>`.
pub fn data_frame_to_ndarray_f32(
    data_frame: DataFrame,
    column_names: Vec<&str>,
) -> Array<f32, Ix2> {
    data_frame
        .lazy()
        .select(&[cols(column_names)])
        .collect()
        .unwrap()
        .to_ndarray::<Float32Type>(IndexOrder::C)
        .unwrap()
}
