//! # frame_utils
//!
//! DataFrame utilities.

use polars::prelude::DataFrame;

/// Extract an `f64` from a DataFrame row.
pub fn extract_f64_from_frame(series: &DataFrame, column_name: &str) -> f64 {
    series[column_name]
        .get(0)
        .unwrap_or_else(|_| panic!("{column_name} does not exist."))
        .try_extract::<f64>()
        .unwrap()
}

/// Extract a `usize` from a DataFrame row.
pub fn extract_usize_from_frame(series: &DataFrame, column_name: &str) -> usize {
    series[column_name]
        .get(0)
        .unwrap_or_else(|_| panic!("{column_name} does not exist."))
        .try_extract::<usize>()
        .unwrap()
}
