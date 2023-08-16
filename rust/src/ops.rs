//! # ops
//!
//! Optimized operations for data processing.

use itertools::Itertools;
use ndarray::{azip, par_azip, s, Array1, Array2, ArrayView2, Axis};
use std::{
    collections::HashMap,
    ops::{AddAssign, DivAssign},
};

/// Convert unraveled coordinates (i.e., multi-dimensional indices) to a linear index.
pub fn ravel_multi_index(unraveled_coords: &ArrayView2<usize>, size: Vec<usize>) -> Array2<usize> {
    let shape_arr = Array1::<usize>::from_vec(size.into_iter().chain(vec![1]).collect::<Vec<_>>());

    let mut coefs = shape_arr
        .slice(s![1..])
        .into_iter()
        .rev()
        .copied()
        .collect::<Array1<usize>>();

    coefs.accumulate_axis_inplace(Axis(0), |prev, curr| *curr *= prev);
    coefs = coefs.iter().rev().copied().collect::<Array1<usize>>();
    (unraveled_coords * &coefs)
        .sum_axis(Axis(1))
        .insert_axis(Axis(1))
}

/// Cluster a group of features into a set of voxels based on their indices.
pub fn voxelize(
    indices: &ArrayView2<usize>,
    features: &ArrayView2<f32>,
    length: usize,
    width: usize,
    height: usize,
) -> (Array2<usize>, Array2<f32>, Array2<f32>) {
    let shape = vec![length, width, height];
    let raveled_indices = ravel_multi_index(&indices.view(), shape);

    let num_features = features.shape()[1];

    let unique_indices = raveled_indices
        .clone()
        .into_raw_vec()
        .into_iter()
        .unique()
        .enumerate()
        .map(|(k, v)| (v, k))
        .collect::<HashMap<_, _>>();

    let mut indices_buffer = Array2::<usize>::zeros([unique_indices.len(), 3]);
    let mut values_buffer = Array2::<f32>::zeros([unique_indices.len(), num_features]);
    let mut counts = Array2::<f32>::zeros([unique_indices.len(), 1]);

    azip!((idx in raveled_indices.rows(), index in indices.rows(), val in features.rows()) {
        let i = *unique_indices.get(&idx[0]).unwrap();

        indices_buffer.slice_mut(s![i, ..]).assign(&index);
        values_buffer.slice_mut(s![i, ..]).add_assign(&val);
        counts[[i, 0]] += 1.;
    });

    par_azip!((mut val in values_buffer.rows_mut(), count in counts.rows()) {
        val.div_assign(&count);
    });
    (indices_buffer, values_buffer, counts)
}
