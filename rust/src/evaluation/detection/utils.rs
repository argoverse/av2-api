use crate::evaluation::detection::eval::DetectionCfg;
use argminmax::ArgMinMax;
use itertools::Itertools;
use ndarray::{par_azip, Array2, ArrayView2, Axis};
use polars::prelude::*;

use super::constants::AffinityType;

/// Compute pairwise Euclidean distance between two sets of points.
pub fn cdist(x1: &ArrayView2<f32>, x2: &ArrayView2<f32>) -> Array2<f32> {
    let n = x1.shape()[0];
    let m = x2.shape()[0];
    let mut dists = Array2::<f32>::zeros([n, m]);
    par_azip!((mut d_i in dists.rows_mut(), x_i in x1.rows()) {
        par_azip!((mut d_ij in d_i, x_j in x2.rows()) {
            *d_ij = x_i.dot(&x_j);
        });
    });
    dists
}

/// Calculate the affinity matrix between detections and ground truth annotations.
pub fn compute_affinity_matrix(
    dts: &ArrayView2<f32>,
    gts: &ArrayView2<f32>,
    metric: AffinityType,
) -> Array2<f32> {
    match metric {
        AffinityType::CENTER => {
            let dts_xy_m = dts;
            let gts_xy_m = gts;
            -cdist(&dts_xy_m, &gts_xy_m)
        }
    }
}

/// Attempt assignment of each detection to a ground truth label.
///
/// The detections (gts) and ground truth annotations (gts) are expected to be shape (N,10) and (M,10)
/// respectively. Their _ordered_ columns are shown below:
/// dts: tx_m, ty_m, tz_m, length_m, width_m, height_m, qw, qx, qy, qz.
/// gts: tx_m, ty_m, tz_m, length_m, width_m, height_m, qw, qx, qy, qz.
///
/// NOTE: The columns for dts and gts only differ by their _last_ column. Score represents the
///     "confidence" of the detection and `num_interior_pts` are the number of points interior
///     to the ground truth cuboid at the time of annotation.
pub fn assign(dts: &DataFrame, gts: &DataFrame, cfg: DetectionCfg) {
    let dts_nd = dts.to_ndarray::<Float32Type>().unwrap();
    let gts_nd = gts.to_ndarray::<Float32Type>().unwrap();

    let affinity_type = cfg.affinity_type;
    let affinity_matrix = compute_affinity_matrix(&dts_nd.view(), &gts_nd.view(), affinity_type);

    // Get the GT label for each max-affinity GT label, detection pair.
    let idx_gts = affinity_matrix
        .columns()
        .into_iter()
        .map(|col| {
            let (_, max_index) = col.argminmax();
            max_index
        })
        .collect_vec();

    // The affinity matrix is an N by M matrix of the detections and ground truth labels respectively.
    // We want to take the corresponding affinity for each of the initial assignments using `gt_matches`.
    // The following line grabs the max affinity for each detection to a ground truth label.
    let affinities = affinity_matrix.t().select(Axis(0), &idx_gts);

    // Find the indices of the _first_ detection assigned to each GT.
    let (idx_gts, idx_dts): (Vec<_>, Vec<_>) = idx_gts
        .into_iter()
        .enumerate()
        .unique_by(|(i, x)| *x)
        .unzip();

    let num_dts = dts.shape().0;
    let num_gts = gts.shape().0;
    let affinity_thresholds_m = cfg.clone().affinity_thresholds_m;
    let metrics_defaults = cfg.clone().metrics_defaults();
    let mut dts_metrics = DataFrame::new(
        affinity_thresholds_m
            .clone()
            .into_iter()
            .zip(metrics_defaults.get(1..4).unwrap())
            .map(|(column, metric_default)| {
                Series::new(&column.to_string(), vec![*metric_default; num_dts])
            })
            .collect(),
    )
    .unwrap();

    let mut gts_metrics = DataFrame::new(
        affinity_thresholds_m
            .clone()
            .into_iter()
            .zip(metrics_defaults.get(1..4).unwrap())
            .map(|(column, metric_default)| {
                Series::new(&column.to_string(), vec![*metric_default; num_gts])
            })
            .collect(),
    )
    .unwrap();

    for (i, threshold_m) in affinity_thresholds_m.into_iter().enumerate() {
        let is_tp = affinities
            .select(Axis(0), &idx_dts)
            .mapv(|x| x > -threshold_m);

        let tp_indices = idx_dts
            .clone()
            .into_iter()
            .zip(is_tp)
            .filter_map(|(idx, flag)| match flag {
                true => Some(idx),
                _ => None,
            })
            .collect_vec();
    }
}
