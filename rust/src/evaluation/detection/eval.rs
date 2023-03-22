use itertools::Itertools;
use polars::prelude::IntoLazy;

use std::{collections::HashMap, path::PathBuf};

use polars::{
    lazy::dsl::cols,
    prelude::{DataFrame, NamedFrom, SortOptions},
    series::Series,
};
use rayon::prelude::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};

use crate::evaluation::detection::constants::{
    MAX_NORMALIZED_ASE, MAX_YAW_RAD_ERROR, MIN_AP, MIN_CDS, ORDERED_CUBOID_COLUMNS, UUID_COLUMNS,
};

use super::{
    constants::{AffinityType, FilterMetricType},
    utils::assign,
};

/// Detection evaluation configuration.
#[derive(Clone)]
pub struct DetectionCfg {
    /// Affinity thresholds in meters.
    pub affinity_thresholds_m: Vec<f32>,
    /// Affinity type.
    pub affinity_type: AffinityType,
    /// Categories to evaluate.
    pub categories: Vec<String>,
    /// Dataset directory.
    pub dataset_dir: Option<PathBuf>,
    /// Evaluate only ROI instances.
    pub eval_only_roi_instances: bool,
    /// Filter metric type.
    pub filter_metric: FilterMetricType,
    /// Maximum number of detections per category.
    pub max_num_dts_per_category: usize,
    /// Maximum range in meters.
    pub max_range_m: f32,
    /// Number of recall samples.
    pub num_recall_samples: usize,
    /// True positive threshold in meters.
    pub tp_threshold_m: f32,
}

impl Default for DetectionCfg {
    fn default() -> Self {
        Self {
            affinity_thresholds_m: vec![0.5, 1.0, 2.0, 4.0],
            affinity_type: AffinityType::CENTER,
            categories: Default::default(),
            dataset_dir: None,
            eval_only_roi_instances: true,
            filter_metric: FilterMetricType::EUCLIDEAN,
            max_num_dts_per_category: 100,
            max_range_m: 150.0,
            num_recall_samples: 100,
            tp_threshold_m: 2.0,
        }
    }
}

impl DetectionCfg {
    /// Return the evaluation summary default values.
    pub fn metrics_defaults(self) -> Vec<f32> {
        vec![
            MIN_AP,
            self.tp_threshold_m,
            MAX_NORMALIZED_ASE,
            MAX_YAW_RAD_ERROR,
            MIN_CDS,
        ]
    }
}

/// Partition data-frame based on the provided columns.
pub fn partition_frame(
    frame: &DataFrame,
    columns: Vec<&str>,
) -> HashMap<(String, usize), DataFrame> {
    frame
        .partition_by_stable(columns)
        .unwrap()
        .par_iter()
        .map(|frame| {
            let uuid_frame = frame.select(["log_id", "timestamp_ns"]).unwrap();
            let row = uuid_frame.get_row(0).unwrap().0;
            let uuid = (
                row.get(0).unwrap().get_str().unwrap().to_string(),
                row.get(1).unwrap().try_extract::<u64>().unwrap() as usize,
            );
            (uuid, frame.to_owned())
        })
        .collect()
}

/// Accumulate the true / false positives (boolean flags) and true positive errors for each class.
pub fn accumulate(
    mut dts: DataFrame,
    mut gts: DataFrame,
    cfg: DetectionCfg,
) -> (DataFrame, DataFrame) {
    let n = dts.shape().0;
    let m = gts.shape().0;

    // Sort the detections by score in _descending_ order.
    let mut scores = &dts["score"];
    let permutation = scores.argsort(SortOptions {
        descending: true,
        nulls_last: true,
        multithreaded: true,
    });
    dts = dts.take(&permutation).unwrap();

    let is_evaluated_dts = Series::new("is_evaluated", vec![true; n]);
    let is_evaluated_gts = Series::new("is_evaluated", vec![true; m]);
    if dts.shape().0 > 0 {
        assign(&dts, &gts, cfg);
    }
    (dts.clone(), gts.clone())
}

/// Evaluate a set of detections against the ground truth annotations.
pub fn evaluate(mut dts: DataFrame, mut gts: DataFrame, cfg: DetectionCfg) {
    // Sort both the detections and annotations by lexicographic order for grouping.
    dts = dts
        .lazy()
        .sort_by_exprs(&[cols(UUID_COLUMNS)], &[false], false)
        .collect()
        .unwrap();
    gts = gts
        .lazy()
        .sort_by_exprs(&[cols(UUID_COLUMNS)], &[false], false)
        .collect()
        .unwrap();

    let uuid_to_dts = partition_frame(&dts, UUID_COLUMNS.to_vec());
    let uuid_to_gts = partition_frame(&gts, UUID_COLUMNS.to_vec());
    let outputs: Vec<_> = uuid_to_gts
        .into_par_iter()
        .map(|(uuid, sweep_gts)| {
            let sweep_dts = if !uuid_to_dts.contains_key(&uuid) {
                uuid_to_dts.get(&uuid).unwrap().to_owned()
            } else {
                let columns = ORDERED_CUBOID_COLUMNS
                    .into_iter()
                    .map(|x| Series::new(x, Vec::<f32>::new()))
                    .collect_vec();
                DataFrame::new(columns).unwrap()
            };
            accumulate(sweep_dts, sweep_gts, cfg.clone())
        })
        .collect();

    println!("Evaluation complete.");
}

#[cfg(test)]
mod tests {
    use std::{path::PathBuf, str::FromStr};

    use once_cell::sync::Lazy;

    use crate::{evaluation::detection::eval::DetectionCfg, io::read_feather};

    use super::evaluate;

    static TEST_DATA_DIR: Lazy<PathBuf> = Lazy::new(|| {
        PathBuf::from_str("/Users/owner/code/av2-api/tests/evaluation/detection/data").unwrap()
    });

    #[test]
    fn test_evaluate() {
        let dts_path = TEST_DATA_DIR.join("detections_identity.feather");
        let dts = read_feather(&dts_path, false);
        let gts = dts.clone();
        let cfg = DetectionCfg::default();

        println!("{}", dts);
        // panic!();

        evaluate(dts, gts, cfg);
    }
}
