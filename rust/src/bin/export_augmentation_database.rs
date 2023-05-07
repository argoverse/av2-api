//! # build_accumulated_sweeps
//!
//! Accumulates lidar sweeps across a contiguous temporal window.
//! Defaults to a 5 frame window (~1 second of lidar data).

use std::{fs, path::PathBuf};

use av2::{
    data_loader::DataLoader,
    io::{ndarray_filtered_from_frame, write_feather_eager},
};
use indicatif::ProgressBar;

#[macro_use]
extern crate log;
use once_cell::sync::Lazy;
use polars::{
    lazy::dsl::{col, lit},
    prelude::{Float32Type, IntoLazy},
};

/// Constants can be changed to fit your directory structure.
/// However, it's recommend to place the datasets in the default folders.

/// Root directory to datasets.
static ROOT_DIR: Lazy<PathBuf> = Lazy::new(|| dirs::home_dir().unwrap().join("data/datasets/"));

/// Dataset name.
static DATASET_NAME: &str = "av2";

/// Dataset type. This will either be "lidar" or "sensor".
static DATASET_TYPE: &str = "sensor";

/// Split names for the dataset.
static SPLIT_NAMES: Lazy<Vec<&str>> = Lazy::new(|| vec!["train", "val"]);

/// Number of accumulated sweeps.
const NUM_ACCUMULATED_SWEEPS: usize = 5;

/// Memory maps the sweeps for fast pre-processing. Requires .feather files to be uncompressed.
const MEMORY_MAPPED: bool = false;

static DST_DATASET_NAME: Lazy<String> =
    Lazy::new(|| format!("{DATASET_NAME}_{NUM_ACCUMULATED_SWEEPS}_sweep"));
static SRC_PREFIX: Lazy<PathBuf> = Lazy::new(|| ROOT_DIR.join(DATASET_NAME).join(DATASET_TYPE));
static DST_PREFIX: Lazy<PathBuf> =
    Lazy::new(|| ROOT_DIR.join(DST_DATASET_NAME.clone()).join(DATASET_TYPE));

/// Script entrypoint.
pub fn main() {
    env_logger::init();
    for split_name in SPLIT_NAMES.clone() {
        if !SRC_PREFIX.join(split_name).exists() {
            error!("Cannot find `{split_name}` split. Skipping ...");
            continue;
        }
        let data_loader = DataLoader::new(
            ROOT_DIR.clone().to_str().unwrap(),
            DATASET_NAME,
            DATASET_TYPE,
            split_name,
            NUM_ACCUMULATED_SWEEPS,
            MEMORY_MAPPED,
        );
        let bar = ProgressBar::new(data_loader.len() as u64);
        for sweep in data_loader {
            let (log_id, timestamp_ns) = sweep.sweep_uuid;
            let lidar = sweep.lidar.0;
            let cuboids = sweep
                .cuboids
                .unwrap()
                .0
                .to_ndarray::<Float32Type>()
                .unwrap();

            compute_inte
            // let cuboids = ndarray_filtered_from_frame(
            //     &sweep.cuboids.unwrap().0,
            //     col("*"),
            //     col("timestamp_ns").eq(lit(timestamp_ns)),
            // );
            // let cuboids = sweep
            //     .cuboids
            //     .unwrap()
            //     .0
            //     .clone()
            //     .lazy()
            //     .filter(col("timestamp_ns").eq(lit(timestamp_ns)))
            //     .collect()
            //     .to_;

            bar.inc(1)
        }
    }
}
