//! # build_accumulated_sweeps
//!
//! Accumulates lidar sweeps across a contiguous temporal window.
//! Defaults to a 5 frame window (~1 second of lidar data).

use std::{fs, path::PathBuf};

use av2::{data_loader::DataLoader, io::write_feather_eager};
use indicatif::ProgressBar;

#[macro_use]
extern crate log;
use once_cell::sync::Lazy;

/// Constants can be changed to fit your directory structure.
/// However, it's recommend to place the datasets in the default folders.

/// Root directory to datasets.
static ROOT_DIR: Lazy<PathBuf> = Lazy::new(|| dirs::home_dir().unwrap().join("data/datasets/"));

/// Dataset name.
static DATASET_NAME: &str = "av2";

/// Dataset type. This will either be "lidar" or "sensor".
static DATASET_TYPE: &str = "sensor";

/// Split names for the dataset.
static SPLIT_NAMES: Lazy<Vec<&str>> = Lazy::new(|| vec!["train", "val", "test"]);

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

        let mut previous_log_id = String::new();
        for sweep in data_loader {
            let lidar = &sweep.lidar.0;
            let (log_id, timestamp_ns) = &sweep.sweep_uuid;

            let log_suffix = format!("{split_name}/{log_id}");
            let suffix = format!("{log_suffix}/sensors/lidar/{timestamp_ns}.feather");
            let dst = DST_PREFIX.clone().join(suffix);
            fs::create_dir_all(dst.parent().unwrap()).unwrap();
            write_feather_eager(&dst, lidar.to_owned());

            let log_id = sweep.sweep_uuid.0;
            if log_id != previous_log_id {
                let city_se3_egovehicle_src = SRC_PREFIX
                    .clone()
                    .join(log_suffix.clone())
                    .join("city_SE3_egovehicle.feather");
                let city_se3_egovehicle_dst = DST_PREFIX
                    .clone()
                    .join(log_suffix.clone())
                    .join("city_SE3_egovehicle.feather");

                fs::copy(city_se3_egovehicle_src, city_se3_egovehicle_dst).unwrap();

                if split_name != "test" {
                    let annotations_src = SRC_PREFIX
                        .clone()
                        .join(log_suffix.clone())
                        .join("annotations.feather");
                    let annotations_dst = DST_PREFIX
                        .clone()
                        .join(log_suffix.clone())
                        .join("annotations.feather");

                    fs::copy(annotations_src, annotations_dst).unwrap();
                }
            }
            previous_log_id = log_id.to_string();
            bar.inc(1)
        }
    }
}
