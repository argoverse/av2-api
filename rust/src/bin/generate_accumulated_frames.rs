use std::{fs, path::PathBuf};

use av2::{data_loader::DataLoader, io::write_feather_eager};
use indicatif::ProgressBar;
use once_cell::sync::Lazy;

static ROOT_DIR: Lazy<PathBuf> = Lazy::new(|| dirs::home_dir().unwrap().join("data/datasets/"));
static DATASET_NAME: &str = "av2";
static DATASET_TYPE: &str = "sensor";
static SPLIT_NAME: &str = "val";

const NUM_ACCUMULATED_SWEEPS: usize = 5;
const MEMORY_MAPPED: bool = false;

static DST_DATASET_NAME: Lazy<String> =
    Lazy::new(|| format!("{DATASET_NAME}_{NUM_ACCUMULATED_SWEEPS}_sweep"));
static DST_DIR: Lazy<PathBuf> = Lazy::new(|| {
    ROOT_DIR
        .join(DST_DATASET_NAME.clone())
        .join(DATASET_TYPE)
        .join(SPLIT_NAME)
});

pub fn main() {
    let data_loader = DataLoader::new(
        ROOT_DIR.clone().to_str().unwrap(),
        DATASET_NAME,
        DATASET_TYPE,
        SPLIT_NAME,
        NUM_ACCUMULATED_SWEEPS,
        MEMORY_MAPPED,
    );

    let bar = ProgressBar::new(data_loader.len() as u64);
    for datum in data_loader {
        let lidar = datum.lidar.0;
        println!("{}", lidar);
        let (log_id, timestamp_ns) = datum.sweep_uuid;
        let dst = DST_DIR
            .clone()
            .join(format!("{log_id}/sensors/lidar/{timestamp_ns}.feather"));

        fs::create_dir_all(dst.parent().unwrap()).unwrap();
        write_feather_eager(&dst, lidar);
        bar.inc(1)
    }
}
