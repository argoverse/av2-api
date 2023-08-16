//! # export_augmentation_database
//!
//! Exports cropped annotations from the `train` dataset for augmentation during training.

#[cfg(feature = "blas")]
extern crate blas_src;
#[macro_use]
extern crate log;

use std::{fs, path::PathBuf};

use av2::{
    data_loader::DataLoader,
    geometry::polytope::{compute_interior_points_mask, cuboids_to_polygons},
    io::write_feather_eager,
};
use indicatif::ProgressBar;

use itertools::Itertools;
use ndarray::{s, Array, Axis, Ix2};
use once_cell::sync::Lazy;
use polars::{
    df,
    prelude::{DataFrame, Float32Type, IndexOrder, NamedFrom},
    series::Series,
};
use std::collections::HashMap;

/// Constants can be changed to fit your directory structure.
/// However, it's recommend to place the datasets in the default folders.

/// Root directory to datasets.
static ROOT_DIR: Lazy<PathBuf> = Lazy::new(|| dirs::home_dir().unwrap().join("data/datasets/"));

/// Dataset name.
static DATASET_NAME: &str = "av2";

/// Dataset type. This will either be "lidar" or "sensor".
static DATASET_TYPE: &str = "sensor";

/// Split names for the dataset.
static SPLIT_NAMES: Lazy<Vec<&str>> = Lazy::new(|| vec!["train"]);

/// Number of accumulated sweeps.
const NUM_ACCUMULATED_SWEEPS: usize = 1;

/// Memory maps the sweeps for fast pre-processing. Requires .feather files to be uncompressed.
const MEMORY_MAPPED: bool = false;

static DST_DATASET_NAME: Lazy<String> =
    Lazy::new(|| format!("{DATASET_NAME}_{NUM_ACCUMULATED_SWEEPS}_database"));
static SRC_PREFIX: Lazy<PathBuf> = Lazy::new(|| ROOT_DIR.join(DATASET_NAME).join(DATASET_TYPE));
static DST_PREFIX: Lazy<PathBuf> =
    Lazy::new(|| ROOT_DIR.join(DST_DATASET_NAME.clone()).join(DATASET_TYPE));

static EXPORTED_COLUMN_NAMES: Lazy<Vec<&str>> = Lazy::new(|| {
    vec![
        "x",
        "y",
        "z",
        "intensity",
        "laser_number",
        "offset_ns",
        "timedelta_ns",
    ]
});

/// Script entrypoint.
pub fn main() {
    env_logger::init();
    for split_name in SPLIT_NAMES.clone() {
        let split_path = SRC_PREFIX.join(split_name);
        if !split_path.exists() {
            error!("Cannot find `{split_path:?}`. Skipping ...");
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

        let mut category_counter: HashMap<String, u64> = HashMap::new();
        let bar = ProgressBar::new(data_loader.len() as u64);
        for sweep in data_loader {
            let lidar = &sweep.lidar.0;
            let lidar_ndarray = lidar.to_ndarray::<Float32Type>(IndexOrder::C).unwrap();

            let cuboids = sweep.cuboids.unwrap().0;
            let category = cuboids["category"]
                .utf8()
                .unwrap()
                .into_iter()
                .map(|x| x.unwrap())
                .collect_vec()
                .clone();

            let cuboids = cuboids
                .clone()
                .to_ndarray::<Float32Type>(IndexOrder::C)
                .unwrap();
            let cuboid_vertices = cuboids_to_polygons(&cuboids.view());
            let points = lidar_ndarray.slice(s![.., ..3]);
            let mask = compute_interior_points_mask(&points.view(), &cuboid_vertices.view());
            for (c, m) in category.into_iter().zip(mask.outer_iter()) {
                let indices = m
                    .iter()
                    .enumerate()
                    .filter_map(|(i, x)| match *x {
                        true => Some(i),
                        _ => None,
                    })
                    .collect_vec();

                let points_i = lidar_ndarray.select(Axis(0), &indices);
                let data_frame_i = _build_data_frame(points_i, EXPORTED_COLUMN_NAMES.clone());

                category_counter
                    .entry(c.to_string())
                    .and_modify(|count| *count += 1)
                    .or_insert(0);
                let count = category_counter.get(&c.to_string()).unwrap();

                let dst = DST_PREFIX.join(c).join(format!("{count:08}.feather"));
                fs::create_dir_all(dst.parent().unwrap()).unwrap();
                write_feather_eager(&dst, data_frame_i);
            }
            bar.inc(1);
        }

        let category = category_counter.keys().cloned().collect_vec();
        let count = category_counter.values().cloned().collect_vec();
        let num_padding = category_counter.values().map(|_| 8_u8).collect_vec();
        let index =
            df!("category" => category, "count" => count, "num_padding" => num_padding).unwrap();

        let dst = DST_PREFIX.join("_index.feather");
        write_feather_eager(&dst, index);
    }
}

// Helper method to build exported `DataFrame`.
fn _build_data_frame(arr: Array<f32, Ix2>, column_names: Vec<&str>) -> DataFrame {
    let series_vec = arr
        .columns()
        .into_iter()
        .zip(column_names)
        .map(|(column, column_name)| match column_name {
            "x" => Series::new("x", column.to_owned().into_raw_vec()),
            "y" => Series::new("y", column.to_owned().into_raw_vec()),
            "z" => Series::new("z", column.to_owned().into_raw_vec()),
            "intensity" => Series::new(
                "intensity",
                column.to_owned().mapv(|x| x as u8).into_raw_vec(),
            ),
            "laser_number" => Series::new(
                "laser_number",
                column.to_owned().mapv(|x| x as u8).into_raw_vec(),
            ),
            "offset_ns" => Series::new(
                "offset_ns",
                column.to_owned().mapv(|x| x as u32).into_raw_vec(),
            ),
            "timedelta_ns" => Series::new("timedelta_ns", column.to_owned().into_raw_vec()),
            _ => panic!(),
        })
        .collect_vec();
    DataFrame::from_iter(series_vec)
}
