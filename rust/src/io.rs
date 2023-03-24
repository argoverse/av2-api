//! # io
//!
//! Reading and writing operations.

use ndarray::s;
use ndarray::Array2;

use polars::lazy::dsl::lit;
use polars::lazy::dsl::Expr;

use polars::prelude::*;

use polars::series::Series;
use polars::{
    self,
    lazy::dsl::{col, cols},
    prelude::{DataFrame, IntoLazy},
};
use rayon::prelude::IntoParallelIterator;
use rayon::prelude::ParallelIterator;
use std::fs::File;
use std::path::PathBuf;

use crate::constants::POSE_COLUMNS;
use crate::se3::SE3;
use crate::so3::quat_to_mat3;

/// Read a feather file and load into a `polars` dataframe.
pub fn read_feather_eager(path: &PathBuf, memory_mapped: bool) -> DataFrame {
    let file =
        File::open(path).unwrap_or_else(|_| panic!("{path} not found.", path = path.display()));
    polars::io::ipc::IpcReader::new(file)
        .memory_mapped(memory_mapped)
        .finish()
        .unwrap_or_else(|_| panic!("The IPC file, {:?}, is malformed.", path))
}

/// Read a feather file and load into a `polars` dataframe.
/// TODO: Implement once upstream half-type is fixed.
// pub fn read_feather_lazy(path: &PathBuf, memory_mapped: bool) -> DataFrame {
//     LazyFrame::scan_ipc(
//         path,
//         ScanArgsIpc {
//             n_rows: None,
//             cache: true,
//             rechunk: true,
//             row_count: None,
//             memmap: memory_mapped,
//         },
//     )
//     .unwrap()
//     .collect()
//     .unwrap()
// }

/// Read and accumulate lidar sweeps.
/// Accumulation will only occur if `num_accumulated_sweeps` > 1.
/// Sweeps are motion-compensated to the most recent sweep (i.e., at `timestamp_ns`).
pub fn read_accumulate_lidar(
    log_dir: PathBuf,
    file_index: &DataFrame,
    log_id: &str,
    timestamp_ns: u64,
    idx: usize,
    num_accumulated_sweeps: usize,
    memory_mapped: bool,
) -> LazyFrame {
    let start_idx = i64::max(idx as i64 - num_accumulated_sweeps as i64 + 1, 0) as usize;
    let log_ids = file_index["log_id"].utf8().unwrap();
    let timestamps = file_index["timestamp_ns"].u64().unwrap();
    let poses_path = log_dir.join("city_SE3_egovehicle.feather");
    let poses = read_feather_eager(&poses_path, memory_mapped);

    let pose_ref = ndarray_filtered_from_frame(
        &poses,
        cols(POSE_COLUMNS),
        col("timestamp_ns").eq(timestamp_ns),
    );

    let translation = pose_ref.slice(s![0, ..3]).as_standard_layout().to_owned();
    let quat_wxyz = pose_ref.slice(s![0, 3..]).as_standard_layout().to_owned();
    let rotation = quat_to_mat3(&quat_wxyz.view());
    let city_se3_ego = SE3 {
        rotation,
        translation,
    };
    let ego_se3_city = city_se3_ego.inverse();
    let indices: Vec<_> = (start_idx..=idx).collect();
    let mut lidar_list = indices
        .into_par_iter()
        .filter_map(|i| {
            let log_id_i = log_ids.get(i).unwrap();
            match log_id_i == log_id {
                true => Some(i),
                _ => None,
            }
        })
        .map(|i| {
            let timestamp_ns_i = timestamps.get(i).unwrap();
            let lidar_path = build_lidar_file_path(log_dir.clone(), timestamp_ns_i);
            let mut lidar = read_feather_eager(&lidar_path, memory_mapped).lazy();

            let xyz = ndarray_from_frame(&lidar.clone().collect().unwrap(), cols(["x", "y", "z"]));
            let timedeltas = Series::new(
                "timedelta_ns",
                vec![(timestamp_ns - timestamp_ns_i) as f32 * 1e-9; xyz.shape()[0]],
            );
            if timestamp_ns_i != timestamp_ns {
                let pose_i = ndarray_filtered_from_frame(
                    &poses,
                    cols(POSE_COLUMNS),
                    col("timestamp_ns").eq(timestamp_ns_i),
                );

                let translation_i = pose_i.slice(s![0, ..3]).as_standard_layout().to_owned();
                let quat_wxyz = pose_i.slice(s![0, 3..]).as_standard_layout().to_owned();
                let rotation_i = quat_to_mat3(&quat_wxyz.view());
                let city_se3_ego_i = SE3 {
                    rotation: rotation_i,
                    translation: translation_i,
                };
                let ego_ref_se3_ego_i = ego_se3_city.compose(&city_se3_ego_i);
                let xyz_ref = ego_ref_se3_ego_i.transform_from(&xyz.view());
                let x_ref = Series::new(
                    "x",
                    xyz_ref
                        .slice(s![.., 0])
                        .as_standard_layout()
                        .to_owned()
                        .into_raw_vec(),
                );
                let y_ref = Series::new(
                    "y",
                    xyz_ref
                        .slice(s![.., 1])
                        .as_standard_layout()
                        .to_owned()
                        .into_raw_vec(),
                );
                let z_ref = Series::new(
                    "z",
                    xyz_ref
                        .slice(s![.., 2])
                        .as_standard_layout()
                        .to_owned()
                        .into_raw_vec(),
                );

                lidar = lidar.with_columns(vec![lit(x_ref), lit(y_ref), lit(z_ref)]);
            }
            lidar = lidar.with_column(lit(timedeltas));
            lidar
        })
        .collect::<Vec<_>>();

    lidar_list.reverse();
    concat(lidar_list, true, true).unwrap()
}

/// Read a dataframe, but filter for the specified timestamp.
pub fn read_timestamped_feather(
    path: &PathBuf,
    columns: &Vec<&str>,
    timestamp_ns: &u64,
    memory_mapped: bool,
) -> LazyFrame {
    read_feather_eager(path, memory_mapped)
        .lazy()
        .filter(col("timestamp_ns").eq(*timestamp_ns))
        .select(&[cols(columns)])
}

/// Build the lidar file path.
pub fn build_lidar_file_path(log_dir: PathBuf, timestamp_ns: u64) -> PathBuf {
    let file_name = format!("{timestamp_ns}.feather");
    let lidar_path = [
        log_dir,
        "sensors".to_string().into(),
        "lidar".to_string().into(),
        file_name.into(),
    ]
    .iter()
    .collect();
    lidar_path
}

/// Convert a dataframe to `ndarray`.
pub fn ndarray_from_frame(frame: &DataFrame, exprs: Expr) -> Array2<f32> {
    frame
        .clone()
        .lazy()
        .select(&[exprs])
        .collect()
        .unwrap()
        .to_ndarray::<Float32Type>()
        .unwrap()
        .as_standard_layout()
        .to_owned()
}

/// Convert a dataframe to `ndarray` and filter.
pub fn ndarray_filtered_from_frame(
    frame: &DataFrame,
    select_exprs: Expr,
    filter_exprs: Expr,
) -> Array2<f32> {
    frame
        .clone()
        .lazy()
        .filter(filter_exprs)
        .select(&[select_exprs])
        .collect()
        .unwrap()
        .to_ndarray::<Float32Type>()
        .unwrap()
        .as_standard_layout()
        .to_owned()
}
