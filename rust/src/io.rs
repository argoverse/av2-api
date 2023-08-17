//! # io
//!
//! Reading and writing operations.

use image::ImageBuffer;
use image::Rgba;
use ndarray::s;
use ndarray::Array;
use ndarray::Array2;
use ndarray::Array3;
use ndarray::Ix1;

use nshare::ToNdarray3;
use polars::lazy::dsl::lit;
use polars::lazy::dsl::Expr;
use polars::lazy::dsl::GetOutput;

use polars::prelude::*;

use polars::series::Series;
use polars::{
    self,
    lazy::dsl::{col, cols},
    prelude::{DataFrame, IntoLazy},
};
use rayon::prelude::IntoParallelRefIterator;
use rayon::prelude::ParallelIterator;
use std::fs::File;
use std::path::PathBuf;

use crate::constants::POSE_COLUMNS;
use crate::geometry::se3::SE3;
use image::io::Reader as ImageReader;

use crate::geometry::so3::_quat_to_mat3;

/// Read a feather file and load into a `polars` dataframe.
pub fn read_feather_eager(path: &PathBuf, memory_mapped: bool) -> DataFrame {
    let file =
        File::open(path).unwrap_or_else(|_| panic!("{path} not found.", path = path.display()));
    polars::io::ipc::IpcReader::new(file)
        .memory_mapped(memory_mapped)
        .finish()
        .unwrap()
}

/// Write a feather file to disk using LZ4 compression.
pub fn write_feather_eager(path: &PathBuf, mut data_frame: DataFrame) {
    let file = File::create(path).expect("could not create file");
    IpcWriter::new(file)
        .with_compression(Some(IpcCompression::LZ4))
        .finish(&mut data_frame)
        .unwrap()
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
    let rotation = _quat_to_mat3(&quat_wxyz.view());
    let city_se3_ego = SE3 {
        rotation,
        translation,
    };
    let ego_se3_city = city_se3_ego.inverse();
    let indices: Vec<_> = (start_idx..=idx).collect();
    let mut lidar_list = indices
        .par_iter()
        .filter_map(|i| {
            let log_id_i = log_ids.get(*i).unwrap();
            match log_id_i == log_id {
                true => Some(i),
                _ => None,
            }
        })
        .map(|i| {
            let timestamp_ns_i = timestamps.get(*i).unwrap();
            let lidar_path = build_lidar_file_path(log_dir.clone(), timestamp_ns_i);
            let mut lidar = read_feather_eager(&lidar_path, memory_mapped)
                .lazy()
                .with_column(
                    col("x")
                        .map(
                            |x| Ok(Some(Series::from_vec("timedelta_ns", vec![0_f32; x.len()]))),
                            GetOutput::float_type(),
                        )
                        .alias("timedelta_ns"),
                );
            if timestamp_ns_i != timestamp_ns {
                let xyz = lidar
                    .clone()
                    .select(&[cols(["x", "y", "z"])])
                    .collect()
                    .unwrap()
                    .to_ndarray::<Float32Type>(IndexOrder::C)
                    .unwrap();

                let pose_i = poses
                    .clone()
                    .lazy()
                    .filter(col("timestamp_ns").eq(lit(timestamp_ns_i)))
                    .select(&[cols(POSE_COLUMNS)])
                    .collect()
                    .unwrap();

                let city_se3_ego_i = data_frame_to_se3(pose_i);
                let ego_ref_se3_ego_i = ego_se3_city.compose(&city_se3_ego_i);
                let xyz_ref = ego_ref_se3_ego_i.transform_from(&xyz.view());
                let x_ref = Series::new("x", xyz_ref.slice(s![.., 0]).to_owned().into_raw_vec());
                let y_ref = Series::new("y", xyz_ref.slice(s![.., 1]).to_owned().into_raw_vec());
                let z_ref = Series::new("z", xyz_ref.slice(s![.., 2]).to_owned().into_raw_vec());
                let timedelta_ns = Series::new(
                    "timedelta_ns",
                    vec![(timestamp_ns - timestamp_ns_i) as f32 * 1e-9; xyz.shape()[0]],
                );
                lidar =
                    lidar.with_columns(vec![lit(x_ref), lit(y_ref), lit(z_ref), lit(timedelta_ns)]);
            }
            lidar
        })
        .collect::<Vec<_>>();

    lidar_list.reverse();
    concat(lidar_list, UnionArgs::default()).unwrap()
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

/// Read an image into an RGBA u8 image.
pub fn read_image_rgba8(path: &PathBuf) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
    ImageReader::open(path)
        .unwrap()
        .decode()
        .unwrap()
        .to_rgba8()
}

/// Read an image into an RGBA u8 image and convert to `ndarray`.
pub fn read_image_rgba8_ndarray(path: &PathBuf) -> Array3<u8> {
    let image = ImageReader::open(path)
        .unwrap()
        .decode()
        .unwrap()
        .to_rgba8();
    image.into_ndarray3()
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
        .to_ndarray::<Float32Type>(IndexOrder::C)
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
        .to_ndarray::<Float32Type>(IndexOrder::C)
        .unwrap()
        .as_standard_layout()
        .to_owned()
}

/// Convert a data_frame with pose columns into an `se3` object.
pub fn data_frame_to_se3(data_frame: DataFrame) -> SE3 {
    let qw = extract_f32_from_data_frame(&data_frame, "qw");
    let qx = extract_f32_from_data_frame(&data_frame, "qx");
    let qy = extract_f32_from_data_frame(&data_frame, "qy");
    let qz = extract_f32_from_data_frame(&data_frame, "qz");
    let tx_m = extract_f32_from_data_frame(&data_frame, "tx_m");
    let ty_m = extract_f32_from_data_frame(&data_frame, "ty_m");
    let tz_m = extract_f32_from_data_frame(&data_frame, "tz_m");
    let quat_wxyz = Array::<f32, Ix1>::from_vec(vec![qw, qx, qy, qz]);
    let rotation = _quat_to_mat3(&quat_wxyz.view());
    let translation = Array::<f32, Ix1>::from_vec(vec![tx_m, ty_m, tz_m]);
    SE3 {
        rotation,
        translation,
    }
}

/// Extract an f32 field from a single row data frame.
pub fn extract_f32_from_data_frame(data_frame: &DataFrame, column: &str) -> f32 {
    data_frame
        .column(column)
        .unwrap()
        .get(0)
        .unwrap()
        .try_extract::<f32>()
        .unwrap()
}

/// Extract a usize field from a single row data frame.
pub fn extract_usize_from_data_frame(data_frame: &DataFrame, column: &str) -> usize {
    data_frame[column]
        .get(0)
        .unwrap()
        .try_extract::<usize>()
        .unwrap()
}
