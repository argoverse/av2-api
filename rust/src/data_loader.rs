//! # data_loaders
//!
//! Data-loader for loading the sensor dataset.

use constants::{ANNOTATION_COLUMNS, POSE_COLUMNS};

use io::{read_accumulate_lidar, read_timestamped_feather};
use itertools::Itertools;
use ndarray::Array3;
use path::walk_dir;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use rayon::{prelude::IntoParallelRefIterator, slice::ParallelSliceMut};
use serde::{Deserialize, Serialize};
use std::{
    path::{Path, PathBuf},
    str::FromStr,
};
use strum::IntoEnumIterator;

use bincode::{deserialize, serialize};
use glob::glob;
use polars::prelude::*;
use pyo3::types::PyBytes;

use crate::{
    constants::{self, CameraNames},
    io::{self, read_image},
    path,
};
use rayon::iter::IndexedParallelIterator;
use rayon::iter::ParallelIterator;

const MIN_NUM_LIDAR_PTS: u64 = 1;
const MAX_CAM_LIDAR_TOL_NS: f32 = 50000000.;

/// Data associated with a single lidar sweep.
#[pyclass]
pub struct Sweep {
    /// Ego-vehicle city pose.
    #[pyo3(get, set)]
    pub city_pose: PyDataFrame,
    /// Point cloud associated with the sweep.
    #[pyo3(get, set)]
    pub lidar: PyDataFrame,
    /// Log id and nanosecond timestamp (unique identifier).
    #[pyo3(get, set)]
    pub sweep_uuid: (String, u64),
    /// Cuboids associated with the sweep.
    #[pyo3(get, set)]
    pub cuboids: Option<PyDataFrame>,
}

/// Encapsulates sensor data associated with a single sweep.
#[pymethods]
impl Sweep {
    /// Initialize a sweep object.
    #[new]
    pub fn new(
        annotations: PyDataFrame,
        city_pose: PyDataFrame,
        lidar: PyDataFrame,
        sweep_uuid: (String, u64),
    ) -> Sweep {
        Sweep {
            city_pose,
            lidar,
            sweep_uuid,
            cuboids: Some(annotations),
        }
    }
}

/// Sensor data-loader for `av2`.
#[derive(Serialize, Deserialize)]
#[pyclass(module = "av2._r")]
pub struct DataLoader {
    /// Root dataset directory.
    #[pyo3(get, set)]
    pub root_dir: PathBuf,
    /// Dataset name (e.g., `av2`).
    #[pyo3(get, set)]
    pub dataset_name: String,
    /// Root dataset type (e.g., `sensor`).
    #[pyo3(get, set)]
    pub dataset_type: String,
    /// Root dataset split name (e.g., `train`).
    #[pyo3(get, set)]
    pub split_name: String,
    /// Number of accumulated lidar sweeps.
    #[pyo3(get, set)]
    pub num_accumulated_sweeps: usize,
    /// Boolean flag to enable memory-mapped data-frame loading.
    #[pyo3(get, set)]
    pub memory_mapped: bool,
    /// Data-frame consisting of `log_id`, `timestamp_ns`, and `city_name`.
    #[pyo3(get, set)]
    pub file_index: PyDataFrame,
    /// Current index of the data-loader.
    #[pyo3(get, set)]
    pub current_index: usize,
}

#[pymethods]
impl DataLoader {
    /// Initialize the data-loader and build the file index.
    #[new]
    pub fn new(
        root_dir: &str,
        dataset_name: &str,
        dataset_type: &str,
        split_name: &str,
        num_accumulated_sweeps: usize,
        memory_mapped: bool,
    ) -> DataLoader {
        let root_dir = PathBuf::from_str(root_dir).unwrap();
        let file_index = PyDataFrame(build_file_index(
            root_dir.as_path(),
            dataset_name,
            dataset_type,
            split_name,
        ));

        let dataset_name = dataset_name.to_string();
        let dataset_type = dataset_type.to_string();
        let split_name = split_name.to_string();
        let current_index = 0;
        DataLoader {
            root_dir,
            dataset_name,
            dataset_type,
            split_name,
            num_accumulated_sweeps,
            memory_mapped,
            file_index,
            current_index,
        }
    }

    fn split_dir(&self) -> PathBuf {
        self.root_dir
            .join(&self.dataset_name)
            .join(&self.dataset_type)
            .join(&self.split_name)
    }

    fn log_dir(&self, log_id: &str) -> PathBuf {
        self.split_dir().join(log_id)
    }

    fn lidar_path(&self, log_id: &str, timestamp_ns: u64) -> PathBuf {
        let file_name = format!("{timestamp_ns}.feather");
        let lidar_path = [
            self.log_dir(log_id),
            "sensors".to_string().into(),
            "lidar".to_string().into(),
            file_name.into(),
        ]
        .iter()
        .collect();
        lidar_path
    }

    fn camera_path(&self, log_id: &str, camera_name: &str, timestamp_ns: u64) -> PathBuf {
        let file_name = format!("{timestamp_ns}.jpg");
        let sensor_path = [
            self.log_dir(log_id),
            "sensors".to_string().into(),
            "cameras".to_string().into(),
            camera_name.to_string().into(),
            file_name.into(),
        ]
        .iter()
        .collect();
        sensor_path
    }

    fn annotations_path(&self, log_id: &str) -> PathBuf {
        self.log_dir(log_id).join("annotations.feather")
    }

    fn city_pose_path(&self, log_id: &str) -> PathBuf {
        self.log_dir(log_id).join("city_SE3_egovehicle.feather")
    }

    fn map_dir(&self, log_id: &str) -> PathBuf {
        self.log_dir(log_id).join("map")
    }

    fn read_city_pose(&self, log_id: &str, timestamp_ns: u64) -> PyDataFrame {
        PyDataFrame(
            read_timestamped_feather(
                &self.city_pose_path(log_id),
                &POSE_COLUMNS.to_vec(),
                &timestamp_ns,
                self.memory_mapped,
            )
            .collect()
            .unwrap(),
        )
    }

    fn read_lidar(&self, log_id: &str, timestamp_ns: u64, index: usize) -> PyDataFrame {
        PyDataFrame(
            read_accumulate_lidar(
                self.log_dir(log_id),
                &self.file_index.0,
                log_id,
                timestamp_ns,
                index,
                self.num_accumulated_sweeps,
                self.memory_mapped,
            )
            .collect()
            .unwrap(),
        )
    }

    fn read_annotations(&self, log_id: &str, timestamp_ns: u64) -> PyDataFrame {
        PyDataFrame(
            read_timestamped_feather(
                &self.annotations_path(log_id),
                &ANNOTATION_COLUMNS.to_vec(),
                &timestamp_ns,
                self.memory_mapped,
            )
            .filter(col("num_interior_pts").gt_eq(MIN_NUM_LIDAR_PTS))
            .collect()
            .unwrap(),
        )
    }

    /// Get the sweep at `index`.
    pub fn get(&self, index: usize) -> Sweep {
        let row = self.file_index.0.get_row(index).unwrap().0;
        let (log_id, timestamp_ns) = (
            row.get(0).unwrap().get_str().unwrap(),
            row.get(1).unwrap().try_extract::<u64>().unwrap(),
        );

        // Annotations aren't available for the test set.
        let cuboids = match self.split_name.as_str() {
            "test" => None,
            _ => Some(self.read_annotations(log_id, timestamp_ns)),
        };

        let city_pose = self.read_city_pose(log_id, timestamp_ns);
        let lidar = self.read_lidar(log_id, timestamp_ns, index);
        let sweep_uuid = (log_id.to_string(), timestamp_ns);

        Sweep {
            city_pose,
            lidar,
            sweep_uuid,
            cuboids,
        }
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<Sweep> {
        slf.next()
    }

    fn __len__(slf: PyRef<'_, Self>) -> usize {
        slf.file_index.0.shape().0
    }

    /// Used for python pickling.
    pub fn __setstate__(&mut self, state: &PyBytes) -> PyResult<()> {
        *self = deserialize(state.as_bytes()).unwrap();
        Ok(())
    }

    /// Used for python pickling.
    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<&'py PyBytes> {
        Ok(PyBytes::new(py, &serialize(&self).unwrap()))
    }

    /// Used for python pickling.
    pub fn __getnewargs__(&self) -> PyResult<(PathBuf, String, String, String, usize)> {
        Ok((
            self.root_dir.clone(),
            self.dataset_type.clone(),
            self.split_name.clone(),
            self.dataset_name.clone(),
            self.num_accumulated_sweeps,
        ))
    }
}

/// Rust methods.
impl DataLoader {
    /// Get the sweep at `index`.
    pub fn get_synchronized_images(&self, index: usize) -> Vec<Array3<u8>> {
        let row = self.file_index.0.get_row(index).unwrap().0;
        let (log_id, _) = (
            row.get(0).unwrap().get_str().unwrap(),
            row.get(1).unwrap().try_extract::<u64>().unwrap(),
        );

        let camera_names = CameraNames::iter().collect_vec();
        let images = camera_names
            .par_iter()
            .enumerate()
            .map(|(i, camera_name)| {
                let maybe_timestamp_ns_camera = row.get(i + 3).unwrap().try_extract::<u64>();
                match maybe_timestamp_ns_camera {
                    Ok(maybe_timestamp_ns_camera) => {
                        let timestamp_ns_camera = maybe_timestamp_ns_camera;
                        let camera_path = self.camera_path(
                            log_id,
                            camera_name.to_string().as_str(),
                            timestamp_ns_camera,
                        );
                        read_image(&camera_path)
                    }
                    _ => {
                        let camera_name = camera_name.to_string();
                        if camera_name == "ring_front_center" {
                            let width = 1550;
                            let height = 2048;
                            Array3::<u8>::zeros([4, height, width])
                        } else {
                            let width = 2048;
                            let height = 1550;
                            Array3::<u8>::zeros([4, height, width])
                        }
                }
            })
            .collect();
        images
    }
}

impl Iterator for DataLoader {
    type Item = Sweep;

    fn next(&mut self) -> Option<Self::Item> {
        let idx = self.current_index;
        let sweep_data = self.get(idx);
        self.current_index += 1;

        Some(sweep_data)
    }
}

fn build_file_index(
    root_dir: &Path,
    dataset_name: &str,
    dataset_type: &str,
    split_name: &str,
) -> DataFrame {
    let split_dir = root_dir.join(format!("{dataset_name}/{dataset_type}/{split_name}"));
    let mut entry_set: Vec<_> = walk_dir(&split_dir)
        .unwrap_or_else(|_| panic!("Cannot walk the following directory: {split_dir:?}"));
    entry_set.par_sort();

    let mut reference_frame = build_lidar_metadata(split_dir.clone(), "lidar");
    reference_frame = reference_frame
        .clone()
        .lazy()
        .sort_by_exprs(
            &[cols(["log_id", "timestamp_ns_lidar"])],
            vec![false],
            false,
        )
        .collect()
        .unwrap();

    for camera_name in CameraNames::iter().map(|x| x.to_string()) {
        let frame = build_camera_metadata(split_dir.clone(), &camera_name);
        frame
            .clone()
            .lazy()
            .sort_by_exprs(
                &[cols([
                    "log_id",
                    format!("timestamp_ns_{camera_name}").as_str(),
                ])],
                vec![false],
                false,
            )
            .collect()
            .unwrap();

        reference_frame = reference_frame
            .join_asof_by(
                &frame,
                "timestamp_ns_lidar",
                format!("timestamp_ns_{}", camera_name).as_str(),
                ["log_id"],
                ["log_id"],
                AsofStrategy::Forward,
                Some(AnyValue::Float32(MAX_CAM_LIDAR_TOL_NS)),
            )
            .unwrap();

        reference_frame = reference_frame.drop_many(&["city_name_right"]);
    }
    reference_frame
        .rename("timestamp_ns_lidar", "timestamp_ns")
        .unwrap();
    reference_frame
}

fn build_lidar_metadata(split_dir: PathBuf, sensor_name: &str) -> DataFrame {
    let pattern = split_dir.join(format!("*/sensors/{sensor_name}/*.feather"));
    let files = glob(pattern.to_str().unwrap())
        .expect("Failed to read glob pattern.")
        .filter_map(|path| match path {
            Ok(x) => Some(x),
            _ => None,
        })
        .collect_vec();

    let log_id: Vec<_> = files
        .iter()
        .map(|x| {
            x.ancestors()
                .nth(3)
                .unwrap()
                .file_stem()
                .unwrap()
                .to_str()
                .unwrap()
        })
        .collect();

    let timestamp_ns: Vec<_> = files
        .clone()
        .iter()
        .map(|x| {
            x.file_stem()
                .unwrap()
                .to_str()
                .unwrap()
                .parse::<u64>()
                .unwrap()
        })
        .collect();
    df!(
        "log_id" => Series::new("log_id", log_id),
        format!("timestamp_ns_{}", sensor_name).as_str() => Series::new("timestamp_ns", timestamp_ns.clone()),
        "city_name" => Series::new("city_name", vec!["DEFAULT"; timestamp_ns.len()])
    )
    .unwrap()
}

fn build_camera_metadata(split_dir: PathBuf, sensor_name: &str) -> DataFrame {
    let pattern = split_dir.join(format!("*/sensors/cameras/{sensor_name}/*.jpg"));
    let files = glob(pattern.to_str().unwrap())
        .expect("Failed to read glob pattern.")
        .filter_map(|path| match path {
            Ok(x) => Some(x),
            _ => None,
        })
        .collect_vec();

    let log_id: Vec<_> = files
        .iter()
        .map(|x| {
            x.ancestors()
                .nth(4)
                .unwrap()
                .file_stem()
                .unwrap()
                .to_str()
                .unwrap()
        })
        .collect();

    let timestamp_ns: Vec<_> = files
        .clone()
        .iter()
        .map(|x| {
            x.file_stem()
                .unwrap()
                .to_str()
                .unwrap()
                .parse::<u64>()
                .unwrap()
        })
        .collect();
    df!(
        "log_id" => Series::new("log_id", log_id),
        format!("timestamp_ns_{}", sensor_name).as_str() => Series::new("timestamp_ns", timestamp_ns.clone()),
        "city_name" => Series::new("city_name", vec!["DEFAULT"; timestamp_ns.len()])
    )
    .unwrap()
}
