//! # dataloaders
//!
//! Dataloader for loading the sensor dataset.

use constants::{ANNOTATION_COLUMNS, POSE_COLUMNS};
use io::{read_accumulate_lidar, read_timestamped_feather};
use itertools::{multiunzip, Itertools};
use path::{extract_file_stem, walk_dir};
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use rayon::slice::ParallelSliceMut;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use anyhow::Result;
use bincode::{deserialize, serialize};
use glob::glob;
use polars::prelude::*;
use pyo3::types::PyBytes;

use crate::{
    constants::{self, DEFAULT_MAP_FILE_NAME},
    io, path,
};

const MIN_NUM_LIDAR_PTS: u64 = 1;

/// Data associated with a single lidar sweep.
#[pyclass]
pub struct Sweep {
    /// Ground truth annotations.
    #[pyo3(get, set)]
    pub annotations: Option<PyDataFrame>,
    /// Ego-vehicle city pose.
    #[pyo3(get, set)]
    pub city_pose: PyDataFrame,
    /// Point cloud associated with the sweep.
    #[pyo3(get, set)]
    pub lidar: PyDataFrame,
    /// Log id and nanosecond timestamp (unique identifier).
    #[pyo3(get, set)]
    pub sweep_uuid: (String, u64),
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
            annotations: Some(annotations),
            city_pose,
            lidar,
            sweep_uuid,
        }
    }
}

/// Sensor dataloader for `av2`.
#[derive(Serialize, Deserialize)]
#[pyclass(module = "av2._r")]
pub struct Dataloader {
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
    /// Boolean flag to enable memory-mapped dataframe loading.
    #[pyo3(get, set)]
    pub memory_mapped: bool,
    /// Dataframe consisting of `log_id`, `timestamp_ns`, and `city_name`.
    #[pyo3(get, set)]
    pub file_index: PyDataFrame,
    /// Current index of the dataloader.
    #[pyo3(get, set)]
    pub current_index: usize,
}

#[pymethods]
impl Dataloader {
    /// Initialize the dataloader and build the file index.
    #[new]
    pub fn new(
        root_dir: &str,
        dataset_name: &str,
        dataset_type: &str,
        split_name: &str,
        num_accumulated_sweeps: usize,
        memory_mapped: bool,
    ) -> Dataloader {
        let root_dir = Path::new(root_dir);
        let file_index = build_file_index(root_dir, dataset_name, dataset_type, split_name);
        let current_idx = 0;
        Dataloader {
            root_dir: root_dir.to_path_buf(),
            dataset_name: dataset_name.to_string(),
            dataset_type: dataset_type.to_string(),
            split_name: split_name.to_string(),
            num_accumulated_sweeps,
            memory_mapped,
            file_index: PyDataFrame(file_index),
            current_index: current_idx,
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
        let (log_id, timestamp_ns) = unsafe {
            (
                row.get_unchecked(0).get_str().unwrap(),
                row.get_unchecked(1).try_extract::<u64>().unwrap(),
            )
        };

        // Annotations aren't available for the test set.
        let annotations = match self.split_name.as_str() {
            "test" => None,
            _ => Some(self.read_annotations(log_id, timestamp_ns)),
        };

        let city_pose = self.read_city_pose(log_id, timestamp_ns);
        let lidar = self.read_lidar(log_id, timestamp_ns, index);
        let sweep_uuid = (log_id.to_string(), timestamp_ns);

        Sweep {
            annotations,
            city_pose,
            lidar,
            sweep_uuid,
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

impl Iterator for Dataloader {
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

    let city_name_index = 7;
    let (log_ids, timestamp_nss, city_names): (Vec<_>, Vec<_>, Vec<_>) =
        multiunzip(entry_set.into_iter().map(|log_path| {
            let log_id = extract_file_stem(&log_path).unwrap();
            let lidar_dir = log_path.join("sensors/lidar");
            let x = log_path.join("map/log_map_archive_*.json");
            let city_name = glob(x.to_str().unwrap())
                .unwrap()
                .find_map(Result::ok)
                .unwrap_or(DEFAULT_MAP_FILE_NAME.into())
                .to_str()
                .unwrap()
                .split('_')
                .collect_vec()[city_name_index]
                .to_string();

            let mut lidar_entry_set: Vec<_> = glob(lidar_dir.join("*.feather").to_str().unwrap())
                .unwrap()
                .filter_map(|x| x.ok())
                .map(|x| x.as_path().to_owned())
                .collect();
            lidar_entry_set.par_sort();

            let num_entries = lidar_entry_set.len();
            (
                vec![log_id; num_entries],
                lidar_entry_set
                    .into_iter()
                    .map(|lidar_path| {
                        let lidar_path = lidar_path.file_stem().unwrap().to_str();
                        lidar_path.unwrap().parse::<u64>().unwrap()
                    })
                    .collect_vec(),
                vec![city_name; num_entries],
            )
        }));
    df!(
        "log_id" => log_ids.into_iter().flatten().collect_vec(),
        "timestamp_ns" => timestamp_nss.into_iter().flatten().collect_vec(),
        "city_name" => city_names.into_iter().flatten().collect_vec(),
    )
    .unwrap()
}
