pub mod io;
pub mod ops;
pub mod path;
pub mod se3;
pub mod so3;

use io::{read_filter_timestamp, read_lidar};
use itertools::{multiunzip, Itertools};
use ndarray::Dim;
use numpy::{IntoPyArray, PyArray};
use path::{file_stem, walk_dir};
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use rayon::slice::ParallelSliceMut;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use anyhow::Result;
use bincode::{deserialize, serialize};
use glob::glob;
use numpy::PyReadonlyArray2;
use polars::prelude::*;
use pyo3::types::PyBytes;

use crate::ops::voxelize;

pub const ANNOTATION_COLUMNS: [&str; 13] = [
    "tx_m",
    "ty_m",
    "tz_m",
    "length_m",
    "width_m",
    "height_m",
    "qw",
    "qx",
    "qy",
    "qz",
    "num_interior_pts",
    "category",
    "track_uuid",
];

pub const POSE_COLUMNS: [&str; 7] = ["tx_m", "ty_m", "tz_m", "qw", "qx", "qy", "qz"];

#[pyclass]
pub struct Sweep {
    #[pyo3(get, set)]
    pub annotations: PyDataFrame,
    #[pyo3(get, set)]
    pub city_pose: PyDataFrame,
    #[pyo3(get, set)]
    pub lidar: PyDataFrame,
    #[pyo3(get, set)]
    pub sweep_uuid: (String, u64),
}

#[pymethods]
impl Sweep {
    #[new]
    pub fn new(
        annotations: PyDataFrame,
        city_pose: PyDataFrame,
        lidar: PyDataFrame,
        sweep_uuid: (String, u64),
    ) -> Sweep {
        Sweep {
            annotations,
            city_pose,
            lidar,
            sweep_uuid,
        }
    }
}

#[derive(Serialize, Deserialize)]
#[pyclass(module = "av2._r")]
pub struct Dataloader {
    #[pyo3(get, set)]
    pub root_dir: PathBuf,
    #[pyo3(get, set)]
    pub dataset_name: String,
    #[pyo3(get, set)]
    pub dataset_type: String,
    #[pyo3(get, set)]
    pub split_name: String,
    #[pyo3(get, set)]
    pub num_accum_sweeps: usize,
    #[pyo3(get, set)]
    pub file_index: PyDataFrame,
    #[pyo3(get, set)]
    pub current_idx: usize,
    #[pyo3(get, set)]
    pub memory_map: bool,
}

#[pymethods]
impl Dataloader {
    #[new]
    pub fn new(
        root_dir: &str,
        dataset_type: &str,
        split_name: &str,
        dataset_name: &str,
        num_accum_sweeps: usize,
        memory_map: bool,
    ) -> Dataloader {
        let root_dir = Path::new(root_dir);
        let file_index = build_file_index(root_dir, dataset_name, dataset_type, split_name);
        let current_idx = 0;
        Dataloader {
            root_dir: root_dir.to_path_buf(),
            dataset_name: dataset_name.to_string(),
            dataset_type: dataset_type.to_string(),
            split_name: split_name.to_string(),
            num_accum_sweeps,
            file_index: PyDataFrame(file_index),
            current_idx,
            memory_map,
        }
    }

    pub fn split_dir(&self) -> PathBuf {
        self.root_dir
            .join(&self.dataset_name)
            .join(&self.dataset_type)
            .join(&self.split_name)
    }

    pub fn log_dir(&self, log_id: &str) -> PathBuf {
        self.split_dir().join(log_id)
    }

    pub fn lidar_path(&self, log_id: &str, timestamp_ns: u64) -> PathBuf {
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

    pub fn annotations_path(&self, log_id: &str) -> PathBuf {
        self.log_dir(log_id).join("annotations.feather")
    }

    pub fn city_pose_path(&self, log_id: &str) -> PathBuf {
        self.log_dir(log_id).join("city_SE3_egovehicle.feather")
    }

    pub fn map_dir(&self, log_id: &str) -> PathBuf {
        self.log_dir(log_id).join("map")
    }

    pub fn get(&self, idx: usize) -> Sweep {
        let log_id = self.file_index.0["log_id"]
            .utf8()
            .unwrap()
            .get(idx)
            .unwrap();
        let timestamp_ns = self.file_index.0["timestamp_ns"]
            .u64()
            .unwrap()
            .get(idx)
            .unwrap();
        let lidar = read_lidar(
            self.log_dir(log_id),
            &self.file_index.0,
            log_id,
            timestamp_ns,
            idx,
            self.num_accum_sweeps,
        )
        .collect()
        .unwrap();

        let annotations_path = self.annotations_path(log_id);
        let annotations = read_filter_timestamp(
            &annotations_path,
            &ANNOTATION_COLUMNS.to_vec(),
            &timestamp_ns,
            self.memory_map,
        )
        .filter(col("num_interior_pts").gt_eq(1.))
        .collect()
        .unwrap();

        let city_pose_path = self.city_pose_path(log_id);
        let city_pose = read_filter_timestamp(
            &city_pose_path,
            &POSE_COLUMNS.to_vec(),
            &timestamp_ns,
            self.memory_map,
        )
        .collect()
        .unwrap();

        let sweep_uuid = (log_id.to_string(), timestamp_ns);

        Sweep {
            annotations: PyDataFrame(annotations),
            city_pose: PyDataFrame(city_pose),
            lidar: PyDataFrame(lidar),
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

    pub fn __setstate__(&mut self, state: &PyBytes) -> PyResult<()> {
        *self = deserialize(state.as_bytes()).unwrap();
        Ok(())
    }
    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<&'py PyBytes> {
        Ok(PyBytes::new(py, &serialize(&self).unwrap()))
    }
    pub fn __getnewargs__(&self) -> PyResult<(PathBuf, String, String, String, usize)> {
        Ok((
            self.root_dir.clone(),
            self.dataset_type.clone(),
            self.split_name.clone(),
            self.dataset_name.clone(),
            self.num_accum_sweeps,
        ))
    }
}

impl Iterator for Dataloader {
    type Item = Sweep;

    fn next(&mut self) -> Option<Self::Item> {
        let idx = self.current_idx;
        let sweep_data = self.get(idx);
        self.current_idx += 1;

        Some(sweep_data)
    }
}

pub fn build_file_index(
    root_dir: &Path,
    dataset_name: &str,
    dataset_type: &str,
    split_name: &str,
) -> DataFrame {
    let split_dir = root_dir.join(format!("{dataset_name}/{dataset_type}/{split_name}"));
    let mut entry_set: Vec<_> = walk_dir(&split_dir)
        .unwrap_or_else(|_| panic!("Cannot walk the following directory: {split_dir:?}"));
    entry_set.par_sort();

    let (log_ids, timestamp_nss, city_names): (Vec<_>, Vec<_>, Vec<_>) =
        multiunzip(entry_set.into_iter().map(|log_path| {
            let log_id = file_stem(&log_path).unwrap();
            let lidar_dir = log_path.join("sensors/lidar");
            let x = log_path.join("map/log_map_archive_*.json");
            let city_name = glob(x.to_str().unwrap())
                .unwrap()
                .find_map(Result::ok)
                .unwrap_or("log_map_archive___DEFAULT_city_00000.json".into())
                .to_str()
                .unwrap()
                .split('_')
                .collect::<Vec<_>>()[7]
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
                    .collect::<Vec<_>>(),
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

#[pyfunction]
#[pyo3(name = "voxelize")]
fn py_voxelize<'py>(
    py: Python<'py>,
    indices: PyReadonlyArray2<usize>,
    features: PyReadonlyArray2<f32>,
    length: usize,
    width: usize,
    height: usize,
) -> (
    &'py PyArray<usize, Dim<[usize; 2]>>,
    &'py PyArray<f32, Dim<[usize; 2]>>,
    &'py PyArray<f32, Dim<[usize; 2]>>,
) {
    let (indices, values, counts) = voxelize(
        &indices.as_array(),
        &features.as_array(),
        length,
        width,
        height,
    );
    (
        indices.into_pyarray(py),
        values.into_pyarray(py),
        counts.into_pyarray(py),
    )
}

/// A Python module implemented in Rust.
#[pymodule]
fn _r(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Dataloader>()?;
    m.add_class::<Sweep>()?;
    m.add_function(wrap_pyfunction!(py_voxelize, m)?)?;
    Ok(())
}
