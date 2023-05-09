//! # benchmark
//!
//! Benchmarking suite.

use av2::{
    geometry::polytope::{compute_interior_points_mask, cuboids_to_polygons},
    io::{ndarray_from_frame, read_feather_eager},
};
use criterion::{criterion_group, criterion_main, Criterion};
use once_cell::sync::Lazy;
use polars::{
    lazy::dsl::{col, cols, lit},
    prelude::IntoLazy,
};
use std::{env::current_dir, path::PathBuf};

static TEST_DATA_DIR: Lazy<PathBuf> = Lazy::new(|| {
    current_dir()
        .unwrap()
        .join("../tests/unit/test_data/sensor_dataset_logs/adcf7d18-0510-35b0-a2fa-b4cea13a6d76")
});

fn io_benchmark(c: &mut Criterion) {
    let path = current_dir().unwrap().join("../tests/unit/test_data/sensor_dataset_logs/adcf7d18-0510-35b0-a2fa-b4cea13a6d76/sensors/lidar/315973157959879000.feather");
    c.bench_function("read_feather_eager", |b| {
        b.iter(|| read_feather_eager(&path, false))
    });
}

fn geometry_benchmark(c: &mut Criterion) {
    let timestamp_ns = 315973157959879000_u64;
    let annotations_path = TEST_DATA_DIR.join("annotations.feather");
    let annotations = read_feather_eager(&annotations_path, false)
        .lazy()
        .filter(col("timestamp_ns").eq(lit(timestamp_ns)))
        .collect()
        .unwrap();
    let columns = [
        "tx_m", "ty_m", "tz_m", "length_m", "width_m", "height_m", "qw", "qx", "qy", "qz",
    ];
    let annotations_ndarray = ndarray_from_frame(&annotations, cols(columns));
    let cuboid_vertices = cuboids_to_polygons(&annotations_ndarray.view());
    let lidar_path = TEST_DATA_DIR.join(format!("sensors/lidar/{timestamp_ns}.feather"));
    let lidar = read_feather_eager(&lidar_path, false);
    let lidar_ndarray = ndarray_from_frame(&lidar, cols(["x", "y", "z"]));
    c.bench_function("compute_interior_points_mask", |b| {
        b.iter(|| compute_interior_points_mask(&lidar_ndarray.view(), &cuboid_vertices.view()))
    });
}

criterion_group!(benches, io_benchmark, geometry_benchmark);
criterion_main!(benches);
