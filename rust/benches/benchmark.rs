/// Benchmarking suite.
use av2::io::read_feather_eager;
use criterion::{criterion_group, criterion_main, Criterion};
use std::env::current_dir;

fn criterion_benchmark(c: &mut Criterion) {
    let path = current_dir().unwrap().join("../tests/unit/test_data/sensor_dataset_logs/adcf7d18-0510-35b0-a2fa-b4cea13a6d76/sensors/lidar/315973157959879000.feather");
    c.bench_function("read_feather_eager", |b| {
        b.iter(|| read_feather_eager(&path, false))
    });
    // c.bench_function("read_feather_lazy", |b| b.iter(|| read_feather_lazy(&path, false)));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
