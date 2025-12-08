//! Benchmarks for distance functions.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use puffer_core::{cosine_distance, l2_distance_squared};
use rand::Rng;

fn random_vector(dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect()
}

fn bench_l2_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("l2_distance_squared");

    for dim in [128, 256, 512, 768, 1024, 1536].iter() {
        let a = random_vector(*dim);
        let b = random_vector(*dim);

        group.throughput(Throughput::Elements(*dim as u64));
        group.bench_with_input(BenchmarkId::from_parameter(dim), dim, |bench, _| {
            bench.iter(|| l2_distance_squared(black_box(&a), black_box(&b)))
        });
    }

    group.finish();
}

fn bench_cosine_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine_distance");

    for dim in [128, 256, 512, 768, 1024, 1536].iter() {
        let a = random_vector(*dim);
        let b = random_vector(*dim);

        group.throughput(Throughput::Elements(*dim as u64));
        group.bench_with_input(BenchmarkId::from_parameter(dim), dim, |bench, _| {
            bench.iter(|| cosine_distance(black_box(&a), black_box(&b)))
        });
    }

    group.finish();
}

fn bench_batch_distances(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_l2_distance");

    let dim = 768;
    let query = random_vector(dim);

    for num_vectors in [100, 1000, 10000].iter() {
        let vectors: Vec<Vec<f32>> = (0..*num_vectors).map(|_| random_vector(dim)).collect();

        group.throughput(Throughput::Elements(*num_vectors as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(num_vectors),
            num_vectors,
            |bench, _| {
                bench.iter(|| {
                    vectors
                        .iter()
                        .map(|v| l2_distance_squared(black_box(&query), black_box(v)))
                        .collect::<Vec<_>>()
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_l2_distance,
    bench_cosine_distance,
    bench_batch_distances
);
criterion_main!(benches);
