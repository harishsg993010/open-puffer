//! Segment and search benchmarks.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use puffer_bench::{generate_ids, random_vector, random_vectors};
use puffer_core::Metric;
use puffer_index::{
    kmeans::{build_cluster_data, kmeans, KMeansConfig},
    search::search_segment,
};
use puffer_storage::{LoadedSegment, SegmentBuilder};
use std::path::PathBuf;
use tempfile::tempdir;

fn bench_kmeans(c: &mut Criterion) {
    let mut group = c.benchmark_group("kmeans");
    group.sample_size(10);

    for (num_vectors, num_clusters) in [(10_000, 100), (50_000, 200), (100_000, 300)] {
        let dim = 768;
        let vectors = random_vectors(num_vectors, dim);

        group.throughput(Throughput::Elements(num_vectors as u64));
        group.bench_with_input(
            BenchmarkId::new(format!("k={}", num_clusters), num_vectors),
            &vectors,
            |bench, vectors| {
                bench.iter(|| {
                    let config = KMeansConfig {
                        num_clusters,
                        max_iterations: 10,
                        convergence_threshold: 0.01,
                        seed: Some(42),
                    };
                    kmeans(black_box(vectors), &config)
                })
            },
        );
    }

    group.finish();
}

fn bench_segment_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("segment_build");
    group.sample_size(10);

    for num_vectors in [10_000, 50_000, 100_000] {
        let dim = 768;
        let num_clusters = (num_vectors as f64).sqrt() as usize;

        let vectors = random_vectors(num_vectors, dim);
        let ids = generate_ids(num_vectors);
        let payloads: Vec<Option<serde_json::Value>> = vec![None; num_vectors];

        // Pre-compute clustering
        let config = KMeansConfig {
            num_clusters,
            max_iterations: 10,
            convergence_threshold: 0.01,
            seed: Some(42),
        };
        let result = kmeans(&vectors, &config);
        let cluster_data = build_cluster_data(&result.centroids, &result.assignments);

        let dir = tempdir().unwrap();

        group.throughput(Throughput::Elements(num_vectors as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(num_vectors),
            &(cluster_data, vectors, ids, payloads, dir.path().to_path_buf()),
            |bench, (cluster_data, vectors, ids, payloads, dir_path)| {
                bench.iter(|| {
                    let path = dir_path.join(format!("bench_{}.seg", rand::random::<u32>()));
                    let builder = SegmentBuilder::new(768, Metric::Cosine);
                    builder
                        .build(
                            cluster_data.clone(),
                            black_box(vectors),
                            black_box(ids),
                            black_box(payloads),
                            &path,
                        )
                        .unwrap();
                    // Clean up
                    let _ = std::fs::remove_file(&path);
                })
            },
        );
    }

    group.finish();
}

fn bench_segment_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("segment_search");
    group.sample_size(50);

    // Create a test segment
    let num_vectors = 100_000;
    let dim = 768;
    let num_clusters = 100;

    let vectors = random_vectors(num_vectors, dim);
    let ids = generate_ids(num_vectors);
    let payloads: Vec<Option<serde_json::Value>> = vec![None; num_vectors];

    let config = KMeansConfig {
        num_clusters,
        max_iterations: 15,
        convergence_threshold: 0.001,
        seed: Some(42),
    };
    let result = kmeans(&vectors, &config);
    let cluster_data = build_cluster_data(&result.centroids, &result.assignments);

    let dir = tempdir().unwrap();
    let path = dir.path().join("bench.seg");

    let builder = SegmentBuilder::new(dim, Metric::Cosine);
    builder
        .build(cluster_data, &vectors, &ids, &payloads, &path)
        .unwrap();

    let segment = LoadedSegment::open(&path).unwrap();

    for nprobe in [1, 4, 8, 16] {
        let query = random_vector(dim);

        group.bench_with_input(
            BenchmarkId::new("nprobe", nprobe),
            &(&segment, &query, nprobe),
            |bench, (segment, query, nprobe)| {
                bench.iter(|| search_segment(black_box(*segment), black_box(*query), 10, *nprobe, false))
            },
        );
    }

    for top_k in [1, 10, 50, 100] {
        let query = random_vector(dim);

        group.bench_with_input(
            BenchmarkId::new("top_k", top_k),
            &(&segment, &query, top_k),
            |bench, (segment, query, top_k)| {
                bench.iter(|| search_segment(black_box(*segment), black_box(*query), *top_k, 4, false))
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_kmeans, bench_segment_build, bench_segment_search);
criterion_main!(benches);
