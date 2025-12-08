//! Recall benchmark tool for Puffer vector database.
//!
//! Compares brute-force exact search with ANN (IVF + router) search
//! to measure recall@K and latency at various nprobe values.

use clap::Parser;
use puffer_core::{distance, Metric, VectorId};
use puffer_query::QueryEngine;
use puffer_storage::{Catalog, LoadedSegment};
use rand::prelude::*;
use rand::SeedableRng;
use rayon::prelude::*;
use std::collections::HashSet;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

/// Recall benchmark tool for Puffer vector database.
#[derive(Parser, Debug)]
#[command(name = "recall-bench")]
#[command(about = "Benchmark recall@K of ANN search vs brute-force ground truth")]
struct Args {
    /// Path to the data directory containing collections.
    #[arg(long, default_value = "./data")]
    data_dir: PathBuf,

    /// Name of the collection to benchmark.
    #[arg(long)]
    collection: String,

    /// Number of query vectors to use.
    #[arg(long, default_value = "100")]
    num_queries: usize,

    /// Number of top results to retrieve (K in recall@K).
    #[arg(long, default_value = "10")]
    top_k: usize,

    /// Comma-separated list of nprobe values to test.
    #[arg(long, default_value = "4,8,12,16")]
    nprobe: String,

    /// Number of vectors to sample from collection for evaluation.
    /// If 0 or larger than collection size, uses all vectors.
    #[arg(long, default_value = "50000")]
    sample_size: usize,

    /// Output CSV file path.
    #[arg(long, default_value = "recall_results.csv")]
    output: PathBuf,

    /// Random seed for reproducibility.
    #[arg(long, default_value = "42")]
    seed: u64,

    /// Number of segments to search via router (0 = use collection default).
    #[arg(long, default_value = "0")]
    router_top_m: usize,
}

/// A sampled vector with its ID and data.
#[derive(Clone)]
struct SampledVector {
    id: VectorId,
    vector: Vec<f32>,
}

/// Results for a single nprobe configuration.
#[derive(Debug)]
struct BenchmarkResult {
    nprobe: usize,
    top_k: usize,
    num_queries: usize,
    mean_recall: f64,
    p50_latency_ms: f64,
    p95_latency_ms: f64,
    p99_latency_ms: f64,
    mean_latency_ms: f64,
    qps: f64,
}

fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let args = Args::parse();

    // Parse nprobe values
    let nprobe_values: Vec<usize> = args
        .nprobe
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    if nprobe_values.is_empty() {
        anyhow::bail!("No valid nprobe values provided");
    }

    tracing::info!("Starting recall benchmark");
    tracing::info!("  Collection: {}", args.collection);
    tracing::info!("  Data dir: {:?}", args.data_dir);
    tracing::info!("  Num queries: {}", args.num_queries);
    tracing::info!("  Top-K: {}", args.top_k);
    tracing::info!("  nprobe values: {:?}", nprobe_values);
    tracing::info!("  Sample size: {}", args.sample_size);
    tracing::info!("  Seed: {}", args.seed);

    // Open catalog and collection
    let catalog = Arc::new(Catalog::open(&args.data_dir)?);
    let engine = QueryEngine::new(catalog.clone());

    let coll_arc = catalog.get_collection(&args.collection)?;
    let coll = coll_arc.read().unwrap();

    let dim = coll.meta.config.dimension;
    let metric = coll.meta.config.metric;

    tracing::info!("Collection info:");
    tracing::info!("  Dimension: {}", dim);
    tracing::info!("  Metric: {:?}", metric);
    tracing::info!("  Total vectors: {}", coll.meta.total_vectors);
    tracing::info!("  Segments: {}", coll.meta.get_segment_names().len());

    // Load all vectors from segments
    tracing::info!("Loading vectors from segments...");
    let all_vectors = load_all_vectors(&coll, &catalog)?;
    tracing::info!("Loaded {} vectors", all_vectors.len());

    drop(coll); // Release lock

    if all_vectors.is_empty() {
        anyhow::bail!("No vectors found in collection");
    }

    // Sample vectors for evaluation
    let mut rng = StdRng::seed_from_u64(args.seed);
    let sample_size = if args.sample_size == 0 || args.sample_size >= all_vectors.len() {
        all_vectors.len()
    } else {
        args.sample_size
    };

    let sampled_vectors: Vec<SampledVector> = if sample_size == all_vectors.len() {
        all_vectors
    } else {
        let indices: Vec<usize> = (0..all_vectors.len()).choose_multiple(&mut rng, sample_size);
        indices.into_iter().map(|i| all_vectors[i].clone()).collect()
    };

    tracing::info!("Using {} sampled vectors for evaluation", sampled_vectors.len());

    // Select query vectors from sampled set
    let num_queries = args.num_queries.min(sampled_vectors.len());
    let query_indices: Vec<usize> = (0..sampled_vectors.len())
        .choose_multiple(&mut rng, num_queries);
    let query_vectors: Vec<&SampledVector> = query_indices
        .iter()
        .map(|&i| &sampled_vectors[i])
        .collect();

    tracing::info!("Selected {} query vectors", query_vectors.len());

    // Compute ground truth using brute-force search
    tracing::info!("Computing ground truth (brute-force search)...");
    let ground_truth = compute_ground_truth(
        &query_vectors,
        &sampled_vectors,
        metric,
        args.top_k,
    );
    tracing::info!("Ground truth computed for {} queries", ground_truth.len());

    // Run benchmarks for each nprobe value
    let mut results = Vec::new();

    for &nprobe in &nprobe_values {
        tracing::info!("Benchmarking with nprobe={}", nprobe);

        let result = run_ann_benchmark(
            &engine,
            &args.collection,
            &query_vectors,
            &ground_truth,
            args.top_k,
            nprobe,
            if args.router_top_m > 0 { Some(args.router_top_m) } else { None },
        )?;

        tracing::info!(
            "  nprobe={}: recall={:.4}, p50={:.2}ms, p95={:.2}ms, qps={:.0}",
            result.nprobe,
            result.mean_recall,
            result.p50_latency_ms,
            result.p95_latency_ms,
            result.qps
        );

        results.push(result);
    }

    // Write results to CSV
    write_csv(&args.output, &results)?;
    tracing::info!("Results written to {:?}", args.output);

    // Print summary
    println!("\n{}", "=".repeat(80));
    println!("RECALL BENCHMARK RESULTS");
    println!("{}", "=".repeat(80));
    println!(
        "{:>8} {:>8} {:>12} {:>12} {:>12} {:>12} {:>12} {:>10}",
        "nprobe", "top_k", "queries", "recall", "p50_ms", "p95_ms", "p99_ms", "qps"
    );
    println!("{}", "-".repeat(80));

    for r in &results {
        println!(
            "{:>8} {:>8} {:>12} {:>12.4} {:>12.2} {:>12.2} {:>12.2} {:>10.0}",
            r.nprobe,
            r.top_k,
            r.num_queries,
            r.mean_recall,
            r.p50_latency_ms,
            r.p95_latency_ms,
            r.p99_latency_ms,
            r.qps
        );
    }

    println!("{}", "=".repeat(80));

    Ok(())
}

/// Load all vectors from a collection's segments.
fn load_all_vectors(
    coll: &puffer_storage::Collection,
    _catalog: &Arc<Catalog>,
) -> anyhow::Result<Vec<SampledVector>> {
    let mut all_vectors = Vec::new();

    // Load vectors from segments
    for seg_name in coll.meta.get_segment_names() {
        let path = coll.segment_path(&seg_name);
        let segment = LoadedSegment::open(&path)?;

        for i in 0..segment.num_vectors() {
            let vector = segment.get_vector(i).to_vec();
            let id = segment.get_id(i)?;
            all_vectors.push(SampledVector { id, vector });
        }
    }

    // Also include staging vectors
    for i in 0..coll.staging_vectors.len() {
        all_vectors.push(SampledVector {
            id: coll.staging_ids[i].clone(),
            vector: coll.staging_vectors[i].clone(),
        });
    }

    Ok(all_vectors)
}

/// Compute brute-force ground truth for all queries.
fn compute_ground_truth(
    queries: &[&SampledVector],
    dataset: &[SampledVector],
    metric: Metric,
    top_k: usize,
) -> Vec<Vec<VectorId>> {
    queries
        .par_iter()
        .map(|query| {
            // Compute distances to all vectors
            let mut distances: Vec<(usize, f32)> = dataset
                .iter()
                .enumerate()
                .filter(|(_, v)| v.id != query.id) // Exclude query itself
                .map(|(i, v)| {
                    let dist = distance::distance(&query.vector, &v.vector, metric);
                    (i, dist)
                })
                .collect();

            // Sort by distance (ascending)
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            // Take top-K IDs
            distances
                .into_iter()
                .take(top_k)
                .map(|(i, _)| dataset[i].id.clone())
                .collect()
        })
        .collect()
}

/// Run ANN benchmark and compute metrics.
fn run_ann_benchmark(
    engine: &QueryEngine,
    collection_name: &str,
    queries: &[&SampledVector],
    ground_truth: &[Vec<VectorId>],
    top_k: usize,
    nprobe: usize,
    router_top_m: Option<usize>,
) -> anyhow::Result<BenchmarkResult> {
    let mut latencies: Vec<Duration> = Vec::with_capacity(queries.len());
    let mut recalls: Vec<f64> = Vec::with_capacity(queries.len());

    let total_start = Instant::now();

    for (i, query) in queries.iter().enumerate() {
        let start = Instant::now();

        let ann_results = engine.search_with_options(
            collection_name,
            &query.vector,
            top_k,
            nprobe,
            false, // don't include payload
            router_top_m,
        )?;

        let elapsed = start.elapsed();
        latencies.push(elapsed);

        // Compute recall for this query
        let ann_ids: HashSet<String> = ann_results
            .iter()
            .map(|r| r.id.to_string())
            .collect();

        let gt_ids: HashSet<String> = ground_truth[i]
            .iter()
            .map(|id| id.to_string())
            .collect();

        let intersection = ann_ids.intersection(&gt_ids).count();
        let recall = intersection as f64 / top_k as f64;
        recalls.push(recall);
    }

    let total_elapsed = total_start.elapsed();

    // Compute statistics
    latencies.sort();

    let num_queries = queries.len();
    let mean_recall = recalls.iter().sum::<f64>() / num_queries as f64;

    let p50_idx = num_queries / 2;
    let p95_idx = (num_queries as f64 * 0.95) as usize;
    let p99_idx = (num_queries as f64 * 0.99) as usize;

    let p50_latency_ms = latencies[p50_idx].as_secs_f64() * 1000.0;
    let p95_latency_ms = latencies[p95_idx.min(num_queries - 1)].as_secs_f64() * 1000.0;
    let p99_latency_ms = latencies[p99_idx.min(num_queries - 1)].as_secs_f64() * 1000.0;

    let total_latency: Duration = latencies.iter().sum();
    let mean_latency_ms = total_latency.as_secs_f64() * 1000.0 / num_queries as f64;

    let qps = num_queries as f64 / total_elapsed.as_secs_f64();

    Ok(BenchmarkResult {
        nprobe,
        top_k,
        num_queries,
        mean_recall,
        p50_latency_ms,
        p95_latency_ms,
        p99_latency_ms,
        mean_latency_ms,
        qps,
    })
}

/// Write results to CSV file.
fn write_csv(path: &PathBuf, results: &[BenchmarkResult]) -> anyhow::Result<()> {
    let mut file = File::create(path)?;

    // Write header
    writeln!(
        file,
        "nprobe,top_k,num_queries,mean_recall,p50_ms,p95_ms,p99_ms,mean_ms,qps"
    )?;

    // Write data rows
    for r in results {
        writeln!(
            file,
            "{},{},{},{:.6},{:.3},{:.3},{:.3},{:.3},{:.1}",
            r.nprobe,
            r.top_k,
            r.num_queries,
            r.mean_recall,
            r.p50_latency_ms,
            r.p95_latency_ms,
            r.p99_latency_ms,
            r.mean_latency_ms,
            r.qps
        )?;
    }

    Ok(())
}
