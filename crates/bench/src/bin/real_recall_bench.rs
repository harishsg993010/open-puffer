//! Real embeddings recall benchmark tool for Puffer vector database.
//!
//! Loads precomputed embeddings from NumPy .npy files and benchmarks
//! ANN recall vs brute-force ground truth on real embedding data.

use anyhow::{bail, Context, Result};
use clap::Parser;
use ndarray::Array2;
use ndarray_npy::ReadNpyExt;
use puffer_core::{distance, Metric, VectorId, VectorRecord};
use puffer_query::QueryEngine;
use puffer_storage::{Catalog, CollectionConfig};
use rand::prelude::*;
use rand::SeedableRng;
use rayon::prelude::*;
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

/// Real embeddings recall benchmark tool for Puffer vector database.
#[derive(Parser, Debug)]
#[command(name = "real-recall-bench")]
#[command(about = "Benchmark recall@K using real embeddings from NumPy files")]
struct Args {
    /// Path to the NumPy .npy file containing embeddings (shape: [N, dim], dtype: float32).
    #[arg(long)]
    embeddings_file: PathBuf,

    /// Path to the text file containing IDs (one per line). If not provided, auto-generates IDs.
    #[arg(long)]
    ids_file: Option<PathBuf>,

    /// Name of the collection to create/use.
    #[arg(long)]
    collection_name: String,

    /// Path to the data directory.
    #[arg(long, default_value = "./data")]
    data_dir: PathBuf,

    /// Distance metric to use.
    #[arg(long, default_value = "cosine")]
    metric: String,

    /// Number of top results to retrieve (K in recall@K).
    #[arg(long, default_value = "10")]
    top_k: usize,

    /// Number of query vectors to use.
    #[arg(long, default_value = "500")]
    num_queries: usize,

    /// Comma-separated list of nprobe values to test.
    #[arg(long, default_value = "4,8,16,32")]
    nprobe: String,

    /// Number of embeddings to sample (0 = use all).
    #[arg(long, default_value = "0")]
    sample_size: usize,

    /// Output CSV file path.
    #[arg(long, default_value = "recall_real.csv")]
    output: PathBuf,

    /// Random seed for reproducibility.
    #[arg(long, default_value = "42")]
    seed: u64,

    /// Rebuild collection even if it exists (deletes and recreates).
    #[arg(long)]
    rebuild_collection: bool,

    /// Number of segments to search via router (0 = use collection default).
    #[arg(long, default_value = "0")]
    router_top_m: usize,

    /// Staging threshold for segment building.
    #[arg(long, default_value = "10000")]
    staging_threshold: usize,

    /// Batch size for inserting vectors.
    #[arg(long, default_value = "5000")]
    batch_size: usize,
}

/// A loaded embedding with its ID.
#[derive(Clone)]
struct EmbeddingRecord {
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

fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let args = Args::parse();

    // Parse metric
    let metric = match args.metric.to_lowercase().as_str() {
        "cosine" => Metric::Cosine,
        "l2" => Metric::L2,
        _ => bail!("Invalid metric: {}. Use 'cosine' or 'l2'.", args.metric),
    };

    // Parse nprobe values
    let nprobe_values: Vec<usize> = args
        .nprobe
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    if nprobe_values.is_empty() {
        bail!("No valid nprobe values provided");
    }

    tracing::info!("Starting real embeddings recall benchmark");
    tracing::info!("  Embeddings file: {:?}", args.embeddings_file);
    tracing::info!("  IDs file: {:?}", args.ids_file);
    tracing::info!("  Collection: {}", args.collection_name);
    tracing::info!("  Data dir: {:?}", args.data_dir);
    tracing::info!("  Metric: {:?}", metric);
    tracing::info!("  Num queries: {}", args.num_queries);
    tracing::info!("  Top-K: {}", args.top_k);
    tracing::info!("  nprobe values: {:?}", nprobe_values);
    tracing::info!("  Sample size: {}", args.sample_size);
    tracing::info!("  Seed: {}", args.seed);

    // Load embeddings from NumPy file
    tracing::info!("Loading embeddings from {:?}...", args.embeddings_file);
    let embeddings = load_npy_embeddings(&args.embeddings_file)
        .context("Failed to load embeddings file")?;

    let (num_vectors, dim) = (embeddings.nrows(), embeddings.ncols());
    tracing::info!("Loaded {} embeddings with {} dimensions", num_vectors, dim);

    // Load or generate IDs
    let ids = if let Some(ids_path) = &args.ids_file {
        tracing::info!("Loading IDs from {:?}...", ids_path);
        load_ids(ids_path, num_vectors).context("Failed to load IDs file")?
    } else {
        tracing::info!("Auto-generating IDs...");
        (0..num_vectors).map(|i| format!("vec_{}", i)).collect()
    };

    if ids.len() != num_vectors {
        bail!(
            "ID count ({}) doesn't match embedding count ({})",
            ids.len(),
            num_vectors
        );
    }

    // Convert to EmbeddingRecord format
    let mut all_embeddings: Vec<EmbeddingRecord> = (0..num_vectors)
        .map(|i| EmbeddingRecord {
            id: VectorId(ids[i].clone()),
            vector: embeddings.row(i).to_vec(),
        })
        .collect();

    // Sample embeddings if requested
    let mut rng = StdRng::seed_from_u64(args.seed);
    if args.sample_size > 0 && args.sample_size < all_embeddings.len() {
        tracing::info!("Sampling {} embeddings from {}...", args.sample_size, all_embeddings.len());
        all_embeddings.shuffle(&mut rng);
        all_embeddings.truncate(args.sample_size);
    }

    let total_embeddings = all_embeddings.len();
    tracing::info!("Using {} embeddings for benchmark", total_embeddings);

    // Debug: print first few IDs in our sampled dataset
    tracing::debug!("First 10 embedding IDs in sample: {:?}",
        all_embeddings.iter().take(10).map(|e| e.id.to_string()).collect::<Vec<_>>());

    // Handle rebuild collection option
    if args.rebuild_collection {
        let coll_path = args.data_dir.join(&args.collection_name);
        if coll_path.exists() {
            tracing::info!("Removing existing collection directory '{}'...", args.collection_name);
            std::fs::remove_dir_all(&coll_path)?;
        }
    }

    // Open or create catalog
    let catalog = Arc::new(Catalog::open(&args.data_dir)?);
    let engine = QueryEngine::new(catalog.clone());

    // Create collection if it doesn't exist
    let collection_exists = catalog.get_collection(&args.collection_name).is_ok();

    if !collection_exists {
        tracing::info!("Creating collection '{}'...", args.collection_name);
        let config = CollectionConfig {
            name: args.collection_name.clone(),
            dimension: dim,
            metric,
            staging_threshold: args.staging_threshold,
            num_clusters: ((total_embeddings as f64).sqrt() as usize).max(100),
            router_top_m: if args.router_top_m > 0 { args.router_top_m } else { 5 },
            l0_max_segments: 10,
            segment_target_size: 100_000,
        };
        catalog.create_collection(config)?;

        // Insert embeddings in batches
        tracing::info!("Inserting {} embeddings in batches of {}...", total_embeddings, args.batch_size);
        let insert_start = Instant::now();

        for (batch_idx, chunk) in all_embeddings.chunks(args.batch_size).enumerate() {
            let records: Vec<VectorRecord> = chunk
                .iter()
                .map(|e| VectorRecord {
                    id: e.id.clone(),
                    vector: e.vector.clone(),
                    payload: None,
                })
                .collect();

            engine.insert(&args.collection_name, records)?;

            if (batch_idx + 1) % 10 == 0 {
                tracing::info!(
                    "  Inserted {} / {} embeddings",
                    ((batch_idx + 1) * args.batch_size).min(total_embeddings),
                    total_embeddings
                );
            }
        }

        // Force flush any remaining staging buffer
        engine.force_flush(&args.collection_name)?;

        let insert_elapsed = insert_start.elapsed();
        let throughput = total_embeddings as f64 / insert_elapsed.as_secs_f64();
        tracing::info!(
            "Inserted {} embeddings in {:.2}s ({:.0} vec/s)",
            total_embeddings,
            insert_elapsed.as_secs_f64(),
            throughput
        );
    } else {
        tracing::info!("Using existing collection '{}'", args.collection_name);
    }

    // Select query vectors
    let num_queries = args.num_queries.min(all_embeddings.len());
    let query_indices: Vec<usize> = (0..all_embeddings.len())
        .choose_multiple(&mut rng, num_queries);
    let query_embeddings: Vec<&EmbeddingRecord> = query_indices
        .iter()
        .map(|&i| &all_embeddings[i])
        .collect();

    tracing::info!("Selected {} query vectors", query_embeddings.len());

    // Debug: verify collection contents match our embeddings
    if query_embeddings.len() > 0 {
        let sample_query = &query_embeddings[0];
        let debug_result = engine.search_with_options(
            &args.collection_name,
            &sample_query.vector,
            1,
            100, // high nprobe
            false,
            Some(100),
        )?;
        if let Some(result) = debug_result.first() {
            tracing::debug!(
                "Debug: Query {} best match from ANN: {} (dist: {:.6})",
                sample_query.id,
                result.id,
                result.distance
            );
        }

        // Also show total vectors in collection
        let stats = engine.stats(&args.collection_name)?;
        tracing::info!("Collection stats: {:?}", stats);
    }

    // Compute ground truth using brute-force search
    tracing::info!("Computing ground truth (brute-force search)...");
    let gt_start = Instant::now();
    let ground_truth = compute_ground_truth(
        &query_embeddings,
        &all_embeddings,
        metric,
        args.top_k,
    );
    let gt_elapsed = gt_start.elapsed();
    tracing::info!(
        "Ground truth computed for {} queries in {:.2}s",
        ground_truth.len(),
        gt_elapsed.as_secs_f64()
    );

    // Run benchmarks for each nprobe value
    let mut results = Vec::new();

    for &nprobe in &nprobe_values {
        tracing::info!("Benchmarking with nprobe={}", nprobe);

        let result = run_ann_benchmark(
            &engine,
            &args.collection_name,
            &query_embeddings,
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
    println!("REAL EMBEDDINGS RECALL BENCHMARK RESULTS");
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

/// Load embeddings from a NumPy .npy file.
fn load_npy_embeddings(path: &PathBuf) -> Result<Array2<f32>> {
    let file = File::open(path)
        .with_context(|| format!("Failed to open embeddings file: {:?}", path))?;

    let reader = BufReader::new(file);
    let arr: Array2<f32> = Array2::read_npy(reader)
        .with_context(|| format!("Failed to parse NumPy file: {:?}", path))?;

    Ok(arr)
}

/// Load IDs from a text file (one ID per line).
fn load_ids(path: &PathBuf, expected_count: usize) -> Result<Vec<String>> {
    let file = File::open(path)
        .with_context(|| format!("Failed to open IDs file: {:?}", path))?;

    let reader = BufReader::new(file);
    let ids: Vec<String> = reader
        .lines()
        .collect::<std::io::Result<Vec<_>>>()
        .with_context(|| format!("Failed to read IDs file: {:?}", path))?;

    if ids.len() != expected_count {
        bail!(
            "IDs file has {} lines but expected {} (matching embeddings count)",
            ids.len(),
            expected_count
        );
    }

    Ok(ids)
}

/// Ground truth result with distances for debugging.
struct GroundTruthResult {
    ids: Vec<VectorId>,
}

/// Compute brute-force ground truth for all queries.
fn compute_ground_truth(
    queries: &[&EmbeddingRecord],
    dataset: &[EmbeddingRecord],
    metric: Metric,
    top_k: usize,
) -> Vec<GroundTruthResult> {
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

            // Take top-K
            let top_results: Vec<_> = distances.into_iter().take(top_k).collect();

            GroundTruthResult {
                ids: top_results.iter().map(|(i, _)| dataset[*i].id.clone()).collect(),
            }
        })
        .collect()
}

/// Run ANN benchmark and compute metrics.
fn run_ann_benchmark(
    engine: &QueryEngine,
    collection_name: &str,
    queries: &[&EmbeddingRecord],
    ground_truth: &[GroundTruthResult],
    top_k: usize,
    nprobe: usize,
    router_top_m: Option<usize>,
) -> Result<BenchmarkResult> {
    let mut latencies: Vec<Duration> = Vec::with_capacity(queries.len());
    let mut recalls: Vec<f64> = Vec::with_capacity(queries.len());

    let total_start = Instant::now();

    for (i, query) in queries.iter().enumerate() {
        let start = Instant::now();

        // Request top_k + 1 because ANN will include the query itself as the first result
        let ann_results = engine.search_with_options(
            collection_name,
            &query.vector,
            top_k + 1, // +1 to account for query itself
            nprobe,
            false, // don't include payload
            router_top_m,
        )?;

        let elapsed = start.elapsed();
        latencies.push(elapsed);

        // Filter out the query itself from results (it will be at distance ~0)
        // and take only top_k results
        let ann_results_filtered: Vec<_> = ann_results
            .iter()
            .filter(|r| r.id.to_string() != query.id.to_string())
            .take(top_k)
            .collect();

        // Compute recall for this query
        let ann_ids: HashSet<String> = ann_results_filtered
            .iter()
            .map(|r| r.id.to_string())
            .collect();

        let gt_ids: HashSet<String> = ground_truth[i].ids
            .iter()
            .map(|id| id.to_string())
            .collect();

        let intersection = ann_ids.intersection(&gt_ids).count();
        let recall = intersection as f64 / top_k as f64;
        recalls.push(recall);

        // Debug output for first few queries
        if i < 3 {
            tracing::debug!(
                "Query {}: recall={:.2}, GT={:?}, ANN={:?}",
                i,
                recall,
                ground_truth[i].ids.iter().take(3).map(|id| id.to_string()).collect::<Vec<_>>(),
                ann_results_filtered.iter().take(3).map(|r| r.id.to_string()).collect::<Vec<_>>()
            );
        }
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
fn write_csv(path: &PathBuf, results: &[BenchmarkResult]) -> Result<()> {
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
