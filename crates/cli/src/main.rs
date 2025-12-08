//! Puffer vector database CLI tools.

use anyhow::Result;
use clap::{Parser, Subcommand};
use rand_distr::{Distribution, Normal};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Parser, Debug)]
#[command(name = "puffer-cli")]
#[command(about = "Puffer vector database CLI tools")]
struct Args {
    /// Server URL
    #[arg(long, default_value = "http://localhost:8080")]
    server: String,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Create a new collection
    CreateCollection {
        /// Collection name
        #[arg(long)]
        name: String,
        /// Vector dimension
        #[arg(long)]
        dimension: usize,
        /// Distance metric (l2 or cosine)
        #[arg(long, default_value = "cosine")]
        metric: String,
    },
    /// List all collections
    ListCollections,
    /// Load random vectors into a collection
    LoadRandom {
        /// Collection name
        #[arg(long)]
        collection: String,
        /// Number of vectors to insert
        #[arg(long)]
        count: usize,
        /// Vector dimension
        #[arg(long)]
        dimension: usize,
        /// Batch size for inserts
        #[arg(long, default_value = "1000")]
        batch_size: usize,
    },
    /// Run random search queries
    SearchRandom {
        /// Collection name
        #[arg(long)]
        collection: String,
        /// Vector dimension
        #[arg(long)]
        dimension: usize,
        /// Number of queries to run
        #[arg(long, default_value = "100")]
        num_queries: usize,
        /// Number of results per query
        #[arg(long, default_value = "10")]
        top_k: usize,
        /// Number of clusters to probe
        #[arg(long, default_value = "4")]
        nprobe: usize,
    },
    /// Get collection statistics
    Stats {
        /// Collection name
        #[arg(long)]
        collection: String,
    },
    /// Flush staging buffer to segment
    Flush {
        /// Collection name
        #[arg(long)]
        collection: String,
    },
    /// Rebuild router index for a collection
    RebuildRouter {
        /// Collection name
        #[arg(long)]
        collection: String,
    },
}

#[derive(Debug, Serialize)]
struct CreateCollectionRequest {
    name: String,
    dimension: usize,
    metric: String,
}

#[derive(Debug, Serialize)]
struct PointInput {
    id: String,
    vector: Vec<f32>,
}

#[derive(Debug, Serialize)]
struct InsertRequest {
    points: Vec<PointInput>,
}

#[derive(Debug, Serialize)]
struct SearchRequest {
    vector: Vec<f32>,
    top_k: usize,
    nprobe: usize,
    include_payload: bool,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct SearchResponse {
    results: Vec<SearchResultItem>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct SearchResultItem {
    id: String,
    distance: f32,
}

#[derive(Debug, Deserialize)]
struct CollectionInfo {
    name: String,
    dimension: usize,
    metric: String,
}

fn generate_random_vector(dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, 1.0).unwrap();
    (0..dim).map(|_| normal.sample(&mut rng) as f32).collect()
}

async fn create_collection(client: &Client, server: &str, name: &str, dimension: usize, metric: &str) -> Result<()> {
    let url = format!("{}/v1/collections", server);
    let req = CreateCollectionRequest {
        name: name.to_string(),
        dimension,
        metric: metric.to_string(),
    };

    let resp = client.post(&url).json(&req).send().await?;

    if resp.status().is_success() {
        println!("Collection '{}' created successfully", name);
    } else {
        let text = resp.text().await?;
        println!("Error creating collection: {}", text);
    }

    Ok(())
}

async fn list_collections(client: &Client, server: &str) -> Result<()> {
    let url = format!("{}/v1/collections", server);
    let resp = client.get(&url).send().await?;

    if resp.status().is_success() {
        let collections: Vec<CollectionInfo> = resp.json().await?;
        if collections.is_empty() {
            println!("No collections found");
        } else {
            println!("{:<20} {:<10} {:<10}", "NAME", "DIMENSION", "METRIC");
            println!("{}", "-".repeat(40));
            for c in collections {
                println!("{:<20} {:<10} {:<10}", c.name, c.dimension, c.metric);
            }
        }
    } else {
        let text = resp.text().await?;
        println!("Error listing collections: {}", text);
    }

    Ok(())
}

async fn load_random(
    client: &Client,
    server: &str,
    collection: &str,
    count: usize,
    dimension: usize,
    batch_size: usize,
) -> Result<()> {
    let url = format!("{}/v1/collections/{}/points", server, collection);

    let start = Instant::now();
    let mut total_inserted = 0;

    for batch_start in (0..count).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(count);
        let batch_count = batch_end - batch_start;

        let points: Vec<PointInput> = (batch_start..batch_end)
            .map(|i| PointInput {
                id: format!("vec_{}", i),
                vector: generate_random_vector(dimension),
            })
            .collect();

        let req = InsertRequest { points };
        let resp = client.post(&url).json(&req).send().await?;

        if !resp.status().is_success() {
            let text = resp.text().await?;
            println!("Error inserting batch: {}", text);
            return Ok(());
        }

        total_inserted += batch_count;
        print!("\rInserted {}/{} vectors...", total_inserted, count);
    }

    let elapsed = start.elapsed();
    println!("\nInserted {} vectors in {:.2?}", total_inserted, elapsed);
    println!(
        "Throughput: {:.0} vectors/sec",
        total_inserted as f64 / elapsed.as_secs_f64()
    );

    Ok(())
}

async fn search_random(
    client: &Client,
    server: &str,
    collection: &str,
    dimension: usize,
    num_queries: usize,
    top_k: usize,
    nprobe: usize,
) -> Result<()> {
    let url = format!("{}/v1/collections/{}/search", server, collection);

    let mut latencies: Vec<Duration> = Vec::with_capacity(num_queries);

    println!("Running {} random queries...", num_queries);

    for i in 0..num_queries {
        let query_vector = generate_random_vector(dimension);
        let req = SearchRequest {
            vector: query_vector,
            top_k,
            nprobe,
            include_payload: false,
        };

        let start = Instant::now();
        let resp = client.post(&url).json(&req).send().await?;
        let elapsed = start.elapsed();

        if resp.status().is_success() {
            let _results: SearchResponse = resp.json().await?;
            latencies.push(elapsed);
        } else {
            let text = resp.text().await?;
            println!("Error on query {}: {}", i, text);
        }

        if (i + 1) % 10 == 0 {
            print!("\rCompleted {}/{} queries...", i + 1, num_queries);
        }
    }

    println!("\n");

    if latencies.is_empty() {
        println!("No successful queries");
        return Ok(());
    }

    // Calculate statistics
    latencies.sort();

    let sum: Duration = latencies.iter().sum();
    let mean = sum / latencies.len() as u32;
    let p50 = latencies[latencies.len() / 2];
    let p95 = latencies[(latencies.len() as f64 * 0.95) as usize];
    let p99 = latencies[(latencies.len() as f64 * 0.99) as usize];
    let min = latencies[0];
    let max = latencies[latencies.len() - 1];

    println!("Search Latency Statistics:");
    println!("  Mean:  {:>10.2?}", mean);
    println!("  Min:   {:>10.2?}", min);
    println!("  Max:   {:>10.2?}", max);
    println!("  P50:   {:>10.2?}", p50);
    println!("  P95:   {:>10.2?}", p95);
    println!("  P99:   {:>10.2?}", p99);
    println!(
        "  QPS:   {:>10.0}",
        num_queries as f64 / sum.as_secs_f64()
    );

    Ok(())
}

async fn get_stats(client: &Client, server: &str, collection: &str) -> Result<()> {
    let url = format!("{}/v1/collections/{}/stats", server, collection);
    let resp = client.get(&url).send().await?;

    if resp.status().is_success() {
        let stats: serde_json::Value = resp.json().await?;
        println!("{}", serde_json::to_string_pretty(&stats)?);
    } else {
        let text = resp.text().await?;
        println!("Error getting stats: {}", text);
    }

    Ok(())
}

async fn flush(client: &Client, server: &str, collection: &str) -> Result<()> {
    let url = format!("{}/v1/collections/{}/flush", server, collection);
    let resp = client.post(&url).send().await?;

    if resp.status().is_success() {
        println!("Collection '{}' flushed successfully", collection);
    } else {
        let text = resp.text().await?;
        println!("Error flushing collection: {}", text);
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "warn".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let args = Args::parse();
    let client = Client::new();

    match args.command {
        Commands::CreateCollection {
            name,
            dimension,
            metric,
        } => {
            create_collection(&client, &args.server, &name, dimension, &metric).await?;
        }
        Commands::ListCollections => {
            list_collections(&client, &args.server).await?;
        }
        Commands::LoadRandom {
            collection,
            count,
            dimension,
            batch_size,
        } => {
            load_random(&client, &args.server, &collection, count, dimension, batch_size).await?;
        }
        Commands::SearchRandom {
            collection,
            dimension,
            num_queries,
            top_k,
            nprobe,
        } => {
            search_random(
                &client,
                &args.server,
                &collection,
                dimension,
                num_queries,
                top_k,
                nprobe,
            )
            .await?;
        }
        Commands::Stats { collection } => {
            get_stats(&client, &args.server, &collection).await?;
        }
        Commands::Flush { collection } => {
            flush(&client, &args.server, &collection).await?;
        }
        Commands::RebuildRouter { collection } => {
            rebuild_router(&client, &args.server, &collection).await?;
        }
    }

    Ok(())
}

async fn rebuild_router(client: &Client, server: &str, collection: &str) -> Result<()> {
    let url = format!("{}/v1/collections/{}/rebuild-router", server, collection);
    let resp = client.post(&url).send().await?;

    if resp.status().is_success() {
        let result: serde_json::Value = resp.json().await?;
        println!("Router rebuilt: {}", serde_json::to_string_pretty(&result)?);
    } else {
        let text = resp.text().await?;
        println!("Error rebuilding router: {}", text);
    }

    Ok(())
}
