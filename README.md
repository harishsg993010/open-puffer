# open-puffer  High-Performance Local Vector Database

A single-node, NVMe-backed, segment-based vector database implemented in Rust.

## Features

- **IVF-Flat indexing** with k-means++ clustering
- **Immutable segment files** with memory-mapped I/O
- **HTTP JSON API** with axum
- **L2 and Cosine distance** metrics
- **Payload support** for metadata storage
- **SIMD-optimized** distance calculations

## Quick Start

### Build

```bash
cargo build --release
```

### Run the Server

```bash
./target/release/puffer-server --data-dir ./data --bind-addr 0.0.0.0:8080
```

### CLI Usage

Create a collection:
```bash
./target/release/puffer-cli create-collection --name my_vectors --dimension 768 --metric cosine
```

Load random vectors:
```bash
./target/release/puffer-cli load-random --collection my_vectors --count 100000 --dimension 768
```

Search:
```bash
./target/release/puffer-cli search-random --collection my_vectors --dimension 768 --num-queries 100
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/healthz` | Health check |
| POST | `/v1/collections` | Create collection |
| GET | `/v1/collections` | List collections |
| DELETE | `/v1/collections/{name}` | Delete collection |
| GET | `/v1/collections/{name}/stats` | Get statistics |
| POST | `/v1/collections/{name}/points` | Insert points |
| POST | `/v1/collections/{name}/search` | Search vectors |
| POST | `/v1/collections/{name}/flush` | Flush staging buffer |

### Create Collection

```bash
curl -X POST http://localhost:8080/v1/collections \
  -H "Content-Type: application/json" \
  -d '{"name": "my_collection", "dimension": 768, "metric": "cosine"}'
```

### Insert Points

```bash
curl -X POST http://localhost:8080/v1/collections/my_collection/points \
  -H "Content-Type: application/json" \
  -d '{
    "points": [
      {"id": "vec1", "vector": [0.1, 0.2, ...], "payload": {"key": "value"}}
    ]
  }'
```

### Search

```bash
curl -X POST http://localhost:8080/v1/collections/my_collection/search \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.1, 0.2, ...],
    "top_k": 10,
    "nprobe": 4,
    "include_payload": true
  }'
```

## Project Structure

```
puffer-mvp/
  crates/
    core/        # Vector math, SIMD, metrics
    storage/     # Segment format + disk I/O
    index/       # K-means clustering, IVF-Flat
    query/       # Query planner + ANN execution
    api/         # HTTP server (axum)
    cli/         # CLI tools
    bench/       # Criterion benchmarks
```

## Segment File Format

Binary layout for immutable segment files:

```
HEADER (64 bytes):
  magic: "PRSEG1\0" (7 bytes)
  version: u32
  dimension: u32
  metric: u8
  num_vectors: u32
  num_clusters: u32
  offsets (5 x u64)

CLUSTER METADATA:
  centroid[dim], start_index, length

VECTOR DATA:
  Contiguous f32 values in cluster order

ID TABLE:
  Length-prefixed strings

PAYLOAD OFFSET TABLE:
  (offset: u64, length: u32) per vector

PAYLOAD BLOB:
  Concatenated JSON payloads
```

## Benchmarks

### Criterion Benchmarks
```bash
cargo bench
```

### Recall Benchmark

Measure recall@K of ANN search vs brute-force ground truth:

```bash
# Run recall benchmark on a collection
./target/release/recall-bench --collection my_collection --data-dir ./data \
    --num-queries 100 --top-k 10 --nprobe 4,8,12,16 --output recall_results.csv

# Generate plots (requires Python with matplotlib and pandas)
python scripts/plot_recall_latency.py recall_results.csv --output-dir ./plots
```

Options:
- `--collection`: Name of the collection to benchmark (required)
- `--data-dir`: Path to data directory (default: `./data`)
- `--num-queries`: Number of query vectors (default: 100)
- `--top-k`: K in recall@K (default: 10)
- `--nprobe`: Comma-separated nprobe values to test (default: `4,8,12,16`)
- `--sample-size`: Number of vectors to sample for evaluation (default: 50000)
- `--output`: Output CSV file path (default: `recall_results.csv`)
- `--seed`: Random seed for reproducibility (default: 42)
- `--router-top-m`: Number of segments to search via router (0 = use default)

Output files:
- `recall_results.csv`: Raw metrics data
- `recall_vs_nprobe.png`: Recall@K vs nprobe
- `latency_vs_nprobe.png`: Latency percentiles vs nprobe
- `recall_latency_tradeoff.png`: Recall vs latency Pareto curve
- `qps_vs_nprobe.png`: Throughput vs nprobe

### Real Embeddings Recall Benchmark

Benchmark with real embeddings from NumPy files (e.g., sentence-transformers):

```bash
# Run benchmark with real embeddings
./target/release/real-recall-bench \
    --embeddings-file data/embeddings.npy \
    --ids-file data/ids.txt \
    --collection-name st_embeddings \
    --metric cosine \
    --top-k 10 \
    --num-queries 500 \
    --nprobe 4,8,16,32 \
    --sample-size 100000 \
    --output recall_real.csv

# Generate plots
python scripts/plot_real_recall.py recall_real.csv --output-dir ./plots
```

**Input format:**
- `embeddings.npy`: NumPy array with shape `[N, dim]`, dtype `float32`
- `ids.txt` (optional): One ID per line. If omitted, auto-generates `vec_0`, `vec_1`, ...

**Options:**
- `--embeddings-file`: Path to NumPy .npy file (required)
- `--ids-file`: Path to IDs text file (optional)
- `--collection-name`: Name of collection to create/use (required)
- `--data-dir`: Path to data directory (default: `./data`)
- `--metric`: Distance metric: `cosine` or `l2` (default: `cosine`)
- `--top-k`: K in recall@K (default: 10)
- `--num-queries`: Number of queries (default: 500)
- `--nprobe`: Comma-separated nprobe values (default: `4,8,16,32`)
- `--sample-size`: Number of embeddings to sample, 0 = all (default: 0)
- `--output`: Output CSV path (default: `recall_real.csv`)
- `--rebuild-collection`: Delete and recreate collection if exists
- `--staging-threshold`: Vectors per segment (default: 10000)
- `--batch-size`: Insert batch size (default: 5000)

**Output files:**
- `recall_real.csv`: Metrics data
- `real_recall_vs_nprobe.png`: Recall vs nprobe
- `real_latency_vs_nprobe.png`: Latency percentiles vs nprobe
- `real_recall_latency_tradeoff.png`: Recall vs latency curve
- `real_qps_vs_nprobe.png`: Throughput vs nprobe

## Testing

```bash
cargo test
```

## License

MIT
