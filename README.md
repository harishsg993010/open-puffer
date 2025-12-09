# Open Puffer - High-Performance Vector Database

A single-node, NVMe-backed, segment-based vector database implemented in Rust with advanced retrieval capabilities.

## Features

### Core Vector Search
- **IVF-Flat indexing** with k-means++ clustering
- **HNSW graph index** for high-recall approximate nearest neighbor search
- **IVF-HNSW hybrid** combining coarse quantization with graph search
- **Product Quantization (PQ)** with OPQ rotation for memory-efficient storage
- **Immutable segment files** with memory-mapped I/O
- **L2 and Cosine distance** metrics
- **SIMD-optimized** distance calculations

### Full-Text Search
- **BM25 ranking** powered by Tantivy
- **Inverted index** with configurable tokenization
- **Phrase search** and term queries
- **Fuzzy matching** for typo tolerance

### Hybrid Search
- **Vector + BM25 fusion** for semantic + keyword search
- **Multiple fusion methods**:
  - Weighted Sum
  - Reciprocal Rank Fusion (RRF)
  - Normalized Weighted Sum
  - Softmax Fusion
- **Configurable lambda** for text vs vector weighting

### Additional Features
- **HTTP JSON API** with axum
- **Payload support** for metadata storage
- **Automatic compaction** with tiered storage (L0/L1 segments)
- **Multi-segment prefetching** with mmap warming
- **Async refinement** for long-tail queries

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

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/healthz` | Health check |
| POST | `/v1/collections` | Create collection |
| GET | `/v1/collections` | List collections |
| DELETE | `/v1/collections/{name}` | Delete collection |
| GET | `/v1/collections/{name}/stats` | Get statistics |
| POST | `/v1/collections/{name}/points` | Insert points |
| POST | `/v1/collections/{name}/search` | Vector search |
| POST | `/v1/collections/{name}/flush` | Flush staging buffer |
| POST | `/v1/collections/{name}/compact` | Trigger compaction |
| POST | `/v1/collections/{name}/rebuild-router` | Rebuild router index |

### Full-Text Search Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/v1/collections/{name}/text-documents` | Add text documents |
| POST | `/v1/collections/{name}/text-search` | BM25 text search |
| POST | `/v1/collections/{name}/hybrid-search` | Hybrid vector + text search |

## API Examples

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
      {"id": "vec1", "vector": [0.1, 0.2, ...], "payload": {"title": "Document 1"}}
    ]
  }'
```

### Vector Search

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

### Add Text Documents

```bash
curl -X POST http://localhost:8080/v1/collections/my_collection/text-documents \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {
        "vector_id": "vec1",
        "text": "Machine learning enables computers to learn from data",
        "title": "ML Introduction",
        "tags": ["ml", "ai"]
      }
    ]
  }'
```

### BM25 Text Search

```bash
curl -X POST http://localhost:8080/v1/collections/my_collection/text-search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning",
    "top_k": 10,
    "include_text": true
  }'
```

### Hybrid Search

```bash
curl -X POST http://localhost:8080/v1/collections/my_collection/hybrid-search \
  -H "Content-Type: application/json" \
  -d '{
    "text_query": "machine learning",
    "vector": [0.1, 0.2, ...],
    "top_k": 10,
    "lambda": 0.5,
    "fusion_method": "rrf",
    "candidates_per_source": 100
  }'
```

**Fusion Methods:**
- `weighted` - Linear combination of scores
- `rrf` - Reciprocal Rank Fusion (recommended)
- `normalized` - Normalized weighted sum
- `softmax` - Softmax-based fusion

**Lambda Parameter:**
- `0.0` = Vector search only
- `0.5` = Equal weight (default)
- `1.0` = Text search only

## Project Structure

```
puffer-mvp/
  crates/
    core/        # Vector math, SIMD, metrics
    storage/     # Segment format + disk I/O
    index/       # K-means clustering, IVF-Flat, IVF-HNSW
    query/       # Query planner + ANN execution
    api/         # HTTP server (axum)
    cli/         # CLI tools
    bench/       # Criterion benchmarks
    pq/          # Product Quantization + OPQ
    hnsw/        # HNSW graph index
    fts/         # Full-text search (Tantivy)
```

## Performance

### Vector Search at Scale (1M Vectors)

128 dimensions, L2 metric, IVF-Flat index.

| Dataset Size | P50 Latency | P95 Latency | QPS | Ingestion |
|-------------|-------------|-------------|-----|-----------|
| 100K | 0.54ms | 1.62ms | 261 | 12,316 vec/s |
| 500K | 1.08ms | 2.67ms | 223 | 8,943 vec/s |
| **1M** | **1.45ms** | **3.05ms** | **207** | 8,467 vec/s |

### BM25 & Hybrid Search (20K Documents)

| Method | P50 (ms) | P95 (ms) | QPS | Recall@10 | MRR |
|--------|----------|----------|-----|-----------|-----|
| **Vector Search** | 3.03 | 4.92 | 301 | 1.000 | 0.833 |
| **BM25 Search** | 4.05 | 6.99 | 221 | 0.180 | 0.047 |
| **Hybrid (weighted)** | 5.85 | 8.05 | 173 | 0.960 | 0.660 |
| **Hybrid (RRF)** | 6.00 | 8.80 | 161 | **1.000** | **0.751** |
| **Hybrid (normalized)** | 5.00 | 6.65 | 200 | 0.960 | 0.719 |

### Hybrid Search Lambda Tuning

| Lambda (text weight) | Recall@10 | MRR | Notes |
|---------------------|-----------|-----|-------|
| 0.00 (vector only) | 1.000 | 0.797 | Best for embedding queries |
| 0.25 | 1.000 | 0.735 | Good balance |
| 0.50 | 1.000 | 0.707 | Equal weight |
| 0.75 | 0.960 | 0.601 | Text-heavy |
| 1.00 (text only) | 0.140 | 0.048 | Best for keyword queries |

### Recall@10 (GloVe-100 Dataset)

10,000 vectors, 100 dimensions, cosine metric.

| nprobe | Recall@10 | P50 Latency | QPS |
|--------|-----------|-------------|-----|
| 4 | 69.4% | 0.022ms | 20,444 |
| 8 | 79.5% | 0.033ms | 17,661 |
| 16 | 89.5% | 0.053ms | 12,401 |
| 32 | 96.8% | 0.092ms | 8,464 |
| 64 | 99.7% | 0.155ms | 5,388 |

### Recommended Settings

| Use Case | nprobe | Expected Recall | P50 Latency |
|----------|--------|-----------------|-------------|
| Low latency | 4 | ~70% | <0.03ms |
| Balanced | 16 | ~90% | <0.06ms |
| High accuracy | 32 | ~97% | <0.1ms |
| Maximum recall | 64+ | ~99%+ | <0.2ms |

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

### Run All Tests

```bash
cargo test
```

### Criterion Benchmarks

```bash
cargo bench
```

### Hybrid Search Benchmark

```bash
python scripts/benchmark_hybrid.py 10000 128 100
```

Arguments: `<num_documents> <dimension> <num_queries>`

### Recall Benchmark

```bash
./target/release/recall-bench --collection my_collection --data-dir ./data \
    --num-queries 100 --top-k 10 --nprobe 4,8,12,16 --output recall_results.csv
```

### Real Embeddings Benchmark

```bash
./target/release/real-recall-bench \
    --embeddings-file data/embeddings.npy \
    --collection-name embeddings \
    --metric cosine \
    --top-k 10 \
    --nprobe 4,8,16,32 \
    --output recall_real.csv
```

## Architecture

### Index Types

1. **IVF-Flat**: Inverted file index with flat (exact) search within clusters
2. **HNSW**: Hierarchical Navigable Small World graph for high-recall ANN
3. **IVF-HNSW**: Hybrid combining IVF partitioning with HNSW intra-cluster search
4. **PQ/OPQ**: Product Quantization for memory-efficient vector compression

### Storage Tiers

- **L0 Segments**: Small, recently flushed segments (auto-compacted)
- **L1 Segments**: Larger, compacted segments with optimized clustering
- **Router Index**: Coarse quantizer for segment selection

### Full-Text Search

- Powered by [Tantivy](https://github.com/quickwit-oss/tantivy)
- BM25 scoring with configurable k1 and b parameters
- Supports title, text, tags, and metadata fields
- Automatic tokenization with stemming

## License

MIT
