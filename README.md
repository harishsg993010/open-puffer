# OpenPuffer - High-Performance Local Vector Database

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

Run benchmarks:
```bash
cargo bench
```

## Testing

```bash
cargo test
```

## License

MIT
