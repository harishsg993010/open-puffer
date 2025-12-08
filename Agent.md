# Puffer Vector Database - Agent Reference

> Comprehensive atomic-level documentation for AI agents working with this codebase.

## 1. Project Overview

**Puffer** is a high-performance, single-node vector database written in Rust implementing:
- **IVF-Flat** (Inverted File with Flat quantization) indexing
- **K-means++** clustering for index construction
- **Memory-mapped** immutable segment files
- **LSM-style** compaction (L0 → L1)
- **REST HTTP API** via Axum

---

## 2. Directory Structure

```
puffer-mvp/
├── Cargo.toml                    # Workspace configuration
├── Cargo.lock                    # Dependency lock file
├── README.md                     # User documentation
├── BENCHMARKS.md                 # Performance benchmarks
├── Agent.md                      # This file
│
├── crates/
│   ├── core/                     # Vector math & types
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── types.rs          # VectorId, VectorRecord
│   │       ├── metric.rs         # Metric enum (L2, Cosine)
│   │       └── distance.rs       # SIMD distance functions
│   │
│   ├── storage/                  # Persistence layer
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── segment.rs        # Segment file format
│   │       ├── catalog.rs        # Collection management
│   │       ├── router.rs         # Segment routing index
│   │       └── error.rs          # StorageError
│   │
│   ├── index/                    # Indexing algorithms
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── kmeans.rs         # K-means++ clustering
│   │       └── search.rs         # IVF-Flat search
│   │
│   ├── query/                    # Query execution
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── engine.rs         # QueryEngine
│   │       ├── compaction.rs     # LSM compaction
│   │       └── error.rs          # QueryError
│   │
│   ├── api/                      # HTTP server
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── main.rs           # puffer-server binary
│   │       ├── routes.rs         # Router definition
│   │       ├── handlers.rs       # Request handlers
│   │       └── state.rs          # AppState
│   │
│   ├── cli/                      # CLI tools
│   │   └── src/
│   │       └── main.rs           # puffer-cli binary
│   │
│   └── bench/                    # Benchmarks
│       └── src/
│           ├── lib.rs
│           └── bin/
│               ├── recall_bench.rs
│               └── real_recall_bench.rs
│
├── scripts/
│   ├── download_glove.py         # Download ANN-benchmarks data
│   ├── plot_recall_latency.py    # Plotting scripts
│   └── plot_real_recall.py
│
└── data/                         # Runtime data (gitignored)
    ├── catalog.json
    └── {collection}/
        ├── meta.json
        ├── router_index.json
        └── {uuid}.seg
```

---

## 3. Crate Dependencies

### 3.1 Internal Dependencies
```
puffer-api     → puffer-query → puffer-index → puffer-core
                             → puffer-storage → puffer-core
puffer-cli     → puffer-core (via reqwest to API)
puffer-bench   → puffer-query, puffer-storage, puffer-core
```

### 3.2 Key External Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| `tokio` | 1.35 | Async runtime |
| `axum` | 0.7 | HTTP framework |
| `serde` / `serde_json` | 1.0 | Serialization |
| `rayon` | 1.8 | Parallel iteration |
| `memmap2` | 0.9 | Memory-mapped files |
| `clap` | 4.4 | CLI parsing |
| `uuid` | 1.6 | Segment naming |
| `tracing` | 0.1 | Structured logging |
| `rand` | 0.8 | Random number generation |
| `ndarray` / `ndarray-npy` | - | NumPy file loading (bench) |

---

## 4. Core Data Types

### 4.1 VectorId (`crates/core/src/types.rs`)
```rust
pub struct VectorId(pub String);

impl VectorId {
    pub fn new(id: impl Into<String>) -> Self;
    pub fn as_str(&self) -> &str;
    pub fn to_bytes(&self) -> Vec<u8>;      // [len: u8][bytes...]
    pub fn from_bytes(data: &[u8]) -> Option<(Self, usize)>;
}
```
- Max length: 255 bytes (length-prefixed)
- Used as primary key for vectors

### 4.2 VectorRecord (`crates/core/src/types.rs`)
```rust
pub struct VectorRecord {
    pub id: VectorId,
    pub vector: Vec<f32>,
    pub payload: Option<serde_json::Value>,
}
```

### 4.3 Metric (`crates/core/src/metric.rs`)
```rust
pub enum Metric {
    L2,      // Euclidean (uses squared distance internally)
    Cosine,  // Angular (1 - cosine_similarity)
}

impl Metric {
    pub fn to_byte(&self) -> u8;        // L2=0, Cosine=1
    pub fn from_byte(b: u8) -> Option<Self>;
}
```

---

## 5. Distance Functions (`crates/core/src/distance.rs`)

### 5.1 Available Functions

| Function | Returns | Notes |
|----------|---------|-------|
| `l2_distance_squared(a, b)` | `f32` | **Fastest** - no sqrt |
| `l2_distance(a, b)` | `f32` | Full Euclidean |
| `dot_product(a, b)` | `f32` | Vector dot product |
| `l2_norm(v)` | `f32` | Vector magnitude |
| `cosine_similarity(a, b)` | `f32` | Range: [-1, 1] |
| `cosine_distance(a, b)` | `f32` | Range: [0, 2] |
| `cosine_distance_with_norms(a, b, norm_a, norm_b)` | `f32` | **Fast** - precomputed norms |
| `distance(a, b, metric)` | `f32` | Generic dispatch |
| `normalize(v)` | `()` | In-place normalization |

### 5.2 SIMD Optimization Pattern
```rust
// 4-way accumulation for pipeline efficiency
let mut sum0 = 0.0f32;
let mut sum1 = 0.0f32;
let mut sum2 = 0.0f32;
let mut sum3 = 0.0f32;

for (a_chunk, b_chunk) in a.chunks_exact(8).zip(b.chunks_exact(8)) {
    sum0 += (a[0]-b[0])² + (a[4]-b[4])²;
    sum1 += (a[1]-b[1])² + (a[5]-b[5])²;
    sum2 += (a[2]-b[2])² + (a[6]-b[6])²;
    sum3 += (a[3]-b[3])² + (a[7]-b[7])²;
}
```

---

## 6. Segment File Format (`crates/storage/src/segment.rs`)

### 6.1 Header (72 bytes, version 2)

| Offset | Size | Field | Type |
|--------|------|-------|------|
| 0 | 7 | magic | `"PRSEG1\0"` |
| 7 | 4 | version | u32 (=2) |
| 11 | 4 | dimension | u32 |
| 15 | 1 | metric | u8 |
| 16 | 4 | num_vectors | u32 |
| 20 | 4 | num_clusters | u32 |
| 24 | 8 | cluster_meta_offset | u64 |
| 32 | 8 | vector_data_offset | u64 |
| 40 | 8 | norms_offset | u64 |
| 48 | 8 | id_table_offset | u64 |
| 56 | 8 | payload_offset_table_offset | u64 |
| 64 | 8 | payload_blob_offset | u64 |

### 6.2 Data Sections (in order)

1. **Cluster Metadata**: Per cluster `[centroid[dim], start_index: u32, length: u32]`
2. **Vector Data**: Contiguous f32 in cluster order
3. **Norms** (v2): Precomputed L2 norms per vector
4. **ID Table**: Length-prefixed strings
5. **Payload Offset Table**: `[(offset: u64, length: u32)]` per vector
6. **Payload Blob**: Concatenated JSON

### 6.3 Key Structures

```rust
pub struct LoadedSegment {
    pub header: SegmentHeader,
    pub clusters: Vec<ClusterMeta>,
    mmap: Mmap,              // Memory-mapped file
    id_offsets: Vec<usize>,  // Cached ID positions
    has_norms: bool,
}

pub struct ClusterMeta {
    pub centroid: Vec<f32>,
    pub start_index: u32,
    pub length: u32,
}
```

### 6.4 Access Methods
```rust
impl LoadedSegment {
    pub fn get_vector(&self, index: usize) -> &[f32];
    pub fn get_id(&self, index: usize) -> StorageResult<VectorId>;
    pub fn get_payload(&self, index: usize) -> StorageResult<Option<Value>>;
    pub fn get_norm(&self, index: usize) -> Option<f32>;  // v2 only
    pub fn has_norms(&self) -> bool;
}
```

---

## 7. Collection Configuration (`crates/storage/src/catalog.rs`)

```rust
pub struct CollectionConfig {
    pub name: String,
    pub dimension: usize,
    pub metric: Metric,              // Default: Cosine
    pub staging_threshold: usize,    // Default: 10,000
    pub num_clusters: usize,         // Default: 100
    pub router_top_m: usize,         // Default: 5
    pub l0_max_segments: usize,      // Default: 10
    pub segment_target_size: usize,  // Default: 100,000
}
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `staging_threshold` | 10,000 | Vectors before auto-flush |
| `num_clusters` | 100 | Baseline K for k-means |
| `router_top_m` | 5 | Segments to search |
| `l0_max_segments` | 10 | Trigger compaction |
| `segment_target_size` | 100,000 | L1 segment size |

---

## 8. K-Means Clustering (`crates/index/src/kmeans.rs`)

### 8.1 Configuration
```rust
pub struct KMeansConfig {
    pub num_clusters: usize,
    pub max_iterations: usize,        // Default: 20
    pub convergence_threshold: f64,   // Default: 0.001
    pub seed: Option<u64>,
}
```

### 8.2 Algorithm
1. **K-means++ initialization**: Sample centroids proportional to squared distance
2. **Assignment**: Parallel assignment to nearest centroid
3. **Update**: Parallel centroid recomputation
4. **Convergence**: Stop when <0.1% vectors change cluster

### 8.3 Functions
```rust
pub fn kmeans(vectors: &[Vec<f32>], config: &KMeansConfig) -> KMeansResult;
pub fn build_cluster_data(centroids, assignments) -> Vec<(Vec<f32>, Vec<usize>)>;
```

---

## 9. IVF-Flat Search (`crates/index/src/search.rs`)

### 9.1 Search Result
```rust
pub struct SearchResult {
    pub id: VectorId,
    pub distance: f32,
    pub payload: Option<serde_json::Value>,
}
```

### 9.2 Search Algorithm
```rust
pub fn search_segment(
    segment: &LoadedSegment,
    query: &[f32],
    k: usize,
    nprobe: usize,
    include_payload: bool,
) -> Vec<SearchResult>;
```

**Steps:**
1. Rank clusters by distance to query
2. Select top `nprobe` clusters
3. Search vectors in those clusters using max-heap
4. Return top `k` results

### 9.3 Heap Implementation
```rust
// Max-heap keeps LARGEST distance at top
// Enables efficient replacement of worst candidate
impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance.partial_cmp(&other.distance)  // Natural ordering
    }
}
```

---

## 10. Query Engine (`crates/query/src/engine.rs`)

### 10.1 Structure
```rust
pub struct QueryEngine {
    catalog: Arc<Catalog>,
    segment_cache: RwLock<HashMap<String, Vec<Arc<LoadedSegment>>>>,
}
```

### 10.2 Key Methods

```rust
// Insert vectors (auto-flush when threshold reached)
pub fn insert(&self, collection: &str, records: Vec<VectorRecord>)
    -> QueryResult<usize>;

// Search with options
pub fn search_with_options(
    &self,
    collection: &str,
    query: &[f32],
    k: usize,
    nprobe: usize,
    include_payload: bool,
    router_top_m: Option<usize>,
) -> QueryResult<Vec<SearchResult>>;

// Force flush staging to segment
pub fn force_flush(&self, collection: &str) -> QueryResult<()>;

// Trigger compaction
pub fn compact(&self, collection: &str) -> QueryResult<CompactionResult>;

// Get statistics
pub fn stats(&self, collection: &str) -> QueryResult<CollectionStats>;

// Rebuild router index
pub fn rebuild_router(&self, collection: &str) -> QueryResult<usize>;
```

### 10.3 Search Flow
1. Search staging buffer (brute-force)
2. Select segments via router (or all if few segments)
3. Load segments (with caching)
4. Parallel search across segments
5. Merge and return top k

---

## 11. LSM Compaction (`crates/query/src/compaction.rs`)

### 11.1 Trigger Condition
```rust
pub fn needs_compaction(coll: &Collection, config: &CompactionConfig) -> bool {
    coll.meta.l0_segment_count() >= config.l0_max_segments
}
```

### 11.2 Compaction Process
1. Select batch of L0 segments
2. Load all vectors from batch
3. Re-cluster with k-means (clusters = sqrt(N))
4. Write new L1 segment
5. Update router (remove L0s, add L1)
6. Delete old segment files

---

## 12. Router Index (`crates/storage/src/router.rs`)

### 12.1 Structure
```rust
pub struct RouterEntry {
    pub segment_id: String,
    pub level: u32,              // 0=L0, 1=L1
    pub segment_centroid: Vec<f32>,
    pub num_vectors: usize,
}

pub struct RouterIndex {
    pub entries: Vec<RouterEntry>,
}
```

### 12.2 Segment Selection
```rust
pub fn select_segments(
    &self,
    query: &[f32],
    metric: Metric,
    top_m: usize,
) -> Vec<String>;
```
- Computes distance from query to each segment centroid
- Returns top M closest segment IDs

---

## 13. HTTP API (`crates/api/`)

### 13.1 Endpoints

| Method | Path | Handler | Purpose |
|--------|------|---------|---------|
| GET | `/healthz` | `health_check` | Health check |
| POST | `/v1/collections` | `create_collection` | Create collection |
| GET | `/v1/collections` | `list_collections` | List collections |
| DELETE | `/v1/collections/:name` | `delete_collection` | Delete collection |
| GET | `/v1/collections/:name/stats` | `get_stats` | Get stats |
| POST | `/v1/collections/:name/points` | `insert_points` | Insert vectors |
| POST | `/v1/collections/:name/search` | `search` | Search vectors |
| POST | `/v1/collections/:name/flush` | `flush_collection` | Flush staging |
| POST | `/v1/collections/:name/rebuild-router` | `rebuild_router` | Rebuild router |
| POST | `/v1/collections/:name/compact` | `compact_collection` | Trigger compaction |

### 13.2 Request/Response Examples

**Create Collection:**
```json
POST /v1/collections
{
    "name": "embeddings",
    "dimension": 768,
    "metric": "cosine"
}
```

**Insert Points:**
```json
POST /v1/collections/embeddings/points
{
    "points": [
        {"id": "doc1", "vector": [0.1, ...], "payload": {"text": "hello"}}
    ]
}
```

**Search:**
```json
POST /v1/collections/embeddings/search
{
    "vector": [0.1, ...],
    "top_k": 10,
    "nprobe": 16,
    "include_payload": true
}
```

---

## 14. CLI Commands (`crates/cli/`)

```bash
# Server management
puffer-server --data-dir ./data --bind-addr 0.0.0.0:8080

# Collection operations
puffer-cli create-collection --name my_coll --dimension 768 --metric cosine
puffer-cli list-collections
puffer-cli stats --collection my_coll
puffer-cli flush --collection my_coll
puffer-cli rebuild-router --collection my_coll

# Data operations
puffer-cli load-random --collection my_coll --count 100000 --dimension 768
puffer-cli search-random --collection my_coll --dimension 768 --num-queries 100
```

---

## 15. Benchmarking (`crates/bench/`)

### 15.1 Recall Benchmark
```bash
./target/release/real-recall-bench \
    --embeddings-file data/embeddings.npy \
    --collection-name test \
    --metric cosine \
    --top-k 10 \
    --num-queries 200 \
    --nprobe 4,8,16,32,64 \
    --sample-size 10000 \
    --output results.csv
```

### 15.2 Output Format
```csv
nprobe,top_k,num_queries,mean_recall,p50_ms,p95_ms,p99_ms,mean_ms,qps
4,10,200,0.693500,0.022,0.028,0.030,0.022,20443.6
```

---

## 16. Error Types

### StorageError (`crates/storage/src/error.rs`)
```rust
pub enum StorageError {
    Io(std::io::Error),
    InvalidMagic,
    UnsupportedVersion(u32),
    InvalidMetric(u8),
    DimensionMismatch { expected, got },
    CollectionNotFound(String),
    CollectionAlreadyExists(String),
    InvalidSegment(String),
    Json(serde_json::Error),
}
```

### QueryError (`crates/query/src/error.rs`)
```rust
pub enum QueryError {
    Storage(StorageError),
    CollectionNotFound(String),
    DimensionMismatch { expected, got },
    InvalidQuery(String),
}
```

---

## 17. Threading Model

### Thread-Safe Components
- **Catalog**: `RwLock<HashMap<String, Arc<RwLock<Collection>>>>`
- **Collection**: `Arc<RwLock<Collection>>`
- **Segment Cache**: `RwLock<HashMap<String, Vec<Arc<LoadedSegment>>>>`

### Parallelism (Rayon)
- K-means assignment and update
- Segment search across multiple segments
- Centroid computation

### Immutability
- Segments are immutable after creation
- Memory-mapped for safe concurrent reads
- No locking during search operations

---

## 18. File Persistence

### Catalog Level
```
data/catalog.json                 # List of collection names
```

### Collection Level
```
data/{collection}/
├── meta.json                     # Config + segment list
├── router_index.json             # Segment routing
├── staging.json                  # Temporary staging buffer
└── {uuid}.seg                    # Immutable segments
```

### Atomic Writes
- All file writes use temp file + rename pattern
- Ensures crash consistency

---

## 19. Performance Characteristics

### Time Complexity

| Operation | Complexity |
|-----------|------------|
| Insert (staging) | O(1) amortized |
| Flush to segment | O(N × K × iter) for k-means |
| Search | O(M × nprobe × cluster_size × dim) |
| Compaction | O(N × K × iter) |

Where:
- N = vectors in segment
- K = number of clusters
- M = segments searched
- iter = k-means iterations

### Space Complexity
- Segment file: `~N × (dim + 1) × 4` bytes
- Memory: Only accessed pages loaded via mmap

---

## 20. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| IVF-Flat over HNSW | Simpler, predictable memory, good for single-node |
| Immutable segments | No write amplification, safe concurrent access |
| K-means++ | Better cluster quality than random init |
| Precomputed norms | Avoids 2 sqrt per cosine distance |
| LSM compaction | Separates write (L0) and read (L1) paths |
| Router index | Avoids scanning all segments |
| Memory-mapped files | Zero-copy, OS handles paging |
| 4-way SIMD accumulation | Better CPU pipeline utilization |

---

## 21. Common Modification Points

### Adding a New Distance Metric
1. Add variant to `Metric` enum in `crates/core/src/metric.rs`
2. Implement distance function in `crates/core/src/distance.rs`
3. Update `distance()` dispatch function
4. Update segment format if metric needs special handling

### Adding a New API Endpoint
1. Add handler in `crates/api/src/handlers.rs`
2. Add route in `crates/api/src/routes.rs`
3. Add request/response types as needed

### Changing Index Parameters
1. Modify `CollectionConfig` in `crates/storage/src/catalog.rs`
2. Update defaults in `CollectionConfig::default()` or API handler
3. Consider backward compatibility for existing collections

### Adding New Segment Data
1. Update `SegmentHeader` in `crates/storage/src/segment.rs`
2. Increment `SEGMENT_VERSION`
3. Update `SegmentBuilder::build()` to write new data
4. Update `LoadedSegment::open()` to read new data
5. Handle backward compatibility for v1 segments
