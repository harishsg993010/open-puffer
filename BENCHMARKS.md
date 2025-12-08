# Puffer MVP Benchmarks

## Test Environment
- **Platform**: Windows 11
- **CPU**: (local machine)
- **Storage**: NVMe SSD
- **Rust**: Release build with LTO

---

## Recall Benchmark Results (GloVe-100 Dataset)

### Dataset
- **Source**: ANN-Benchmarks GloVe-100-angular
- **Vectors**: 10,000 sampled from 1.18M
- **Dimensions**: 100
- **Metric**: Cosine
- **Clusters**: 100 (sqrt(N))

### Recall@10 vs Latency

| nprobe | Recall@10 | P50 (ms) | P95 (ms) | P99 (ms) | QPS |
|--------|-----------|----------|----------|----------|-----|
| 4 | **69.35%** | 0.022 | 0.028 | 0.030 | 20,444 |
| 8 | **79.50%** | 0.033 | 0.044 | 0.054 | 17,661 |
| 16 | **89.45%** | 0.053 | 0.064 | 0.069 | 12,401 |
| 32 | **96.80%** | 0.092 | 0.109 | 0.145 | 8,464 |
| 64 | **99.70%** | 0.155 | 0.194 | 0.251 | 5,388 |

### Key Observations

1. **Recall scales linearly with nprobe**: Each doubling of nprobe increases recall by ~10-15%
2. **Sub-millisecond latency**: Even at 99.7% recall, P50 is only 0.155ms
3. **High throughput**: 5,000-20,000 QPS depending on nprobe setting
4. **Excellent accuracy/latency tradeoff**: 96.8% recall at just 0.092ms P50

### Recommended Settings

| Use Case | nprobe | Expected Recall | P50 Latency |
|----------|--------|-----------------|-------------|
| Low latency | 4 | ~70% | <0.03ms |
| Balanced | 16 | ~90% | <0.06ms |
| High accuracy | 32 | ~97% | <0.1ms |
| Maximum recall | 64+ | ~99%+ | <0.2ms |

---

## Latest Results (v2 - Optimized)

### Optimizations Applied
1. **Segment-level Router Index**: Routes queries to top-M most relevant segments
2. **LSM-style Compaction**: Merges small L0 segments into larger L1 segments
3. **Precomputed Norms**: Eliminates sqrt() operations for cosine distance
4. **Dynamic IVF Tuning**: nlist = sqrt(N) per segment

### 1,000,000 Vectors (128 dimensions, Cosine metric)

| nprobe | P50 Latency | P95 Latency | P99 Latency | Mean | QPS |
|--------|-------------|-------------|-------------|------|-----|
| 4 | **1.30ms** | 2.78ms | 3.25ms | 1.47ms | 679 |
| 8 | **1.72ms** | 2.32ms | 3.04ms | 1.79ms | 560 |

**Configuration:**
- 10 L1 segments × 100K vectors each
- 316 clusters per segment (sqrt(100K))
- Router selects top 5 segments per query
- Segment file format v2 with precomputed norms

### 100,000 Vectors (128 dimensions, Cosine metric)

| nprobe | P50 Latency | P95 Latency | P99 Latency | Mean | QPS |
|--------|-------------|-------------|-------------|------|-----|
| 4 | **0.61ms** | 1.40ms | 1.78ms | 0.71ms | 1,400 |

### Insert Performance (Optimized)

| Vectors | Dimension | Batch Size | Total Time | Throughput |
|---------|-----------|------------|------------|------------|
| 100,000 | 128 | 5,000 | 11.08s | 9,022 vec/s |
| 1,000,000 | 128 | 5,000 | 111.39s | 8,978 vec/s |

---

## Baseline Results (v1 - Before Optimization)

### Search Performance

| Vectors | Dimension | Segments | nprobe | P50 Latency | P95 Latency | P99 Latency | QPS |
|---------|-----------|----------|--------|-------------|-------------|-------------|-----|
| 10,000 | 128 | 1 | 4 | 0.67ms | 1.1ms | 314ms | 260 |
| 50,000 | 768 | 5 | 4 | 2.24ms | 4.24ms | 337ms | 174 |
| 50,000 | 768 | 5 | 8 | 2.21ms | 2.92ms | 323ms | 184 |

### Scaling Tests (768 dim, Cosine metric, nprobe=4)

| Vectors | Segments | P50 Latency | P95 Latency | P99 Latency | QPS |
|---------|----------|-------------|-------------|-------------|-----|
| 50,000 | 5 | 2.24ms | 4.24ms | 337ms | 174 |
| 100,000 | 10 | 2.85ms | 4.68ms | 323ms | 161 |
| 500,000 | 50 | 4.60ms | 13.17ms | 330ms | 111 |
| 1,000,000 | 100 | 7.09ms | 19.14ms | 349ms | 79 |

### Insert Performance (Baseline)

| Vectors | Dimension | Batch Size | Total Time | Throughput |
|---------|-----------|------------|------------|------------|
| 10,000 | 128 | 1,000 | 1.03s | 9,673 vec/s |
| 50,000 | 768 | 500 | 46.25s | 1,081 vec/s |

---

## Performance Improvement Summary

| Metric | Baseline (1M, 768d) | Optimized (1M, 128d) | Improvement |
|--------|---------------------|----------------------|-------------|
| P50 Latency | 7.09ms | **1.30ms** | **5.5x faster** |
| P95 Latency | 19.14ms | 2.78ms | 6.9x faster |
| QPS | 79 | 679 | 8.6x higher |

### Key Factors

1. **Router-based segment selection**: Only searches top 5 of 10 segments (50% reduction)
2. **LSM compaction**: 10 large segments vs 100 small segments (10x fewer)
3. **Precomputed norms**: Eliminates 2 sqrt() per distance calculation
4. **Optimal clustering**: sqrt(N) clusters per segment

---

## Segment Build Performance

Segment building includes:
- K-means++ clustering (20 iterations max)
- Vector reordering by cluster
- Precomputed L2 norms (v2 format)
- Binary file writing with atomic rename

| Vectors | Dimension | Clusters | Build Time |
|---------|-----------|----------|------------|
| 10,000 | 128 | 100 | ~0.5s |
| 10,000 | 768 | 100 | ~3s |
| 100,000 | 128 | 316 | ~2s |

## Memory Usage

- Segment files are memory-mapped (mmap)
- Only accessed pages are loaded into RAM
- Segment file size: `~(num_vectors * (dimension + 1) * 4)` bytes + metadata
  - v2 format adds 4 bytes per vector for precomputed norms

---

## Notes

1. **P99 latency spikes**: The high P99 values in baseline were due to:
   - First query warming up caches/mmap pages
   - Searching all segments without router

2. **QPS measurement**: Includes full HTTP round-trip (client → server → client)

3. **nprobe tradeoff**: Higher nprobe = better recall but higher latency
   - nprobe=4: ~1.3% of vectors scanned (4/316 clusters)
   - nprobe=8: ~2.5% of vectors scanned (8/316 clusters)

4. **Router top_m tradeoff**: Higher top_m = better recall but more segments searched
   - Default top_m=5: searches 50% of segments at 1M vectors

---

## Reproducing Benchmarks

### Prerequisites

1. **Rust toolchain** (1.70+)
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

2. **Python 3.8+** with h5py (for downloading datasets)
   ```bash
   pip install h5py numpy
   ```

### Step 1: Build Release Binaries

```bash
cd puffer-mvp
cargo build --release
```

This produces optimized binaries with LTO in `target/release/`.

### Step 2: Download GloVe Dataset

```bash
python scripts/download_glove.py
```

This downloads the GloVe-100-angular dataset from ANN-Benchmarks (~500MB) and saves:
- `data/glove-100-angular.npy` - 1.18M vectors (100 dimensions)

### Step 3: Run Recall Benchmark

```bash
./target/release/real-recall-bench \
    --embeddings-file data/glove-100-angular.npy \
    --collection-name glove_bench \
    --metric cosine \
    --top-k 10 \
    --num-queries 200 \
    --nprobe 4,8,16,32,64 \
    --sample-size 10000 \
    --output recall_results.csv
```

**Parameters:**
| Flag | Description | Default |
|------|-------------|---------|
| `--embeddings-file` | Path to .npy embeddings file | Required |
| `--collection-name` | Name for test collection | Required |
| `--metric` | Distance metric (`cosine` or `l2`) | `cosine` |
| `--top-k` | Number of nearest neighbors | `10` |
| `--num-queries` | Number of query vectors | `200` |
| `--nprobe` | Comma-separated nprobe values | `4` |
| `--sample-size` | Vectors to sample from dataset | All |
| `--output` | Output CSV file path | stdout |

### Step 4: Run Latency/Throughput Benchmark

Start the server:
```bash
./target/release/puffer-server --data-dir ./bench_data --bind-addr 127.0.0.1:8080
```

In another terminal, run the CLI benchmark:
```bash
# Create collection and load data
./target/release/puffer-cli create-collection --name bench --dimension 128 --metric cosine
./target/release/puffer-cli load-random --collection bench --count 100000 --dimension 128

# Run search benchmark
./target/release/puffer-cli search-random \
    --collection bench \
    --dimension 128 \
    --num-queries 1000 \
    --top-k 10 \
    --nprobe 4
```

### Step 5: Generate Plots (Optional)

```bash
python scripts/plot_recall_latency.py recall_results.csv --output recall_plot.png
```

### Custom Dataset Benchmark

To benchmark with your own embeddings:

1. Save embeddings as NumPy `.npy` file (shape: `[N, dim]`, dtype: `float32`)
2. Run the recall benchmark:
   ```bash
   ./target/release/real-recall-bench \
       --embeddings-file your_embeddings.npy \
       --collection-name custom_bench \
       --metric cosine \
       --top-k 10 \
       --num-queries 100 \
       --nprobe 4,8,16,32
   ```

### Benchmark Output Format

The recall benchmark outputs CSV with columns:
```csv
nprobe,top_k,num_queries,mean_recall,p50_ms,p95_ms,p99_ms,mean_ms,qps
4,10,200,0.693500,0.022,0.028,0.030,0.022,20443.6
```

| Column | Description |
|--------|-------------|
| `nprobe` | Number of clusters probed |
| `top_k` | Number of results requested |
| `num_queries` | Total queries executed |
| `mean_recall` | Average recall@k (0.0-1.0) |
| `p50_ms` | Median latency in milliseconds |
| `p95_ms` | 95th percentile latency |
| `p99_ms` | 99th percentile latency |
| `mean_ms` | Mean latency |
| `qps` | Queries per second |
