# Puffer MVP Benchmarks

## Test Environment
- **Platform**: Windows 11
- **CPU**: (local machine)
- **Storage**: NVMe SSD
- **Rust**: Release build with LTO

## Search Performance

### Configuration
- **Metric**: Cosine distance
- **Top-K**: 10 results
- **Clusters per segment**: 100

### Results

| Vectors | Dimension | Segments | nprobe | P50 Latency | P95 Latency | P99 Latency | QPS |
|---------|-----------|----------|--------|-------------|-------------|-------------|-----|
| 10,000 | 128 | 1 | 4 | 0.67ms | 1.1ms | 314ms | 260 |
| 50,000 | 768 | 5 | 4 | 2.24ms | 4.24ms | 337ms | 174 |
| 50,000 | 768 | 5 | 8 | 2.21ms | 2.92ms | 323ms | 184 |

## Insert Performance

| Vectors | Dimension | Batch Size | Total Time | Throughput |
|---------|-----------|------------|------------|------------|
| 10,000 | 128 | 1,000 | 1.03s | 9,673 vec/s |
| 50,000 | 768 | 500 | 46.25s | 1,081 vec/s |

## Segment Build Performance

Segment building includes:
- K-means++ clustering (20 iterations max)
- Vector reordering by cluster
- Binary file writing with atomic rename

| Vectors | Dimension | Clusters | Build Time |
|---------|-----------|----------|------------|
| 10,000 | 128 | 100 | ~0.5s |
| 10,000 | 768 | 100 | ~3s |

## Memory Usage

- Segment files are memory-mapped (mmap)
- Only accessed pages are loaded into RAM
- Segment file size: `~(num_vectors * dimension * 4)` bytes + metadata

## Scaling Tests

### Full Results Table (768 dim, Cosine metric, nprobe=4)

| Vectors | Segments | P50 Latency | P95 Latency | P99 Latency | QPS |
|---------|----------|-------------|-------------|-------------|-----|
| 50,000 | 5 | 2.24ms | 4.24ms | 337ms | 174 |
| 100,000 | 10 | 2.85ms | 4.68ms | 323ms | 161 |
| 500,000 | 50 | 4.60ms | 13.17ms | 330ms | 111 |
| 1,000,000 | 100 | 7.09ms | 19.14ms | 349ms | 79 |

### 1,000,000 Vectors with nprobe=8 (higher recall)

| Vectors | Segments | P50 Latency | P95 Latency | P99 Latency | QPS |
|---------|----------|-------------|-------------|-------------|-----|
| 1,000,000 | 100 | 9.75ms | 12.38ms | 342ms | 76 |

### Observations

1. **Latency scaling**: P50 latency scales sub-linearly with vector count
   - 50k → 1M (20x data): P50 increases 2.24ms → 7.09ms (~3x)

2. **Throughput**: QPS decreases as more segments need to be searched
   - 50k: 174 QPS → 1M: 79 QPS

3. **Segment overhead**: Each segment adds ~0.05ms to search time (parallel search across segments)

4. **Memory efficiency**: 1M vectors × 768 dims × 4 bytes = ~3GB segment data, all memory-mapped

---

## Notes

1. **P99 latency spikes**: The high P99 values are likely due to:
   - First query warming up caches/mmap pages
   - GC pauses in the test client
   - Background segment building

2. **QPS measurement**: Includes full HTTP round-trip (client → server → client)

3. **nprobe tradeoff**: Higher nprobe = better recall but higher latency
   - nprobe=4: ~4% of vectors scanned
   - nprobe=8: ~8% of vectors scanned
