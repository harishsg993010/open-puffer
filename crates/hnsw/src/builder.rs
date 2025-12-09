//! HNSW index construction.

use crate::config::HnswConfig;
use crate::error::{HnswError, HnswResult};
use crate::graph::{HnswGraph, HnswNode};
use crate::search::{select_neighbors_heuristic, select_neighbors_simple};
use crate::HnswIndex;
use puffer_core::distance::distance;
use puffer_core::Metric;
use rand::prelude::*;
use rayon::prelude::*;
use std::collections::{BinaryHeap, HashSet};
use std::cmp::Ordering;

/// Builder for constructing HNSW indices.
pub struct HnswBuilder {
    /// Configuration.
    config: HnswConfig,

    /// Vector dimension.
    dim: usize,

    /// Distance metric.
    metric: Metric,

    /// Vectors being indexed.
    vectors: Vec<Vec<f32>>,

    /// Graph being built.
    graph: HnswGraph,

    /// Entry point.
    entry_point: Option<usize>,

    /// Maximum layer.
    max_layer: usize,

    /// Random number generator.
    rng: StdRng,

    /// Use heuristic neighbor selection.
    use_heuristic: bool,
}

impl HnswBuilder {
    /// Create a new builder.
    pub fn new(dim: usize, metric: Metric, config: HnswConfig) -> Self {
        let rng = match config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        Self {
            config,
            dim,
            metric,
            vectors: Vec::new(),
            graph: HnswGraph::new(),
            entry_point: None,
            max_layer: 0,
            rng,
            use_heuristic: true,
        }
    }

    /// Use simple neighbor selection instead of heuristic.
    pub fn simple_selection(mut self) -> Self {
        self.use_heuristic = false;
        self
    }

    /// Add a vector to the index.
    pub fn add(&mut self, id: usize, vector: &[f32]) -> HnswResult<()> {
        if vector.len() != self.dim {
            return Err(HnswError::DimensionMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }

        // Determine layer for this node
        let layer = self.random_layer();

        // Create node
        let node = HnswNode::new(id, layer);

        // Add vector
        while self.vectors.len() <= id {
            self.vectors.push(Vec::new());
        }
        self.vectors[id] = vector.to_vec();

        // Add node to graph
        let node_id = self.graph.add_node(node);
        assert_eq!(node_id, id);

        if self.entry_point.is_none() {
            // First node
            self.entry_point = Some(id);
            self.max_layer = layer;
            return Ok(());
        }

        let entry_point = self.entry_point.unwrap();

        // Find entry point for insertion
        let mut curr_node = entry_point;
        let mut curr_dist = self.compute_distance(vector, curr_node);

        // Search from top layer to layer + 1
        for lc in (layer + 1..=self.max_layer).rev() {
            let (new_node, new_dist) = self.greedy_search(vector, curr_node, curr_dist, lc, 1);
            curr_node = new_node;
            curr_dist = new_dist;
        }

        // Insert at layers 0..=layer
        for lc in (0..=layer.min(self.max_layer)).rev() {
            // Find neighbors
            let candidates = self.search_layer(vector, curr_node, self.config.ef_construction, lc);

            // Select M neighbors
            let m = self.config.max_connections(lc);
            let neighbors = if self.use_heuristic {
                let cands: Vec<(usize, f32)> = candidates.iter().map(|c| (c.0, c.1)).collect();
                select_neighbors_heuristic(&cands, m, true, self.metric, &self.vectors)
            } else {
                let cands: Vec<(usize, f32)> = candidates.iter().map(|c| (c.0, c.1)).collect();
                select_neighbors_simple(&cands, m)
            };

            // Set neighbors for new node
            self.graph.set_neighbors(id, lc, neighbors.clone());

            // Add reverse connections and prune if needed
            for &neighbor in &neighbors {
                let neighbor = neighbor as usize;
                self.graph.get_node_mut(neighbor).unwrap().add_neighbor(lc, id as u32);

                // Prune if over capacity
                let max_conn = self.config.max_connections(lc);
                let curr_neighbors = self.graph.get_neighbors(neighbor, lc);
                if curr_neighbors.len() > max_conn {
                    let neighbor_vec = &self.vectors[neighbor];
                    let mut candidates: Vec<(usize, f32)> = curr_neighbors
                        .iter()
                        .map(|&n| {
                            let n = n as usize;
                            let dist = distance(neighbor_vec, &self.vectors[n], self.metric);
                            (n, dist)
                        })
                        .collect();

                    let pruned = if self.use_heuristic {
                        select_neighbors_heuristic(
                            &candidates,
                            max_conn,
                            true,
                            self.metric,
                            &self.vectors,
                        )
                    } else {
                        select_neighbors_simple(&candidates, max_conn)
                    };

                    self.graph.set_neighbors(neighbor, lc, pruned);
                }
            }

            if !candidates.is_empty() {
                curr_node = candidates[0].0;
                curr_dist = candidates[0].1;
            }
        }

        // Update entry point if new node has higher layer
        if layer > self.max_layer {
            self.entry_point = Some(id);
            self.max_layer = layer;
        }

        Ok(())
    }

    /// Build from a batch of vectors.
    pub fn build_batch(mut self, vectors: &[Vec<f32>]) -> HnswResult<HnswIndex> {
        for (id, vec) in vectors.iter().enumerate() {
            self.add(id, vec)?;
        }
        Ok(self.build())
    }

    /// Finalize and return the index.
    pub fn build(self) -> HnswIndex {
        let num_vectors = self.graph.len();
        HnswIndex {
            config: self.config,
            graph: self.graph,
            dim: self.dim,
            metric: self.metric,
            vectors: Some(self.vectors),
            entry_point: self.entry_point,
            max_layer: self.max_layer,
            num_vectors,
        }
    }

    /// Build without storing vectors.
    pub fn build_without_vectors(self) -> HnswIndex {
        let num_vectors = self.graph.len();
        HnswIndex {
            config: self.config,
            graph: self.graph,
            dim: self.dim,
            metric: self.metric,
            vectors: None,
            entry_point: self.entry_point,
            max_layer: self.max_layer,
            num_vectors,
        }
    }

    /// Generate random layer using exponential distribution.
    fn random_layer(&mut self) -> usize {
        let r: f64 = self.rng.gen();
        let layer = (-r.ln() * self.config.ml).floor() as usize;
        layer
    }

    /// Compute distance between query and a stored vector.
    fn compute_distance(&self, query: &[f32], id: usize) -> f32 {
        distance(query, &self.vectors[id], self.metric)
    }

    /// Greedy search returning single best candidate.
    fn greedy_search(
        &self,
        query: &[f32],
        start: usize,
        start_dist: f32,
        layer: usize,
        _ef: usize,
    ) -> (usize, f32) {
        let mut visited = HashSet::new();
        visited.insert(start);

        let mut best_id = start;
        let mut best_dist = start_dist;

        let mut changed = true;
        while changed {
            changed = false;

            for &neighbor in self.graph.get_neighbors(best_id, layer) {
                let neighbor = neighbor as usize;
                if visited.insert(neighbor) {
                    let dist = self.compute_distance(query, neighbor);
                    if dist < best_dist {
                        best_dist = dist;
                        best_id = neighbor;
                        changed = true;
                    }
                }
            }
        }

        (best_id, best_dist)
    }

    /// Search layer returning ef candidates.
    fn search_layer(
        &self,
        query: &[f32],
        start: usize,
        ef: usize,
        layer: usize,
    ) -> Vec<(usize, f32)> {
        let mut visited = HashSet::new();
        visited.insert(start);

        let start_dist = self.compute_distance(query, start);

        // Min-heap for candidates (closest first)
        let mut candidates: BinaryHeap<MinHeapEntry> = BinaryHeap::new();
        candidates.push(MinHeapEntry {
            id: start,
            distance: start_dist,
        });

        // Max-heap for results (furthest first, to evict)
        let mut results: BinaryHeap<MaxHeapEntry> = BinaryHeap::new();
        results.push(MaxHeapEntry {
            id: start,
            distance: start_dist,
        });

        while let Some(curr) = candidates.pop() {
            // Stop if current candidate is worse than worst result
            if let Some(worst) = results.peek() {
                if curr.distance > worst.distance && results.len() >= ef {
                    break;
                }
            }

            for &neighbor in self.graph.get_neighbors(curr.id, layer) {
                let neighbor = neighbor as usize;
                if visited.insert(neighbor) {
                    let dist = self.compute_distance(query, neighbor);

                    let should_add = results.len() < ef || {
                        if let Some(worst) = results.peek() {
                            dist < worst.distance
                        } else {
                            true
                        }
                    };

                    if should_add {
                        candidates.push(MinHeapEntry {
                            id: neighbor,
                            distance: dist,
                        });
                        results.push(MaxHeapEntry {
                            id: neighbor,
                            distance: dist,
                        });

                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }

        let mut result_vec: Vec<(usize, f32)> = results
            .into_iter()
            .map(|e| (e.id, e.distance))
            .collect();
        result_vec.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        result_vec
    }
}

/// Min-heap entry (smallest distance first).
#[derive(Debug)]
struct MinHeapEntry {
    id: usize,
    distance: f32,
}

impl PartialEq for MinHeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for MinHeapEntry {}

impl PartialOrd for MinHeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MinHeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reversed for min-heap
        other.distance
            .partial_cmp(&self.distance)
            .unwrap_or(Ordering::Equal)
    }
}

/// Max-heap entry (largest distance first).
#[derive(Debug)]
struct MaxHeapEntry {
    id: usize,
    distance: f32,
}

impl PartialEq for MaxHeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for MaxHeapEntry {}

impl PartialOrd for MaxHeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MaxHeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Natural order for max-heap
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal)
    }
}

/// Build HNSW index from vectors in parallel batches.
pub fn build_hnsw_parallel(
    vectors: &[Vec<f32>],
    dim: usize,
    metric: Metric,
    config: HnswConfig,
) -> HnswResult<HnswIndex> {
    // For small datasets, use sequential build
    if vectors.len() < 1000 || !config.parallel {
        let builder = HnswBuilder::new(dim, metric, config);
        return builder.build_batch(vectors);
    }

    // For larger datasets, we still use sequential insertion
    // (true parallel HNSW construction requires more complex synchronization)
    let builder = HnswBuilder::new(dim, metric, config);
    builder.build_batch(vectors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::search_hnsw;

    fn generate_random_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect()
    }

    #[test]
    fn test_builder_basic() {
        let vectors = generate_random_vectors(100, 16, 42);
        let config = HnswConfig::new(8)
            .with_ef_construction(50)
            .with_seed(42);

        let mut builder = HnswBuilder::new(16, Metric::L2, config);

        for (id, vec) in vectors.iter().enumerate() {
            builder.add(id, vec).unwrap();
        }

        let index = builder.build();

        assert_eq!(index.num_vectors, 100);
        assert!(index.entry_point.is_some());
    }

    #[test]
    fn test_build_batch() {
        let vectors = generate_random_vectors(200, 32, 42);
        let config = HnswConfig::new(12).with_ef_construction(100);

        let builder = HnswBuilder::new(32, Metric::L2, config);
        let index = builder.build_batch(&vectors).unwrap();

        assert_eq!(index.num_vectors, 200);

        // Test search
        let results = search_hnsw(&index, &vectors[0], 10).unwrap();
        assert_eq!(results.len(), 10);
        assert_eq!(results[0].id, 0);
    }

    #[test]
    fn test_recall() {
        let vectors = generate_random_vectors(500, 32, 42);
        let config = HnswConfig::new(16)
            .with_ef_construction(100)
            .with_ef_search(100);

        let builder = HnswBuilder::new(32, Metric::L2, config);
        let index = builder.build_batch(&vectors).unwrap();

        // Test recall on first 50 vectors
        let k = 10;
        let mut total_recall = 0.0;

        for i in 0..50 {
            let query = &vectors[i];

            // Exact search
            let mut exact: Vec<(usize, f32)> = vectors
                .iter()
                .enumerate()
                .map(|(j, v)| (j, distance(query, v, Metric::L2)))
                .collect();
            exact.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let exact_top_k: Vec<usize> = exact.iter().take(k).map(|x| x.0).collect();

            // HNSW search
            let results = search_hnsw(&index, query, k).unwrap();
            let hnsw_top_k: Vec<usize> = results.iter().map(|r| r.id).collect();

            let hits = hnsw_top_k.iter().filter(|x| exact_top_k.contains(x)).count();
            total_recall += hits as f64 / k as f64;
        }

        let avg_recall = total_recall / 50.0;
        assert!(avg_recall > 0.8, "HNSW recall too low: {:.2}", avg_recall);
    }

    #[test]
    fn test_index_serialization() {
        let vectors = generate_random_vectors(50, 16, 42);
        let config = HnswConfig::new(8).with_ef_construction(50);

        let builder = HnswBuilder::new(16, Metric::L2, config);
        let index = builder.build_batch(&vectors).unwrap();

        let bytes = index.to_bytes();
        let restored = HnswIndex::from_bytes(&bytes).unwrap();

        assert_eq!(index.num_vectors, restored.num_vectors);
        assert_eq!(index.dim, restored.dim);
        assert_eq!(index.max_layer, restored.max_layer);
    }
}
