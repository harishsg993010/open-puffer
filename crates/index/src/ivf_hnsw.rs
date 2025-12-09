//! IVF + HNSW hybrid indexing.
//!
//! This module implements a hybrid index that combines:
//! - IVF (Inverted File) for coarse partitioning via k-means clustering
//! - HNSW (Hierarchical Navigable Small World) for intra-cluster search
//!
//! The approach:
//! 1. Partition vectors into clusters using k-means (IVF)
//! 2. Build an HNSW graph within each cluster for fast nearest neighbor search
//! 3. At query time, probe top-k clusters and search their HNSW graphs

use crate::kmeans::{kmeans, KMeansConfig, KMeansResult};
use puffer_core::{distance, Metric, VectorId};
use puffer_hnsw::{HnswBuilder, HnswConfig, HnswIndex, search_hnsw};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Configuration for IVF-HNSW hybrid index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IvfHnswConfig {
    /// Number of clusters (IVF partitions).
    pub num_clusters: usize,

    /// K-means configuration.
    #[serde(default)]
    pub kmeans: KMeansConfigSerde,

    /// HNSW configuration for intra-cluster graphs.
    #[serde(default)]
    pub hnsw: HnswConfigSerde,

    /// Number of clusters to probe during search.
    pub nprobe: usize,

    /// Whether to use HNSW for intra-cluster search.
    /// If false, falls back to brute-force within clusters.
    pub use_hnsw: bool,

    /// Minimum cluster size to build HNSW graph.
    /// Smaller clusters use brute-force search.
    pub min_cluster_size_for_hnsw: usize,
}

impl Default for IvfHnswConfig {
    fn default() -> Self {
        Self {
            num_clusters: 100,
            kmeans: KMeansConfigSerde::default(),
            hnsw: HnswConfigSerde::default(),
            nprobe: 10,
            use_hnsw: true,
            min_cluster_size_for_hnsw: 32,
        }
    }
}

/// Serializable k-means config.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KMeansConfigSerde {
    pub max_iterations: usize,
    pub convergence_threshold: f64,
    pub seed: Option<u64>,
}

impl Default for KMeansConfigSerde {
    fn default() -> Self {
        Self {
            max_iterations: 20,
            convergence_threshold: 0.001,
            seed: None,
        }
    }
}

impl KMeansConfigSerde {
    pub fn to_config(&self, num_clusters: usize) -> KMeansConfig {
        KMeansConfig {
            num_clusters,
            max_iterations: self.max_iterations,
            convergence_threshold: self.convergence_threshold,
            seed: self.seed,
        }
    }
}

/// Serializable HNSW config.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswConfigSerde {
    pub m: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
}

impl Default for HnswConfigSerde {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 100,
            ef_search: 50,
        }
    }
}

impl HnswConfigSerde {
    pub fn to_config(&self) -> HnswConfig {
        HnswConfig::new(self.m)
            .with_ef_construction(self.ef_construction)
            .with_ef_search(self.ef_search)
    }
}

/// A cluster with optional HNSW graph.
pub struct IvfHnswCluster {
    /// Cluster centroid.
    pub centroid: Vec<f32>,

    /// Indices of vectors in this cluster (relative to original vectors array).
    pub vector_indices: Vec<usize>,

    /// HNSW graph for this cluster (if size >= min threshold).
    pub hnsw: Option<HnswIndex>,

    /// Vectors stored in this cluster (for brute-force fallback or HNSW search).
    pub vectors: Vec<Vec<f32>>,
}

/// IVF-HNSW hybrid index.
pub struct IvfHnswIndex {
    /// Configuration.
    pub config: IvfHnswConfig,

    /// Clusters with HNSW graphs.
    pub clusters: Vec<IvfHnswCluster>,

    /// Vector dimension.
    pub dim: usize,

    /// Distance metric.
    pub metric: Metric,

    /// Total number of vectors.
    pub num_vectors: usize,

    /// Vector IDs (in original order).
    pub ids: Vec<VectorId>,
}

impl IvfHnswIndex {
    /// Build an IVF-HNSW index from vectors.
    pub fn build(
        vectors: &[Vec<f32>],
        ids: &[VectorId],
        metric: Metric,
        config: IvfHnswConfig,
    ) -> Self {
        if vectors.is_empty() {
            return Self {
                config,
                clusters: Vec::new(),
                dim: 0,
                metric,
                num_vectors: 0,
                ids: Vec::new(),
            };
        }

        let dim = vectors[0].len();
        let num_vectors = vectors.len();

        tracing::info!(
            "Building IVF-HNSW index: {} vectors, {} clusters, dim={}",
            num_vectors,
            config.num_clusters,
            dim
        );

        // Step 1: Run k-means clustering
        let kmeans_config = config.kmeans.to_config(config.num_clusters);
        let kmeans_result = kmeans(vectors, &kmeans_config);

        tracing::info!(
            "K-means completed: {} iterations, converged={}",
            kmeans_result.iterations,
            kmeans_result.converged
        );

        // Step 2: Build cluster data
        let mut cluster_data = Self::build_cluster_data(
            &kmeans_result.centroids,
            &kmeans_result.assignments,
            vectors,
        );

        // Step 3: Build HNSW graphs for each cluster
        let hnsw_config = config.hnsw.to_config();
        let clusters: Vec<IvfHnswCluster> = cluster_data
            .into_iter()
            .map(|(centroid, indices, cluster_vectors)| {
                let hnsw = if config.use_hnsw
                    && cluster_vectors.len() >= config.min_cluster_size_for_hnsw
                {
                    tracing::debug!(
                        "Building HNSW for cluster with {} vectors",
                        cluster_vectors.len()
                    );
                    let builder = HnswBuilder::new(dim, metric, hnsw_config.clone());
                    Some(builder.build_batch(&cluster_vectors).unwrap())
                } else {
                    None
                };

                IvfHnswCluster {
                    centroid,
                    vector_indices: indices,
                    hnsw,
                    vectors: cluster_vectors,
                }
            })
            .collect();

        tracing::info!(
            "IVF-HNSW index built: {} clusters, {} with HNSW graphs",
            clusters.len(),
            clusters.iter().filter(|c| c.hnsw.is_some()).count()
        );

        Self {
            config,
            clusters,
            dim,
            metric,
            num_vectors,
            ids: ids.to_vec(),
        }
    }

    /// Build cluster data from k-means results.
    fn build_cluster_data(
        centroids: &[Vec<f32>],
        assignments: &[usize],
        vectors: &[Vec<f32>],
    ) -> Vec<(Vec<f32>, Vec<usize>, Vec<Vec<f32>>)> {
        let k = centroids.len();
        let mut cluster_indices: Vec<Vec<usize>> = vec![Vec::new(); k];

        for (vec_idx, &cluster_idx) in assignments.iter().enumerate() {
            cluster_indices[cluster_idx].push(vec_idx);
        }

        centroids
            .iter()
            .cloned()
            .zip(cluster_indices)
            .map(|(centroid, indices)| {
                let cluster_vectors: Vec<Vec<f32>> =
                    indices.iter().map(|&i| vectors[i].clone()).collect();
                (centroid, indices, cluster_vectors)
            })
            .collect()
    }

    /// Search the index for k nearest neighbors.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<IvfHnswSearchResult> {
        self.search_with_nprobe(query, k, self.config.nprobe)
    }

    /// Search with custom nprobe.
    pub fn search_with_nprobe(
        &self,
        query: &[f32],
        k: usize,
        nprobe: usize,
    ) -> Vec<IvfHnswSearchResult> {
        if self.clusters.is_empty() || k == 0 {
            return Vec::new();
        }

        let nprobe = nprobe.min(self.clusters.len());

        // Step 1: Find top nprobe clusters by distance to centroids
        let mut cluster_distances: Vec<(usize, f32)> = self
            .clusters
            .iter()
            .enumerate()
            .map(|(i, cluster)| {
                let dist = distance::distance(query, &cluster.centroid, self.metric);
                (i, dist)
            })
            .collect();

        cluster_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        // Step 2: Search within top nprobe clusters
        let mut results: Vec<IvfHnswSearchResult> = Vec::new();

        for &(cluster_idx, _) in cluster_distances.iter().take(nprobe) {
            let cluster = &self.clusters[cluster_idx];
            let cluster_results = self.search_cluster(cluster, query, k);

            // Map local indices back to global
            for result in cluster_results {
                let global_idx = cluster.vector_indices[result.local_idx];
                results.push(IvfHnswSearchResult {
                    id: self.ids[global_idx].clone(),
                    distance: result.distance,
                    cluster_idx,
                    local_idx: result.local_idx,
                });
            }
        }

        // Step 3: Sort and truncate to k
        results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(Ordering::Equal));
        results.truncate(k);

        results
    }

    /// Search within a single cluster.
    fn search_cluster(
        &self,
        cluster: &IvfHnswCluster,
        query: &[f32],
        k: usize,
    ) -> Vec<LocalSearchResult> {
        if cluster.vectors.is_empty() {
            return Vec::new();
        }

        if let Some(hnsw) = &cluster.hnsw {
            // Use HNSW search
            match search_hnsw(hnsw, query, k) {
                Ok(hnsw_results) => hnsw_results
                    .into_iter()
                    .map(|r| LocalSearchResult {
                        local_idx: r.id,
                        distance: r.distance,
                    })
                    .collect(),
                Err(_) => self.brute_force_cluster(cluster, query, k),
            }
        } else {
            // Brute-force search
            self.brute_force_cluster(cluster, query, k)
        }
    }

    /// Brute-force search within a cluster.
    fn brute_force_cluster(
        &self,
        cluster: &IvfHnswCluster,
        query: &[f32],
        k: usize,
    ) -> Vec<LocalSearchResult> {
        let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::with_capacity(k + 1);

        for (idx, vector) in cluster.vectors.iter().enumerate() {
            let dist = distance::distance(query, vector, self.metric);

            if heap.len() < k {
                heap.push(HeapEntry {
                    distance: dist,
                    index: idx,
                });
            } else if let Some(worst) = heap.peek() {
                if dist < worst.distance {
                    heap.pop();
                    heap.push(HeapEntry {
                        distance: dist,
                        index: idx,
                    });
                }
            }
        }

        let mut results: Vec<LocalSearchResult> = heap
            .into_iter()
            .map(|e| LocalSearchResult {
                local_idx: e.index,
                distance: e.distance,
            })
            .collect();

        results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(Ordering::Equal));
        results
    }

    /// Get index statistics.
    pub fn stats(&self) -> IvfHnswStats {
        let hnsw_clusters = self.clusters.iter().filter(|c| c.hnsw.is_some()).count();
        let avg_cluster_size = if self.clusters.is_empty() {
            0.0
        } else {
            self.num_vectors as f64 / self.clusters.len() as f64
        };

        let min_cluster_size = self
            .clusters
            .iter()
            .map(|c| c.vectors.len())
            .min()
            .unwrap_or(0);
        let max_cluster_size = self
            .clusters
            .iter()
            .map(|c| c.vectors.len())
            .max()
            .unwrap_or(0);

        IvfHnswStats {
            num_vectors: self.num_vectors,
            num_clusters: self.clusters.len(),
            hnsw_clusters,
            avg_cluster_size,
            min_cluster_size,
            max_cluster_size,
        }
    }
}

/// Local search result within a cluster.
struct LocalSearchResult {
    local_idx: usize,
    distance: f32,
}

/// Search result from IVF-HNSW index.
#[derive(Debug, Clone)]
pub struct IvfHnswSearchResult {
    /// Vector ID.
    pub id: VectorId,

    /// Distance to query.
    pub distance: f32,

    /// Cluster index where this vector was found.
    pub cluster_idx: usize,

    /// Local index within the cluster.
    pub local_idx: usize,
}

/// Max-heap entry for top-k tracking.
#[derive(Debug)]
struct HeapEntry {
    distance: f32,
    index: usize,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal)
    }
}

/// IVF-HNSW index statistics.
#[derive(Debug, Clone)]
pub struct IvfHnswStats {
    pub num_vectors: usize,
    pub num_clusters: usize,
    pub hnsw_clusters: usize,
    pub avg_cluster_size: f64,
    pub min_cluster_size: usize,
    pub max_cluster_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    fn generate_test_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect()
    }

    #[test]
    fn test_ivf_hnsw_build() {
        let vectors = generate_test_vectors(1000, 32, 42);
        let ids: Vec<VectorId> = (0..1000).map(|i| VectorId::new(format!("v{}", i))).collect();

        let config = IvfHnswConfig {
            num_clusters: 10,
            nprobe: 3,
            use_hnsw: true,
            min_cluster_size_for_hnsw: 10,
            ..Default::default()
        };

        let index = IvfHnswIndex::build(&vectors, &ids, Metric::L2, config);

        assert_eq!(index.num_vectors, 1000);
        assert_eq!(index.clusters.len(), 10);

        let stats = index.stats();
        assert!(stats.hnsw_clusters > 0);
    }

    #[test]
    fn test_ivf_hnsw_search() {
        let vectors = generate_test_vectors(500, 16, 123);
        let ids: Vec<VectorId> = (0..500).map(|i| VectorId::new(format!("v{}", i))).collect();

        let config = IvfHnswConfig {
            num_clusters: 5,
            nprobe: 3,
            use_hnsw: true,
            min_cluster_size_for_hnsw: 10,
            ..Default::default()
        };

        let index = IvfHnswIndex::build(&vectors, &ids, Metric::L2, config);

        // Search for a vector that's in the index
        let query = &vectors[0];
        let results = index.search(query, 5);

        assert!(!results.is_empty());
        // First result should be the query itself (or very close)
        assert!(results[0].distance < 0.001);
    }

    #[test]
    fn test_ivf_hnsw_without_hnsw() {
        let vectors = generate_test_vectors(200, 8, 456);
        let ids: Vec<VectorId> = (0..200).map(|i| VectorId::new(format!("v{}", i))).collect();

        let config = IvfHnswConfig {
            num_clusters: 4,
            nprobe: 2,
            use_hnsw: false, // Disable HNSW
            ..Default::default()
        };

        let index = IvfHnswIndex::build(&vectors, &ids, Metric::L2, config);

        let stats = index.stats();
        assert_eq!(stats.hnsw_clusters, 0);

        // Search should still work (brute-force)
        let query = &vectors[50];
        let results = index.search(query, 3);

        assert!(!results.is_empty());
    }

    #[test]
    fn test_empty_index() {
        let vectors: Vec<Vec<f32>> = Vec::new();
        let ids: Vec<VectorId> = Vec::new();

        let config = IvfHnswConfig::default();
        let index = IvfHnswIndex::build(&vectors, &ids, Metric::L2, config);

        assert_eq!(index.num_vectors, 0);
        assert!(index.clusters.is_empty());

        let results = index.search(&[0.0, 0.0, 0.0], 5);
        assert!(results.is_empty());
    }
}
