//! IVF + PQ (Product Quantization) indexing.
//!
//! This module combines:
//! - IVF for coarse partitioning
//! - PQ for compressed vector storage and ADC search
//!
//! Benefits:
//! - Reduced memory footprint (vectors compressed to PQ codes)
//! - Fast ADC (Asymmetric Distance Computation) search

use crate::kmeans::{kmeans, KMeansConfig};
use puffer_core::{distance, Metric, VectorId};
use puffer_pq::{
    codebook::{train_pq, PqCodebook},
    config::PqParams,
    encoding::{decode_vector, encode_vector, encode_vectors},
    search::{compute_distance_table, search_pq_with_refinement, PqSearchResult},
};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

/// Configuration for IVF-PQ index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IvfPqConfig {
    /// Number of IVF clusters.
    pub num_clusters: usize,

    /// Number of PQ subvectors.
    pub num_subvectors: usize,

    /// Codebook size per subvector (typically 256 for u8 codes).
    pub codebook_size: usize,

    /// Number of clusters to probe during search.
    pub nprobe: usize,

    /// Number of training samples for PQ.
    pub pq_training_samples: Option<usize>,

    /// K-means iterations for codebook training.
    pub pq_max_iterations: usize,

    /// Whether to use residual coding (subtract centroid before encoding).
    pub use_residual: bool,

    /// Number of candidates to refine with full vectors.
    pub refine_candidates: Option<usize>,
}

impl Default for IvfPqConfig {
    fn default() -> Self {
        Self {
            num_clusters: 100,
            num_subvectors: 8,
            codebook_size: 256,
            nprobe: 10,
            pq_training_samples: Some(10000),
            pq_max_iterations: 20,
            use_residual: true,
            refine_candidates: Some(100),
        }
    }
}

/// An IVF cluster with PQ-encoded vectors.
pub struct IvfPqCluster {
    /// Cluster centroid.
    pub centroid: Vec<f32>,

    /// Indices of vectors in this cluster.
    pub vector_indices: Vec<usize>,

    /// PQ codes for vectors in this cluster.
    pub codes: Vec<Vec<u16>>,
}

/// IVF-PQ index for compressed vector search.
pub struct IvfPqIndex {
    /// Configuration.
    pub config: IvfPqConfig,

    /// Clusters with PQ-encoded vectors.
    pub clusters: Vec<IvfPqCluster>,

    /// PQ codebook (shared across all clusters, or per-cluster if residual).
    pub codebook: PqCodebook,

    /// Vector dimension.
    pub dim: usize,

    /// Distance metric.
    pub metric: Metric,

    /// Total number of vectors.
    pub num_vectors: usize,

    /// Vector IDs.
    pub ids: Vec<VectorId>,

    /// Original vectors (for refinement, optional).
    pub original_vectors: Option<Vec<Vec<f32>>>,
}

impl IvfPqIndex {
    /// Build an IVF-PQ index.
    pub fn build(
        vectors: &[Vec<f32>],
        ids: &[VectorId],
        metric: Metric,
        config: IvfPqConfig,
        keep_original: bool,
    ) -> Result<Self, String> {
        if vectors.is_empty() {
            return Ok(Self {
                config,
                clusters: Vec::new(),
                codebook: PqCodebook {
                    num_subvectors: 0,
                    codebook_size: 0,
                    dim: 0,
                    subvector_dim: 0,
                    centroids: Vec::new(),
                    trained: false,
                },
                dim: 0,
                metric,
                num_vectors: 0,
                ids: Vec::new(),
                original_vectors: None,
            });
        }

        let dim = vectors[0].len();
        let num_vectors = vectors.len();

        tracing::info!(
            "Building IVF-PQ index: {} vectors, {} clusters, {} subvectors",
            num_vectors,
            config.num_clusters,
            config.num_subvectors
        );

        // Step 1: IVF clustering
        let kmeans_config = KMeansConfig {
            num_clusters: config.num_clusters,
            max_iterations: 20,
            convergence_threshold: 0.001,
            seed: None,
        };
        let kmeans_result = kmeans(vectors, &kmeans_config);

        // Step 2: Prepare training data for PQ
        let training_vectors: Vec<Vec<f32>> = if config.use_residual {
            // Compute residuals (vector - centroid)
            vectors
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let cluster_idx = kmeans_result.assignments[i];
                    let centroid = &kmeans_result.centroids[cluster_idx];
                    v.iter().zip(centroid.iter()).map(|(a, b)| a - b).collect()
                })
                .collect()
        } else {
            vectors.to_vec()
        };

        // Sample for PQ training if needed
        let pq_training_data: Vec<Vec<f32>> = if let Some(max_samples) = config.pq_training_samples
        {
            if training_vectors.len() > max_samples {
                let step = training_vectors.len() / max_samples;
                training_vectors
                    .iter()
                    .step_by(step)
                    .take(max_samples)
                    .cloned()
                    .collect()
            } else {
                training_vectors.clone()
            }
        } else {
            training_vectors.clone()
        };

        // Step 3: Train PQ codebook
        let pq_params = PqParams {
            num_subvectors: config.num_subvectors,
            codebook_size: config.codebook_size,
            training_samples: None, // Already sampled
            max_iterations: config.pq_max_iterations,
            convergence_threshold: 0.001,
            seed: None,
        };

        let codebook = train_pq(&pq_training_data, &pq_params)
            .map_err(|e| format!("PQ training failed: {}", e))?;

        tracing::info!("PQ codebook trained: {} subvectors, {} codes each",
            codebook.num_subvectors, codebook.codebook_size);

        // Step 4: Build clusters with PQ codes
        let mut clusters = Vec::with_capacity(config.num_clusters);
        for cluster_idx in 0..kmeans_result.centroids.len() {
            let centroid = kmeans_result.centroids[cluster_idx].clone();

            // Get indices for this cluster
            let indices: Vec<usize> = kmeans_result
                .assignments
                .iter()
                .enumerate()
                .filter_map(|(i, &c)| if c == cluster_idx { Some(i) } else { None })
                .collect();

            // Encode vectors (residuals or original)
            let codes: Vec<Vec<u16>> = if config.use_residual {
                indices
                    .iter()
                    .map(|&i| {
                        let residual: Vec<f32> = vectors[i]
                            .iter()
                            .zip(centroid.iter())
                            .map(|(a, b)| a - b)
                            .collect();
                        encode_vector(&codebook, &residual).unwrap_or_default()
                    })
                    .collect()
            } else {
                indices
                    .iter()
                    .map(|&i| encode_vector(&codebook, &vectors[i]).unwrap_or_default())
                    .collect()
            };

            clusters.push(IvfPqCluster {
                centroid,
                vector_indices: indices,
                codes,
            });
        }

        Ok(Self {
            config,
            clusters,
            codebook,
            dim,
            metric,
            num_vectors,
            ids: ids.to_vec(),
            original_vectors: if keep_original {
                Some(vectors.to_vec())
            } else {
                None
            },
        })
    }

    /// Search the index.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<IvfPqSearchResult> {
        self.search_with_nprobe(query, k, self.config.nprobe)
    }

    /// Search with custom nprobe.
    pub fn search_with_nprobe(
        &self,
        query: &[f32],
        k: usize,
        nprobe: usize,
    ) -> Vec<IvfPqSearchResult> {
        if self.clusters.is_empty() || k == 0 {
            return Vec::new();
        }

        let nprobe = nprobe.min(self.clusters.len());

        // Step 1: Find top nprobe clusters
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

        // Step 2: Search within clusters using ADC
        let mut all_results: Vec<(usize, usize, f32)> = Vec::new(); // (global_idx, cluster_idx, distance)

        for &(cluster_idx, _centroid_dist) in cluster_distances.iter().take(nprobe) {
            let cluster = &self.clusters[cluster_idx];

            if cluster.codes.is_empty() {
                continue;
            }

            // Compute query residual for this cluster (if using residual coding)
            let query_for_adc: Vec<f32> = if self.config.use_residual {
                query
                    .iter()
                    .zip(cluster.centroid.iter())
                    .map(|(a, b)| a - b)
                    .collect()
            } else {
                query.to_vec()
            };

            // Compute distance table for ADC
            if let Ok(dist_table) = compute_distance_table(&self.codebook, &query_for_adc) {
                // Compute approximate distances
                for (local_idx, codes) in cluster.codes.iter().enumerate() {
                    let approx_dist = dist_table.compute_distance(codes);
                    let global_idx = cluster.vector_indices[local_idx];
                    all_results.push((global_idx, cluster_idx, approx_dist));
                }
            }
        }

        // Step 3: Sort by approximate distance
        all_results.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(Ordering::Equal));

        // Step 4: Optional refinement with exact distances
        let results = if let (Some(refine_k), Some(orig_vectors)) =
            (self.config.refine_candidates, &self.original_vectors)
        {
            let refine_k = refine_k.min(all_results.len());
            let mut refined: Vec<(usize, f32)> = all_results
                .iter()
                .take(refine_k)
                .map(|&(global_idx, _, _)| {
                    let exact_dist = distance::distance(query, &orig_vectors[global_idx], self.metric);
                    (global_idx, exact_dist)
                })
                .collect();

            refined.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
            refined.truncate(k);

            refined
                .into_iter()
                .map(|(idx, dist)| IvfPqSearchResult {
                    id: self.ids[idx].clone(),
                    distance: dist,
                    approximate_distance: all_results
                        .iter()
                        .find(|r| r.0 == idx)
                        .map(|r| r.2),
                })
                .collect()
        } else {
            all_results
                .into_iter()
                .take(k)
                .map(|(global_idx, _, approx_dist)| IvfPqSearchResult {
                    id: self.ids[global_idx].clone(),
                    distance: approx_dist,
                    approximate_distance: Some(approx_dist),
                })
                .collect()
        };

        results
    }

    /// Get index statistics.
    pub fn stats(&self) -> IvfPqStats {
        let total_codes: usize = self.clusters.iter().map(|c| c.codes.len()).sum();
        let bytes_per_vector = self.codebook.num_subvectors * 2; // u16 per subvector
        let compressed_size = total_codes * bytes_per_vector;
        let original_size = self.num_vectors * self.dim * 4; // f32 = 4 bytes

        IvfPqStats {
            num_vectors: self.num_vectors,
            num_clusters: self.clusters.len(),
            num_subvectors: self.codebook.num_subvectors,
            codebook_size: self.codebook.codebook_size,
            compressed_bytes: compressed_size,
            original_bytes: original_size,
            compression_ratio: if compressed_size > 0 {
                original_size as f64 / compressed_size as f64
            } else {
                0.0
            },
        }
    }
}

/// Search result from IVF-PQ index.
#[derive(Debug, Clone)]
pub struct IvfPqSearchResult {
    pub id: VectorId,
    pub distance: f32,
    pub approximate_distance: Option<f32>,
}

/// IVF-PQ index statistics.
#[derive(Debug, Clone)]
pub struct IvfPqStats {
    pub num_vectors: usize,
    pub num_clusters: usize,
    pub num_subvectors: usize,
    pub codebook_size: usize,
    pub compressed_bytes: usize,
    pub original_bytes: usize,
    pub compression_ratio: f64,
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
    fn test_ivf_pq_build() {
        let vectors = generate_test_vectors(500, 32, 42);
        let ids: Vec<VectorId> = (0..500).map(|i| VectorId::new(format!("v{}", i))).collect();

        let config = IvfPqConfig {
            num_clusters: 10,
            num_subvectors: 4,
            codebook_size: 16, // Small for testing
            nprobe: 3,
            pq_training_samples: Some(200),
            pq_max_iterations: 5,
            use_residual: true,
            refine_candidates: Some(50),
        };

        let index = IvfPqIndex::build(&vectors, &ids, Metric::L2, config, true).unwrap();

        assert_eq!(index.num_vectors, 500);
        assert_eq!(index.clusters.len(), 10);

        let stats = index.stats();
        assert!(stats.compression_ratio > 1.0);
    }

    #[test]
    fn test_ivf_pq_search() {
        let vectors = generate_test_vectors(300, 16, 123);
        let ids: Vec<VectorId> = (0..300).map(|i| VectorId::new(format!("v{}", i))).collect();

        let config = IvfPqConfig {
            num_clusters: 5,
            num_subvectors: 2,
            codebook_size: 16,
            nprobe: 3,
            pq_training_samples: Some(100),
            pq_max_iterations: 5,
            use_residual: false,
            refine_candidates: None,
        };

        let index = IvfPqIndex::build(&vectors, &ids, Metric::L2, config, false).unwrap();

        let query = &vectors[0];
        let results = index.search(query, 5);

        assert!(!results.is_empty());
    }
}
