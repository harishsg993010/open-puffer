//! IVF-Flat search implementation.

use puffer_core::{distance, Metric, VectorId};
use puffer_storage::LoadedSegment;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Compute distance using the most efficient method available.
///
/// For cosine distance with precomputed norms, this avoids recomputing norms.
#[inline]
fn compute_distance(
    query: &[f32],
    query_norm: f32,
    stored: &[f32],
    stored_norm: Option<f32>,
    metric: Metric,
) -> f32 {
    match metric {
        Metric::L2 => distance::l2_distance_squared(query, stored),
        Metric::Cosine => {
            if let Some(sn) = stored_norm {
                // Fast path: use precomputed norms
                distance::cosine_distance_with_norms(query, stored, query_norm, sn)
            } else {
                // Slow path: compute norms on the fly
                distance::cosine_distance(query, stored)
            }
        }
    }
}

/// A search result with distance and metadata.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: VectorId,
    pub distance: f32,
    pub payload: Option<serde_json::Value>,
}

impl SearchResult {
    pub fn new(id: VectorId, distance: f32) -> Self {
        Self {
            id,
            distance,
            payload: None,
        }
    }

    pub fn with_payload(mut self, payload: Option<serde_json::Value>) -> Self {
        self.payload = payload;
        self
    }
}

/// Internal struct for max-heap to track top-k nearest neighbors.
///
/// We use a max-heap so that peek() returns the element with the LARGEST distance
/// (the worst candidate among our top-k). This allows us to efficiently:
/// 1. Check if a new candidate is better than our current worst
/// 2. If so, remove the worst and add the new candidate
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
        // Natural ordering: larger distances are "greater"
        // This makes BinaryHeap (a max-heap) keep the largest distance at the top
        self.distance.partial_cmp(&other.distance).unwrap_or(Ordering::Equal)
    }
}

/// Search a single segment using IVF-Flat.
///
/// # Arguments
/// * `segment` - The loaded segment to search
/// * `query` - Query vector
/// * `k` - Number of results to return
/// * `nprobe` - Number of clusters to probe
/// * `include_payload` - Whether to load payloads
pub fn search_segment(
    segment: &LoadedSegment,
    query: &[f32],
    k: usize,
    nprobe: usize,
    include_payload: bool,
) -> Vec<SearchResult> {
    if segment.num_vectors() == 0 || k == 0 {
        return Vec::new();
    }

    let metric = segment.metric();
    let nprobe = nprobe.min(segment.clusters.len());
    let has_norms = segment.has_norms();

    // Precompute query norm once for cosine similarity
    let query_norm = if metric == Metric::Cosine {
        distance::l2_norm(query)
    } else {
        0.0 // Not used for L2
    };

    // Step 1: Find top nprobe clusters by distance to centroids
    let mut cluster_distances: Vec<(usize, f32)> = segment
        .clusters
        .iter()
        .enumerate()
        .map(|(i, cluster)| {
            let dist = distance::distance(query, &cluster.centroid, metric);
            (i, dist)
        })
        .collect();

    // Sort by distance (ascending)
    cluster_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

    // Step 2: Search vectors in top nprobe clusters
    let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::with_capacity(k + 1);

    for &(cluster_idx, _) in cluster_distances.iter().take(nprobe) {
        let cluster = &segment.clusters[cluster_idx];
        let start = cluster.start_index as usize;
        let end = start + cluster.length as usize;

        for vec_idx in start..end {
            let vector = segment.get_vector(vec_idx);
            let stored_norm = if has_norms {
                segment.get_norm(vec_idx)
            } else {
                None
            };
            let dist = compute_distance(query, query_norm, vector, stored_norm, metric);

            if heap.len() < k {
                heap.push(HeapEntry {
                    distance: dist,
                    index: vec_idx,
                });
            } else if let Some(worst) = heap.peek() {
                if dist < worst.distance {
                    heap.pop();
                    heap.push(HeapEntry {
                        distance: dist,
                        index: vec_idx,
                    });
                }
            }
        }
    }

    // Step 3: Convert heap to results
    let mut results: Vec<SearchResult> = heap
        .into_iter()
        .filter_map(|entry| {
            let id = segment.get_id(entry.index).ok()?;
            let payload = if include_payload {
                segment.get_payload(entry.index).ok().flatten()
            } else {
                None
            };
            Some(SearchResult {
                id,
                distance: entry.distance,
                payload,
            })
        })
        .collect();

    // Sort by distance (ascending)
    results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(Ordering::Equal));

    results
}

/// Merge search results from multiple segments.
pub fn merge_results(mut all_results: Vec<SearchResult>, k: usize) -> Vec<SearchResult> {
    all_results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(Ordering::Equal));
    all_results.truncate(k);
    all_results
}

/// Brute-force search for correctness verification.
pub fn brute_force_search(
    vectors: &[Vec<f32>],
    ids: &[VectorId],
    query: &[f32],
    metric: Metric,
    k: usize,
) -> Vec<SearchResult> {
    let mut distances: Vec<(usize, f32)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| (i, distance::distance(query, v, metric)))
        .collect();

    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

    distances
        .into_iter()
        .take(k)
        .map(|(i, dist)| SearchResult::new(ids[i].clone(), dist))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use puffer_storage::SegmentBuilder;
    use tempfile::tempdir;

    fn create_test_segment(
        dim: usize,
        num_vectors: usize,
        num_clusters: usize,
    ) -> (tempfile::TempDir, std::path::PathBuf, Vec<Vec<f32>>, Vec<VectorId>) {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.seg");

        let mut rng = rand::thread_rng();

        // Generate vectors
        let vectors: Vec<Vec<f32>> = (0..num_vectors)
            .map(|_| (0..dim).map(|_| rand::Rng::gen_range(&mut rng, -1.0..1.0)).collect())
            .collect();

        let ids: Vec<VectorId> = (0..num_vectors)
            .map(|i| VectorId::new(format!("vec_{}", i)))
            .collect();

        let payloads: Vec<Option<serde_json::Value>> = vec![None; num_vectors];

        // Simple clustering: divide vectors evenly
        let vectors_per_cluster = num_vectors / num_clusters;
        let mut cluster_data = Vec::new();

        for c in 0..num_clusters {
            let start = c * vectors_per_cluster;
            let end = if c == num_clusters - 1 {
                num_vectors
            } else {
                (c + 1) * vectors_per_cluster
            };

            // Compute centroid
            let mut centroid = vec![0.0f32; dim];
            for i in start..end {
                for (j, &v) in vectors[i].iter().enumerate() {
                    centroid[j] += v;
                }
            }
            let count = (end - start) as f32;
            for c in &mut centroid {
                *c /= count;
            }

            let indices: Vec<usize> = (start..end).collect();
            cluster_data.push((centroid, indices));
        }

        let builder = SegmentBuilder::new(dim, Metric::L2);
        builder
            .build(cluster_data, &vectors, &ids, &payloads, &path)
            .unwrap();

        (dir, path, vectors, ids)
    }

    #[test]
    fn test_search_segment_basic() {
        let (dir, path, vectors, ids) = create_test_segment(4, 100, 4);
        let segment = LoadedSegment::open(&path).unwrap();

        let query = &vectors[0];
        let results = search_segment(&segment, query, 5, 4, false);

        assert!(!results.is_empty());
        assert!(results.len() <= 5);

        // First result should be the query vector itself (distance ~0)
        assert!(results[0].distance < 0.001);
    }

    #[test]
    fn test_search_vs_brute_force() {
        let (dir, path, vectors, ids) = create_test_segment(8, 200, 10);
        let segment = LoadedSegment::open(&path).unwrap();

        let query = vec![0.5; 8];
        let k = 10;

        // IVF search with all clusters (should give same results as brute force)
        let ivf_results = search_segment(&segment, &query, k, 10, false);
        let bf_results = brute_force_search(&vectors, &ids, &query, Metric::L2, k);

        // Compare results (may have slight differences due to ordering of equal distances)
        assert_eq!(ivf_results.len(), bf_results.len());

        // First result should have same distance
        let eps = 1e-5;
        assert!(
            (ivf_results[0].distance - bf_results[0].distance).abs() < eps,
            "First result distance mismatch: {} vs {}",
            ivf_results[0].distance,
            bf_results[0].distance
        );
    }

    #[test]
    fn test_merge_results() {
        let results1 = vec![
            SearchResult::new("a".into(), 0.1),
            SearchResult::new("b".into(), 0.3),
        ];
        let results2 = vec![
            SearchResult::new("c".into(), 0.2),
            SearchResult::new("d".into(), 0.4),
        ];

        let mut all: Vec<SearchResult> = results1.into_iter().chain(results2).collect();
        let merged = merge_results(all, 3);

        assert_eq!(merged.len(), 3);
        assert_eq!(merged[0].id.as_str(), "a");
        assert_eq!(merged[1].id.as_str(), "c");
        assert_eq!(merged[2].id.as_str(), "b");
    }
}
