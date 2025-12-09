//! PQ-based approximate nearest neighbor search using ADC (Asymmetric Distance Computation).

use crate::codebook::PqCodebook;
use crate::encoding::decode_vector;
use crate::error::{PqError, PqResult};
use puffer_core::distance::l2_distance_squared;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Result of a PQ search.
#[derive(Debug, Clone)]
pub struct PqSearchResult {
    /// Index of the vector in the original dataset.
    pub index: usize,
    /// Approximate distance computed via ADC.
    pub approx_distance: f32,
    /// Exact distance (if refinement was performed).
    pub exact_distance: Option<f32>,
}

impl PartialEq for PqSearchResult {
    fn eq(&self, other: &Self) -> bool {
        self.approx_distance == other.approx_distance
    }
}

impl Eq for PqSearchResult {}

impl PartialOrd for PqSearchResult {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PqSearchResult {
    fn cmp(&self, other: &Self) -> Ordering {
        // Max-heap: larger distances have higher priority (to be evicted)
        self.approx_distance
            .partial_cmp(&other.approx_distance)
            .unwrap_or(Ordering::Equal)
    }
}

/// Distance table for ADC (Asymmetric Distance Computation).
///
/// Layout: [M][K] where entry [m][k] = distance from query subvector m to centroid k.
pub struct DistanceTable {
    pub table: Vec<f32>,
    pub num_subvectors: usize,
    pub codebook_size: usize,
}

impl DistanceTable {
    /// Get distance for subvector m, code k.
    #[inline]
    pub fn get(&self, m: usize, k: usize) -> f32 {
        self.table[m * self.codebook_size + k]
    }

    /// Compute approximate distance for a set of codes.
    #[inline]
    pub fn compute_distance(&self, codes: &[u16]) -> f32 {
        let mut dist = 0.0f32;
        for (m, &code) in codes.iter().enumerate() {
            dist += self.get(m, code as usize);
        }
        dist
    }

    /// Compute approximate distance from compact bytes.
    #[inline]
    pub fn compute_distance_compact(&self, codes: &[u8]) -> f32 {
        let mut dist = 0.0f32;
        for (m, &code) in codes.iter().enumerate() {
            dist += self.get(m, code as usize);
        }
        dist
    }

    /// Compute approximate distance from u16 compact bytes.
    #[inline]
    pub fn compute_distance_compact_u16(&self, bytes: &[u8]) -> f32 {
        let mut dist = 0.0f32;
        for (m, chunk) in bytes.chunks_exact(2).enumerate() {
            let code = u16::from_le_bytes([chunk[0], chunk[1]]) as usize;
            dist += self.get(m, code);
        }
        dist
    }
}

/// Precompute distance table for a query vector.
///
/// This table allows O(M) approximate distance computation instead of O(D).
pub fn compute_distance_table(codebook: &PqCodebook, query: &[f32]) -> PqResult<DistanceTable> {
    if !codebook.trained {
        return Err(PqError::CodebookNotTrained);
    }

    if query.len() != codebook.dim {
        return Err(PqError::DimensionMismatch {
            expected: codebook.dim,
            got: query.len(),
        });
    }

    let m = codebook.num_subvectors;
    let k = codebook.codebook_size;
    let mut table = vec![0.0f32; m * k];

    for mi in 0..m {
        let query_subvec = codebook.extract_subvector(query, mi);

        for ki in 0..k {
            let centroid = codebook.get_centroid(mi, ki);
            let dist = l2_distance_squared(query_subvec, centroid);
            table[mi * k + ki] = dist;
        }
    }

    Ok(DistanceTable {
        table,
        num_subvectors: m,
        codebook_size: k,
    })
}

/// Search PQ-encoded vectors using ADC.
///
/// # Arguments
/// * `codebook` - Trained PQ codebook
/// * `codes` - PQ codes for each vector (shape: [N][M])
/// * `query` - Query vector
/// * `k` - Number of nearest neighbors to return
///
/// # Returns
/// Top-k results sorted by approximate distance (ascending).
pub fn search_pq(
    codebook: &PqCodebook,
    codes: &[Vec<u16>],
    query: &[f32],
    k: usize,
) -> PqResult<Vec<PqSearchResult>> {
    let dist_table = compute_distance_table(codebook, query)?;

    // Use a max-heap to track top-k
    let mut heap: BinaryHeap<PqSearchResult> = BinaryHeap::with_capacity(k + 1);

    for (index, code) in codes.iter().enumerate() {
        let approx_dist = dist_table.compute_distance(code);

        if heap.len() < k {
            heap.push(PqSearchResult {
                index,
                approx_distance: approx_dist,
                exact_distance: None,
            });
        } else if let Some(worst) = heap.peek() {
            if approx_dist < worst.approx_distance {
                heap.pop();
                heap.push(PqSearchResult {
                    index,
                    approx_distance: approx_dist,
                    exact_distance: None,
                });
            }
        }
    }

    // Convert to sorted vec
    let mut results: Vec<_> = heap.into_vec();
    results.sort_by(|a, b| {
        a.approx_distance
            .partial_cmp(&b.approx_distance)
            .unwrap_or(Ordering::Equal)
    });

    Ok(results)
}

/// Search with compact codes (u8 per subvector, for codebook_size <= 256).
pub fn search_pq_compact(
    codebook: &PqCodebook,
    codes: &[Vec<u8>],
    query: &[f32],
    k: usize,
) -> PqResult<Vec<PqSearchResult>> {
    let dist_table = compute_distance_table(codebook, query)?;

    let mut heap: BinaryHeap<PqSearchResult> = BinaryHeap::with_capacity(k + 1);

    for (index, code) in codes.iter().enumerate() {
        let approx_dist = if codebook.codebook_size <= 256 {
            dist_table.compute_distance_compact(code)
        } else {
            dist_table.compute_distance_compact_u16(code)
        };

        if heap.len() < k {
            heap.push(PqSearchResult {
                index,
                approx_distance: approx_dist,
                exact_distance: None,
            });
        } else if let Some(worst) = heap.peek() {
            if approx_dist < worst.approx_distance {
                heap.pop();
                heap.push(PqSearchResult {
                    index,
                    approx_distance: approx_dist,
                    exact_distance: None,
                });
            }
        }
    }

    let mut results: Vec<_> = heap.into_vec();
    results.sort_by(|a, b| {
        a.approx_distance
            .partial_cmp(&b.approx_distance)
            .unwrap_or(Ordering::Equal)
    });

    Ok(results)
}

/// Search with refinement using exact distances.
///
/// First performs PQ search to get candidates, then refines using full vectors.
pub fn search_pq_with_refinement(
    codebook: &PqCodebook,
    codes: &[Vec<u16>],
    full_vectors: &[Vec<f32>],
    query: &[f32],
    k: usize,
    refine_factor: usize,
) -> PqResult<Vec<PqSearchResult>> {
    // Get more candidates for refinement
    let candidates = k * refine_factor;
    let mut results = search_pq(codebook, codes, query, candidates)?;

    // Refine with exact distances
    for result in &mut results {
        if result.index < full_vectors.len() {
            let exact_dist = l2_distance_squared(query, &full_vectors[result.index]);
            result.exact_distance = Some(exact_dist);
        }
    }

    // Re-sort by exact distance
    results.sort_by(|a, b| {
        match (a.exact_distance, b.exact_distance) {
            (Some(da), Some(db)) => da.partial_cmp(&db).unwrap_or(Ordering::Equal),
            (Some(_), None) => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
            (None, None) => a.approx_distance
                .partial_cmp(&b.approx_distance)
                .unwrap_or(Ordering::Equal),
        }
    });

    results.truncate(k);
    Ok(results)
}

/// Parallel PQ search for large datasets.
pub fn search_pq_parallel(
    codebook: &PqCodebook,
    codes: &[Vec<u16>],
    query: &[f32],
    k: usize,
) -> PqResult<Vec<PqSearchResult>> {
    let dist_table = compute_distance_table(codebook, query)?;

    // Compute distances in parallel
    let distances: Vec<(usize, f32)> = codes
        .par_iter()
        .enumerate()
        .map(|(index, code)| (index, dist_table.compute_distance(code)))
        .collect();

    // Use heap for top-k selection
    let mut heap: BinaryHeap<PqSearchResult> = BinaryHeap::with_capacity(k + 1);

    for (index, approx_dist) in distances {
        if heap.len() < k {
            heap.push(PqSearchResult {
                index,
                approx_distance: approx_dist,
                exact_distance: None,
            });
        } else if let Some(worst) = heap.peek() {
            if approx_dist < worst.approx_distance {
                heap.pop();
                heap.push(PqSearchResult {
                    index,
                    approx_distance: approx_dist,
                    exact_distance: None,
                });
            }
        }
    }

    let mut results: Vec<_> = heap.into_vec();
    results.sort_by(|a, b| {
        a.approx_distance
            .partial_cmp(&b.approx_distance)
            .unwrap_or(Ordering::Equal)
    });

    Ok(results)
}

/// Batch search multiple queries in parallel.
pub fn batch_search_pq(
    codebook: &PqCodebook,
    codes: &[Vec<u16>],
    queries: &[Vec<f32>],
    k: usize,
) -> PqResult<Vec<Vec<PqSearchResult>>> {
    queries
        .par_iter()
        .map(|query| search_pq(codebook, codes, query, k))
        .collect()
}

/// Rerank results using full vectors.
pub fn rerank_with_full_vectors(
    results: &mut [PqSearchResult],
    full_vectors: &[Vec<f32>],
    query: &[f32],
) {
    for result in results.iter_mut() {
        if result.index < full_vectors.len() {
            let exact_dist = l2_distance_squared(query, &full_vectors[result.index]);
            result.exact_distance = Some(exact_dist);
        }
    }

    results.sort_by(|a, b| {
        match (a.exact_distance, b.exact_distance) {
            (Some(da), Some(db)) => da.partial_cmp(&db).unwrap_or(Ordering::Equal),
            (Some(_), None) => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
            (None, None) => a.approx_distance
                .partial_cmp(&b.approx_distance)
                .unwrap_or(Ordering::Equal),
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codebook::train_pq;
    use crate::config::PqParams;
    use crate::encoding::encode_vectors;

    fn generate_random_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        use rand::prelude::*;
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect()
    }

    #[test]
    fn test_distance_table() {
        let vectors = generate_random_vectors(500, 32, 42);
        let params = PqParams::new(4, 16).with_max_iterations(10);
        let codebook = train_pq(&vectors, &params).unwrap();

        let query = &vectors[0];
        let dist_table = compute_distance_table(&codebook, query).unwrap();

        assert_eq!(dist_table.num_subvectors, 4);
        assert_eq!(dist_table.codebook_size, 16);
        assert_eq!(dist_table.table.len(), 4 * 16);

        // All distances should be non-negative
        for &d in &dist_table.table {
            assert!(d >= 0.0);
        }
    }

    #[test]
    fn test_search_pq() {
        let vectors = generate_random_vectors(1000, 32, 42);
        let params = PqParams::new(4, 32).with_max_iterations(15);
        let codebook = train_pq(&vectors, &params).unwrap();
        let codes = encode_vectors(&codebook, &vectors).unwrap();

        let query = &vectors[0];
        let results = search_pq(&codebook, &codes, query, 10).unwrap();

        assert_eq!(results.len(), 10);

        // First result should be the query itself (index=0) - PQ approximation may have larger distance
        assert_eq!(results[0].index, 0,
            "First result should be query itself (index=0), got: {}", results[0].index);

        // Results should be sorted by distance
        for i in 1..results.len() {
            assert!(results[i - 1].approx_distance <= results[i].approx_distance);
        }
    }

    #[test]
    fn test_search_with_refinement() {
        let vectors = generate_random_vectors(1000, 32, 42);
        let params = PqParams::new(4, 32).with_max_iterations(15);
        let codebook = train_pq(&vectors, &params).unwrap();
        let codes = encode_vectors(&codebook, &vectors).unwrap();

        let query = &vectors[0];
        let results = search_pq_with_refinement(
            &codebook, &codes, &vectors, query, 10, 5
        ).unwrap();

        assert_eq!(results.len(), 10);

        // All results should have exact distances
        for result in &results {
            assert!(result.exact_distance.is_some());
        }

        // First result should have exact distance ~0
        assert!(results[0].exact_distance.unwrap() < 1e-6);

        // Results should be sorted by exact distance
        for i in 1..results.len() {
            assert!(
                results[i - 1].exact_distance.unwrap() <= results[i].exact_distance.unwrap()
            );
        }
    }

    #[test]
    fn test_batch_search() {
        let vectors = generate_random_vectors(500, 32, 42);
        let params = PqParams::new(4, 16).with_max_iterations(10);
        let codebook = train_pq(&vectors, &params).unwrap();
        let codes = encode_vectors(&codebook, &vectors).unwrap();

        let queries: Vec<Vec<f32>> = vectors[0..10].to_vec();
        let all_results = batch_search_pq(&codebook, &codes, &queries, 5).unwrap();

        assert_eq!(all_results.len(), 10);
        for results in &all_results {
            assert_eq!(results.len(), 5);
        }
    }

    #[test]
    fn test_recall_quality() {
        // Test that PQ search has reasonable recall
        let vectors = generate_random_vectors(1000, 64, 42);
        let params = PqParams::new(8, 64).with_max_iterations(20);
        let codebook = train_pq(&vectors, &params).unwrap();
        let codes = encode_vectors(&codebook, &vectors).unwrap();

        // Use a subset of vectors as queries
        let num_queries = 50;
        let k = 10;
        let mut total_recall = 0.0;

        for i in 0..num_queries {
            let query = &vectors[i];

            // Exact brute-force search
            let mut exact_distances: Vec<(usize, f32)> = vectors
                .iter()
                .enumerate()
                .map(|(j, v)| (j, l2_distance_squared(query, v)))
                .collect();
            exact_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let exact_top_k: Vec<usize> = exact_distances.iter().take(k).map(|x| x.0).collect();

            // PQ search
            let pq_results = search_pq(&codebook, &codes, query, k).unwrap();
            let pq_top_k: Vec<usize> = pq_results.iter().map(|r| r.index).collect();

            // Compute recall
            let hits = pq_top_k.iter().filter(|x| exact_top_k.contains(x)).count();
            total_recall += hits as f64 / k as f64;
        }

        let avg_recall = total_recall / num_queries as f64;
        assert!(avg_recall > 0.3, "PQ recall too low: {:.2}", avg_recall);
    }
}
