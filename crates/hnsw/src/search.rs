//! HNSW search algorithms.

use crate::config::HnswConfig;
use crate::error::{HnswError, HnswResult};
use crate::graph::HnswGraph;
use crate::HnswIndex;
use puffer_core::distance::{distance, l2_distance_squared};
use puffer_core::Metric;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

/// Result of an HNSW search.
#[derive(Debug, Clone)]
pub struct HnswSearchResult {
    /// Vector ID.
    pub id: usize,
    /// Distance to query.
    pub distance: f32,
}

impl PartialEq for HnswSearchResult {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for HnswSearchResult {}

impl PartialOrd for HnswSearchResult {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HnswSearchResult {
    fn cmp(&self, other: &Self) -> Ordering {
        // Max-heap: larger distances are "greater"
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal)
    }
}

/// Internal candidate for search (inverted ordering for min-heap behavior).
#[derive(Debug, Clone)]
struct Candidate {
    id: usize,
    distance: f32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reversed for min-heap behavior
        other.distance
            .partial_cmp(&self.distance)
            .unwrap_or(Ordering::Equal)
    }
}

/// Search the HNSW index for nearest neighbors.
///
/// # Arguments
/// * `index` - The HNSW index
/// * `query` - Query vector
/// * `k` - Number of neighbors to return
///
/// # Returns
/// Up to k nearest neighbors sorted by distance (ascending).
pub fn search_hnsw(
    index: &HnswIndex,
    query: &[f32],
    k: usize,
) -> HnswResult<Vec<HnswSearchResult>> {
    search_hnsw_with_ef(index, query, k, index.config.ef_search)
}

/// Search with custom ef parameter.
pub fn search_hnsw_with_ef(
    index: &HnswIndex,
    query: &[f32],
    k: usize,
    ef: usize,
) -> HnswResult<Vec<HnswSearchResult>> {
    if query.len() != index.dim {
        return Err(HnswError::DimensionMismatch {
            expected: index.dim,
            got: query.len(),
        });
    }

    let entry_point = index.entry_point.ok_or(HnswError::EmptyIndex)?;
    let vectors = index.vectors.as_ref().ok_or(HnswError::VectorsNotStored)?;

    // Search from top layer down to layer 1
    let mut curr_node = entry_point;
    let mut curr_dist = compute_distance(query, &vectors[curr_node], index.metric);

    for layer in (1..=index.max_layer).rev() {
        let (new_node, new_dist) = greedy_search_layer(
            &index.graph,
            vectors,
            query,
            curr_node,
            curr_dist,
            layer,
            1,
            index.metric,
        );
        curr_node = new_node;
        curr_dist = new_dist;
    }

    // Search layer 0 with ef candidates
    let candidates = search_layer(
        &index.graph,
        vectors,
        query,
        curr_node,
        ef.max(k),
        0,
        index.metric,
    );

    // Return top k
    let mut results: Vec<HnswSearchResult> = candidates
        .into_iter()
        .take(k)
        .map(|c| HnswSearchResult {
            id: c.id,
            distance: c.distance,
        })
        .collect();

    results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(Ordering::Equal));

    Ok(results)
}

/// Search with external vectors (for IVF+HNSW where vectors are stored elsewhere).
///
/// Takes a slice of vectors instead of a closure for simpler lifetime handling.
pub fn search_hnsw_external(
    graph: &HnswGraph,
    vectors: &[Vec<f32>],
    entry_point: usize,
    max_layer: usize,
    query: &[f32],
    k: usize,
    ef: usize,
    metric: Metric,
) -> Vec<HnswSearchResult> {
    if graph.is_empty() || vectors.is_empty() {
        return Vec::new();
    }

    // Search from top layer down to layer 1
    let mut curr_node = entry_point;
    let mut curr_dist = compute_distance(query, &vectors[curr_node], metric);

    for layer in (1..=max_layer).rev() {
        let (new_node, new_dist) = greedy_search_layer(
            graph,
            vectors,
            query,
            curr_node,
            curr_dist,
            layer,
            1,
            metric,
        );
        curr_node = new_node;
        curr_dist = new_dist;
    }

    // Search layer 0
    let candidates = search_layer(
        graph,
        vectors,
        query,
        curr_node,
        ef.max(k),
        0,
        metric,
    );

    candidates
        .into_iter()
        .take(k)
        .map(|c| HnswSearchResult {
            id: c.id,
            distance: c.distance,
        })
        .collect()
}

/// Greedy search on a single layer (returns single best candidate).
fn greedy_search_layer(
    graph: &HnswGraph,
    vectors: &[Vec<f32>],
    query: &[f32],
    start: usize,
    start_dist: f32,
    layer: usize,
    ef: usize,
    metric: Metric,
) -> (usize, f32) {
    let mut visited = HashSet::new();
    visited.insert(start);

    let mut best_id = start;
    let mut best_dist = start_dist;

    let mut candidates: BinaryHeap<Candidate> = BinaryHeap::new();
    candidates.push(Candidate {
        id: start,
        distance: start_dist,
    });

    while let Some(curr) = candidates.pop() {
        if curr.distance > best_dist {
            break;
        }

        for &neighbor in graph.get_neighbors(curr.id, layer) {
            let neighbor = neighbor as usize;
            if visited.insert(neighbor) {
                let dist = compute_distance(query, &vectors[neighbor], metric);
                if dist < best_dist {
                    best_dist = dist;
                    best_id = neighbor;
                }
                candidates.push(Candidate {
                    id: neighbor,
                    distance: dist,
                });
            }
        }
    }

    (best_id, best_dist)
}


/// Search on layer 0 (returns ef candidates).
fn search_layer(
    graph: &HnswGraph,
    vectors: &[Vec<f32>],
    query: &[f32],
    start: usize,
    ef: usize,
    layer: usize,
    metric: Metric,
) -> Vec<Candidate> {
    let mut visited = HashSet::new();
    visited.insert(start);

    let start_dist = compute_distance(query, &vectors[start], metric);

    // Candidates to explore (min-heap by distance)
    let mut candidates: BinaryHeap<Candidate> = BinaryHeap::new();
    candidates.push(Candidate {
        id: start,
        distance: start_dist,
    });

    // Results (max-heap by distance to track worst candidate)
    let mut results: BinaryHeap<HnswSearchResult> = BinaryHeap::new();
    results.push(HnswSearchResult {
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

        for &neighbor in graph.get_neighbors(curr.id, layer) {
            let neighbor = neighbor as usize;
            if visited.insert(neighbor) {
                let dist = compute_distance(query, &vectors[neighbor], metric);

                let should_add = results.len() < ef || {
                    if let Some(worst) = results.peek() {
                        dist < worst.distance
                    } else {
                        true
                    }
                };

                if should_add {
                    candidates.push(Candidate {
                        id: neighbor,
                        distance: dist,
                    });
                    results.push(HnswSearchResult {
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

    // Convert to sorted vec
    let mut result_vec: Vec<Candidate> = results
        .into_iter()
        .map(|r| Candidate {
            id: r.id,
            distance: r.distance,
        })
        .collect();
    result_vec.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(Ordering::Equal));
    result_vec
}


/// Compute distance using the specified metric.
#[inline]
fn compute_distance(a: &[f32], b: &[f32], metric: Metric) -> f32 {
    distance(a, b, metric)
}

/// Select neighbors using simple heuristic (keep M closest).
pub fn select_neighbors_simple(
    candidates: &[(usize, f32)],
    m: usize,
) -> Vec<u32> {
    let mut sorted = candidates.to_vec();
    sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    sorted.truncate(m);
    sorted.iter().map(|(id, _)| *id as u32).collect()
}

/// Select neighbors using heuristic (diversity-aware selection).
pub fn select_neighbors_heuristic(
    candidates: &[(usize, f32)],
    m: usize,
    extend_candidates: bool,
    metric: Metric,
    vectors: &[Vec<f32>],
) -> Vec<u32> {
    if candidates.len() <= m {
        return candidates.iter().map(|(id, _)| *id as u32).collect();
    }

    let mut sorted = candidates.to_vec();
    sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

    let mut selected: Vec<(usize, f32)> = Vec::with_capacity(m);
    let mut discarded: Vec<(usize, f32)> = Vec::new();

    for (id, dist) in sorted {
        if selected.len() >= m {
            break;
        }

        // Check if this candidate is closer to query than to any selected neighbor
        let mut good_candidate = true;
        let vec = &vectors[id];

        for &(sel_id, _) in &selected {
            let dist_to_selected = compute_distance(vec, &vectors[sel_id], metric);
            if dist_to_selected < dist {
                good_candidate = false;
                discarded.push((id, dist));
                break;
            }
        }

        if good_candidate {
            selected.push((id, dist));
        }
    }

    // Extend with discarded if not enough
    if extend_candidates && selected.len() < m {
        for (id, dist) in discarded {
            if selected.len() >= m {
                break;
            }
            selected.push((id, dist));
        }
    }

    selected.iter().map(|(id, _)| *id as u32).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::HnswBuilder;

    fn generate_random_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        use rand::prelude::*;
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect()
    }

    #[test]
    fn test_basic_search() {
        let vectors = generate_random_vectors(100, 16, 42);
        let config = HnswConfig::new(8).with_ef_construction(50);

        let mut builder = HnswBuilder::new(16, Metric::L2, config);
        for (id, vec) in vectors.iter().enumerate() {
            builder.add(id, vec).unwrap();
        }
        let index = builder.build();

        let results = search_hnsw(&index, &vectors[0], 5).unwrap();

        assert_eq!(results.len(), 5);
        // First result should be the query itself
        assert_eq!(results[0].id, 0);
        assert!(results[0].distance < 1e-6);
    }

    #[test]
    fn test_select_neighbors_simple() {
        let candidates = vec![
            (0, 0.5),
            (1, 0.3),
            (2, 0.7),
            (3, 0.1),
        ];

        let selected = select_neighbors_simple(&candidates, 2);

        assert_eq!(selected.len(), 2);
        assert!(selected.contains(&3)); // Closest
        assert!(selected.contains(&1)); // Second closest
    }
}
