//! Segment router index for efficient query routing.
//!
//! The router maintains segment-level centroids to quickly determine
//! which segments are most likely to contain relevant vectors for a query.

use crate::error::StorageResult;
use puffer_core::{distance, Metric};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fs;
use std::path::Path;

/// Entry for a single segment in the router index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterEntry {
    /// Segment identifier (filename).
    pub segment_id: String,
    /// Level in LSM hierarchy (0 = L0 small, 1 = L1 merged, etc.).
    pub level: u32,
    /// Centroid of all vectors in this segment.
    pub segment_centroid: Vec<f32>,
    /// Number of vectors in this segment.
    pub num_vectors: usize,
}

/// Router index for a collection.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RouterIndex {
    /// Entries for all segments.
    pub entries: Vec<RouterEntry>,
}

impl RouterIndex {
    /// Create a new empty router index.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Add or update an entry for a segment.
    pub fn upsert(&mut self, entry: RouterEntry) {
        // Remove existing entry for this segment if present
        self.entries.retain(|e| e.segment_id != entry.segment_id);
        self.entries.push(entry);
    }

    /// Remove an entry by segment ID.
    pub fn remove(&mut self, segment_id: &str) {
        self.entries.retain(|e| e.segment_id != segment_id);
    }

    /// Check if a segment exists in the router.
    pub fn contains(&self, segment_id: &str) -> bool {
        self.entries.iter().any(|e| e.segment_id == segment_id)
    }

    /// Get entry by segment ID.
    pub fn get(&self, segment_id: &str) -> Option<&RouterEntry> {
        self.entries.iter().find(|e| e.segment_id == segment_id)
    }

    /// Select top M segments closest to the query vector.
    ///
    /// Returns segment IDs ordered by distance to query (ascending).
    pub fn select_segments(
        &self,
        query: &[f32],
        metric: Metric,
        top_m: usize,
    ) -> Vec<String> {
        if self.entries.is_empty() {
            return Vec::new();
        }

        let top_m = top_m.min(self.entries.len());

        // Compute distance to each segment centroid
        let mut distances: Vec<(usize, f32)> = self
            .entries
            .iter()
            .enumerate()
            .map(|(i, entry)| {
                let dist = distance::distance(query, &entry.segment_centroid, metric);
                (i, dist)
            })
            .collect();

        // Use partial sort for efficiency when top_m << total segments
        if top_m < distances.len() / 2 {
            // Use a max-heap to find top M minimum distances
            let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::with_capacity(top_m + 1);

            for (idx, dist) in distances {
                if heap.len() < top_m {
                    heap.push(HeapEntry { distance: dist, index: idx });
                } else if let Some(worst) = heap.peek() {
                    if dist < worst.distance {
                        heap.pop();
                        heap.push(HeapEntry { distance: dist, index: idx });
                    }
                }
            }

            // Extract and sort by distance
            let mut results: Vec<(usize, f32)> = heap
                .into_iter()
                .map(|e| (e.index, e.distance))
                .collect();
            results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

            results
                .into_iter()
                .map(|(i, _)| self.entries[i].segment_id.clone())
                .collect()
        } else {
            // Full sort is faster for small lists
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

            distances
                .into_iter()
                .take(top_m)
                .map(|(i, _)| self.entries[i].segment_id.clone())
                .collect()
        }
    }

    /// Get all segment IDs.
    pub fn all_segment_ids(&self) -> Vec<String> {
        self.entries.iter().map(|e| e.segment_id.clone()).collect()
    }

    /// Load router index from disk.
    pub fn load(path: &Path) -> StorageResult<Self> {
        if !path.exists() {
            return Ok(Self::new());
        }

        let json = fs::read_to_string(path)?;
        let router: RouterIndex = serde_json::from_str(&json)?;
        Ok(router)
    }

    /// Save router index to disk atomically.
    pub fn save(&self, path: &Path) -> StorageResult<()> {
        let json = serde_json::to_string_pretty(self)?;
        let tmp_path = path.with_extension("tmp");
        fs::write(&tmp_path, &json)?;
        fs::rename(&tmp_path, path)?;
        Ok(())
    }
}

/// Helper struct for max-heap ordering (we want minimum distances).
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
        // Max-heap: larger distance = higher priority (to be evicted first)
        self.distance.partial_cmp(&other.distance).unwrap_or(Ordering::Equal)
    }
}

/// Compute centroid of a set of vectors.
pub fn compute_centroid(vectors: &[Vec<f32>]) -> Vec<f32> {
    if vectors.is_empty() {
        return Vec::new();
    }

    let dim = vectors[0].len();
    let n = vectors.len() as f64;
    let mut centroid = vec![0.0f64; dim];

    for v in vectors {
        for (i, &val) in v.iter().enumerate() {
            centroid[i] += val as f64;
        }
    }

    centroid.into_iter().map(|c| (c / n) as f32).collect()
}

/// Compute centroid from raw f32 slice (contiguous vector data).
pub fn compute_centroid_from_slice(data: &[f32], dim: usize) -> Vec<f32> {
    if data.is_empty() || dim == 0 {
        return Vec::new();
    }

    let num_vectors = data.len() / dim;
    let n = num_vectors as f64;
    let mut centroid = vec![0.0f64; dim];

    for chunk in data.chunks_exact(dim) {
        for (i, &val) in chunk.iter().enumerate() {
            centroid[i] += val as f64;
        }
    }

    centroid.into_iter().map(|c| (c / n) as f32).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_router_index_basic() {
        let mut router = RouterIndex::new();

        router.upsert(RouterEntry {
            segment_id: "seg1".to_string(),
            level: 0,
            segment_centroid: vec![1.0, 0.0, 0.0],
            num_vectors: 1000,
        });

        router.upsert(RouterEntry {
            segment_id: "seg2".to_string(),
            level: 0,
            segment_centroid: vec![0.0, 1.0, 0.0],
            num_vectors: 1000,
        });

        assert_eq!(router.entries.len(), 2);
        assert!(router.contains("seg1"));
        assert!(router.contains("seg2"));
    }

    #[test]
    fn test_router_select_segments() {
        let mut router = RouterIndex::new();

        // Segment 1: centroid at [1, 0, 0]
        router.upsert(RouterEntry {
            segment_id: "seg1".to_string(),
            level: 0,
            segment_centroid: vec![1.0, 0.0, 0.0],
            num_vectors: 1000,
        });

        // Segment 2: centroid at [0, 1, 0]
        router.upsert(RouterEntry {
            segment_id: "seg2".to_string(),
            level: 0,
            segment_centroid: vec![0.0, 1.0, 0.0],
            num_vectors: 1000,
        });

        // Segment 3: centroid at [0, 0, 1]
        router.upsert(RouterEntry {
            segment_id: "seg3".to_string(),
            level: 0,
            segment_centroid: vec![0.0, 0.0, 1.0],
            num_vectors: 1000,
        });

        // Query close to seg1
        let query = vec![0.9, 0.1, 0.0];
        let selected = router.select_segments(&query, Metric::L2, 2);

        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0], "seg1"); // Closest
    }

    #[test]
    fn test_compute_centroid() {
        let vectors = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];

        let centroid = compute_centroid(&vectors);

        // Mean: [(1+0+1)/3, (0+1+1)/3] = [0.667, 0.667]
        assert!((centroid[0] - 0.6666667).abs() < 0.001);
        assert!((centroid[1] - 0.6666667).abs() < 0.001);
    }

    #[test]
    fn test_router_remove() {
        let mut router = RouterIndex::new();

        router.upsert(RouterEntry {
            segment_id: "seg1".to_string(),
            level: 0,
            segment_centroid: vec![1.0],
            num_vectors: 100,
        });

        router.upsert(RouterEntry {
            segment_id: "seg2".to_string(),
            level: 0,
            segment_centroid: vec![2.0],
            num_vectors: 200,
        });

        router.remove("seg1");

        assert!(!router.contains("seg1"));
        assert!(router.contains("seg2"));
        assert_eq!(router.entries.len(), 1);
    }

    #[test]
    fn test_router_upsert_replaces() {
        let mut router = RouterIndex::new();

        router.upsert(RouterEntry {
            segment_id: "seg1".to_string(),
            level: 0,
            segment_centroid: vec![1.0],
            num_vectors: 100,
        });

        // Update same segment
        router.upsert(RouterEntry {
            segment_id: "seg1".to_string(),
            level: 1,
            segment_centroid: vec![2.0],
            num_vectors: 200,
        });

        assert_eq!(router.entries.len(), 1);
        let entry = router.get("seg1").unwrap();
        assert_eq!(entry.level, 1);
        assert_eq!(entry.num_vectors, 200);
    }
}
