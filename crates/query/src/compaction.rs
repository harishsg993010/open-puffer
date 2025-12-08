//! LSM-style segment compaction.
//!
//! Reduces the number of segments by merging multiple small L0 segments
//! into larger L1 segments.

use crate::error::{QueryError, QueryResult};
use puffer_core::VectorId;
#[cfg(test)]
use puffer_core::Metric;
use puffer_index::kmeans::{build_cluster_data, kmeans, KMeansConfig};
use puffer_storage::{
    compute_centroid, Collection, LoadedSegment, RouterEntry, SegmentBuilder,
};
use std::fs;

/// Compaction configuration.
#[derive(Debug, Clone)]
pub struct CompactionConfig {
    /// Maximum number of L0 segments before triggering compaction.
    pub l0_max_segments: usize,
    /// Target size for merged segments.
    pub target_segment_size: usize,
    /// Number of L0 segments to merge in one compaction.
    pub batch_size: usize,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            l0_max_segments: 10,
            target_segment_size: 100_000,
            batch_size: 10,
        }
    }
}

/// Check if compaction is needed for a collection.
pub fn needs_compaction(coll: &Collection, config: &CompactionConfig) -> bool {
    coll.meta.l0_segment_count() >= config.l0_max_segments
}

/// Perform compaction on a collection.
///
/// Merges multiple L0 segments into a single L1 segment.
/// Returns the number of segments compacted.
pub fn compact_collection(coll: &mut Collection, config: &CompactionConfig) -> QueryResult<usize> {
    let l0_segments = coll.meta.get_l0_segments();

    if l0_segments.len() < config.batch_size {
        tracing::debug!(
            "Not enough L0 segments for compaction: {} < {}",
            l0_segments.len(),
            config.batch_size
        );
        return Ok(0);
    }

    // Select batch of L0 segments to compact
    let segments_to_compact: Vec<String> = l0_segments
        .iter()
        .take(config.batch_size)
        .map(|s| s.name.clone())
        .collect();

    if segments_to_compact.is_empty() {
        return Ok(0);
    }

    tracing::info!(
        "Starting compaction of {} L0 segments",
        segments_to_compact.len()
    );

    // Load all vectors from segments to compact
    let mut all_vectors: Vec<Vec<f32>> = Vec::new();
    let mut all_ids: Vec<VectorId> = Vec::new();
    let mut all_payloads: Vec<Option<serde_json::Value>> = Vec::new();

    for seg_name in &segments_to_compact {
        let path = coll.segment_path(seg_name);
        let segment = LoadedSegment::open(&path).map_err(|e| {
            QueryError::Storage(puffer_storage::StorageError::InvalidSegment(
                format!("Failed to load segment {}: {}", seg_name, e)
            ))
        })?;

        let num_vectors = segment.num_vectors();

        for i in 0..num_vectors {
            let vec = segment.get_vector(i);
            all_vectors.push(vec.to_vec());

            let id = segment.get_id(i).map_err(|e| {
                QueryError::Storage(puffer_storage::StorageError::InvalidId(
                    format!("Failed to get ID at index {}: {}", i, e)
                ))
            })?;
            all_ids.push(id);

            let payload = segment.get_payload(i).map_err(|e| {
                QueryError::Storage(puffer_storage::StorageError::Json(
                    serde_json::Error::io(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("Failed to get payload at index {}: {}", i, e)
                    ))
                ))
            })?;
            all_payloads.push(payload);
        }
    }

    if all_vectors.is_empty() {
        tracing::warn!("No vectors found in segments to compact");
        return Ok(0);
    }

    let total_vectors = all_vectors.len();
    let dim = coll.meta.config.dimension;
    let metric = coll.meta.config.metric;

    tracing::info!(
        "Compacting {} vectors from {} segments into L1 segment",
        total_vectors,
        segments_to_compact.len()
    );

    // Compute optimal clusters for merged segment
    // For larger segments, use sqrt(N) clusters
    let num_clusters = ((total_vectors as f64).sqrt() as usize)
        .max(coll.meta.config.num_clusters)
        .min(total_vectors);

    // Compute segment centroid for router
    let segment_centroid = compute_centroid(&all_vectors);

    // Run k-means clustering
    let kmeans_config = KMeansConfig {
        num_clusters,
        max_iterations: 20,
        convergence_threshold: 0.001,
        seed: None,
    };

    let kmeans_result = kmeans(&all_vectors, &kmeans_config);
    let cluster_data = build_cluster_data(&kmeans_result.centroids, &kmeans_result.assignments);

    // Build new L1 segment
    let segment_name = coll.new_segment_name();
    let segment_path = coll.segment_path(&segment_name);

    let builder = SegmentBuilder::new(dim, metric);
    builder.build(
        cluster_data,
        &all_vectors,
        &all_ids,
        &all_payloads,
        &segment_path,
    )?;

    tracing::info!(
        "Built L1 segment {} with {} vectors and {} clusters",
        segment_name,
        total_vectors,
        num_clusters
    );

    // Update router for new segment
    let router_entry = RouterEntry {
        segment_id: segment_name.clone(),
        level: 1, // L1 segment
        segment_centroid,
        num_vectors: total_vectors,
    };
    coll.router.upsert(router_entry);

    // Remove old segments from router
    for seg_name in &segments_to_compact {
        coll.router.remove(seg_name);
    }
    coll.save_router()?;

    // Update metadata
    coll.meta.add_segment(segment_name, 1, total_vectors);
    coll.meta.remove_segments(&segments_to_compact);

    // Save metadata before deleting old segments
    coll.save_meta()?;

    // Delete old segment files
    for seg_name in &segments_to_compact {
        let old_path = coll.segment_path(seg_name);
        if old_path.exists() {
            if let Err(e) = fs::remove_file(&old_path) {
                tracing::warn!("Failed to delete old segment {}: {}", seg_name, e);
            }
        }
    }

    let compacted_count = segments_to_compact.len();
    tracing::info!(
        "Compaction complete: merged {} L0 segments into 1 L1 segment",
        compacted_count
    );

    Ok(compacted_count)
}

/// Perform multiple rounds of compaction until no more compaction is needed.
pub fn compact_until_done(coll: &mut Collection, config: &CompactionConfig) -> QueryResult<usize> {
    let mut total_compacted = 0;

    loop {
        if !needs_compaction(coll, config) {
            break;
        }

        let compacted = compact_collection(coll, config)?;
        if compacted == 0 {
            break;
        }

        total_compacted += compacted;
    }

    Ok(total_compacted)
}

#[cfg(test)]
mod tests {
    use super::*;
    use puffer_storage::{Catalog, CollectionConfig};
    use tempfile::tempdir;

    fn test_config(name: &str) -> CollectionConfig {
        CollectionConfig {
            name: name.to_string(),
            dimension: 4,
            metric: Metric::L2,
            staging_threshold: 50, // Small threshold to create more segments
            num_clusters: 4,
            router_top_m: 5,
            l0_max_segments: 3,
            segment_target_size: 100_000,
        }
    }

    #[test]
    fn test_needs_compaction() {
        let dir = tempdir().unwrap();
        let catalog = Catalog::open(dir.path()).unwrap();
        catalog.create_collection(test_config("compact_test")).unwrap();

        let coll_arc = catalog.get_collection("compact_test").unwrap();
        let coll = coll_arc.read().unwrap();

        let config = CompactionConfig {
            l0_max_segments: 3,
            ..Default::default()
        };

        // Initially no compaction needed
        assert!(!needs_compaction(&coll, &config));
    }
}
