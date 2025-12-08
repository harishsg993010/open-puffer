//! Query execution engine.

use crate::compaction::{compact_until_done, needs_compaction, CompactionConfig};
use crate::error::{QueryError, QueryResult};
use puffer_core::{distance, Metric, VectorId, VectorRecord};
use puffer_index::{
    kmeans::{build_cluster_data, kmeans, KMeansConfig},
    search::{merge_results, search_segment, SearchResult},
};
use puffer_storage::{
    compute_centroid, Catalog, Collection, LoadedSegment, RouterEntry, SegmentBuilder,
};
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Default number of top segments to search if not configured.
const DEFAULT_ROUTER_TOP_M: usize = 5;

/// Result of a compaction operation.
#[derive(Debug, Clone, serde::Serialize)]
pub struct CompactionResult {
    pub segments_compacted: usize,
    pub l0_segments_before: usize,
    pub l0_segments_after: usize,
    pub total_segments_before: usize,
    pub total_segments_after: usize,
}

/// Query engine for searching vectors.
pub struct QueryEngine {
    catalog: Arc<Catalog>,
    /// Cache of loaded segments per collection
    segment_cache: RwLock<HashMap<String, Vec<Arc<LoadedSegment>>>>,
}

impl QueryEngine {
    /// Create a new query engine.
    pub fn new(catalog: Arc<Catalog>) -> Self {
        Self {
            catalog,
            segment_cache: RwLock::new(HashMap::new()),
        }
    }

    /// Insert vectors into a collection.
    pub fn insert(
        &self,
        collection_name: &str,
        records: Vec<VectorRecord>,
    ) -> QueryResult<usize> {
        let coll_arc = self.catalog.get_collection(collection_name)?;
        let mut coll = coll_arc.write().unwrap();

        let dim = coll.meta.config.dimension;

        // Validate dimensions
        for record in &records {
            if record.vector.len() != dim {
                return Err(QueryError::DimensionMismatch {
                    expected: dim,
                    got: record.vector.len(),
                });
            }
        }

        let count = records.len();

        // Add to staging buffer
        for record in records {
            coll.staging_vectors.push(record.vector);
            coll.staging_ids.push(record.id);
            coll.staging_payloads.push(record.payload);
        }

        coll.meta.staging_count = coll.staging_vectors.len();

        // Check if we need to flush to segment
        if coll.staging_vectors.len() >= coll.meta.config.staging_threshold {
            self.flush_staging_to_segment(&mut coll)?;
        } else {
            // Save staging to disk for durability
            coll.save_staging()?;
        }

        coll.save_meta()?;

        Ok(count)
    }

    /// Flush staging buffer to a new segment.
    fn flush_staging_to_segment(&self, coll: &mut Collection) -> QueryResult<()> {
        if coll.staging_vectors.is_empty() {
            return Ok(());
        }

        let dim = coll.meta.config.dimension;
        let metric = coll.meta.config.metric;
        let num_vectors = coll.staging_vectors.len();

        // Compute optimal number of clusters based on segment size
        // Rule: nlist ≈ sqrt(N), but at least num_clusters from config
        let num_clusters = compute_optimal_clusters(num_vectors, coll.meta.config.num_clusters);

        tracing::info!(
            "Flushing {} vectors to segment with {} clusters",
            num_vectors,
            num_clusters
        );

        // Compute segment centroid BEFORE clustering (for router)
        let segment_centroid = compute_centroid(&coll.staging_vectors);

        // Run k-means clustering
        let kmeans_config = KMeansConfig {
            num_clusters,
            max_iterations: 20,
            convergence_threshold: 0.001,
            seed: None,
        };

        let kmeans_result = kmeans(&coll.staging_vectors, &kmeans_config);
        let cluster_data = build_cluster_data(&kmeans_result.centroids, &kmeans_result.assignments);

        // Build segment
        let segment_name = coll.new_segment_name();
        let segment_path = coll.segment_path(&segment_name);

        let builder = SegmentBuilder::new(dim, metric);
        builder.build(
            cluster_data,
            &coll.staging_vectors,
            &coll.staging_ids,
            &coll.staging_payloads,
            &segment_path,
        )?;

        // Update router index
        let router_entry = RouterEntry {
            segment_id: segment_name.clone(),
            level: 0, // New segments are always L0
            segment_centroid,
            num_vectors,
        };
        coll.router.upsert(router_entry);
        coll.save_router()?;

        // Update metadata with new segment info
        coll.meta.add_segment(segment_name, 0, num_vectors);
        coll.meta.total_vectors += num_vectors;
        coll.meta.staging_count = 0;

        // Clear staging buffer
        coll.staging_vectors.clear();
        coll.staging_ids.clear();
        coll.staging_payloads.clear();
        coll.clear_staging_file()?;

        // Invalidate segment cache for this collection
        {
            let mut cache = self.segment_cache.write().unwrap();
            cache.remove(&coll.meta.config.name);
        }

        // Check if compaction is needed after creating new segment
        self.maybe_compact(coll)?;

        Ok(())
    }

    /// Force flush staging buffer (even if below threshold).
    pub fn force_flush(&self, collection_name: &str) -> QueryResult<()> {
        let coll_arc = self.catalog.get_collection(collection_name)?;
        let mut coll = coll_arc.write().unwrap();
        self.flush_staging_to_segment(&mut coll)?;
        coll.save_meta()?;

        // Check if compaction is needed after flush
        self.maybe_compact(&mut coll)?;

        Ok(())
    }

    /// Trigger compaction if needed.
    fn maybe_compact(&self, coll: &mut Collection) -> QueryResult<()> {
        let config = CompactionConfig {
            l0_max_segments: coll.meta.config.l0_max_segments,
            target_segment_size: coll.meta.config.segment_target_size,
            batch_size: coll.meta.config.l0_max_segments,
        };

        if needs_compaction(coll, &config) {
            let compacted = compact_until_done(coll, &config)?;
            if compacted > 0 {
                // Invalidate segment cache after compaction
                let mut cache = self.segment_cache.write().unwrap();
                cache.remove(&coll.meta.config.name);
            }
        }

        Ok(())
    }

    /// Manually trigger compaction for a collection.
    pub fn compact(&self, collection_name: &str) -> QueryResult<CompactionResult> {
        let coll_arc = self.catalog.get_collection(collection_name)?;
        let mut coll = coll_arc.write().unwrap();

        let config = CompactionConfig {
            l0_max_segments: coll.meta.config.l0_max_segments,
            target_segment_size: coll.meta.config.segment_target_size,
            batch_size: coll.meta.config.l0_max_segments,
        };

        let l0_before = coll.meta.l0_segment_count();
        let total_before = coll.meta.get_segment_names().len();

        let segments_compacted = compact_until_done(&mut coll, &config)?;

        let l0_after = coll.meta.l0_segment_count();
        let total_after = coll.meta.get_segment_names().len();

        // Invalidate segment cache after compaction
        if segments_compacted > 0 {
            let mut cache = self.segment_cache.write().unwrap();
            cache.remove(&coll.meta.config.name);
        }

        Ok(CompactionResult {
            segments_compacted,
            l0_segments_before: l0_before,
            l0_segments_after: l0_after,
            total_segments_before: total_before,
            total_segments_after: total_after,
        })
    }

    /// Search for similar vectors.
    pub fn search(
        &self,
        collection_name: &str,
        query: &[f32],
        k: usize,
        nprobe: usize,
        include_payload: bool,
    ) -> QueryResult<Vec<SearchResult>> {
        self.search_with_options(collection_name, query, k, nprobe, include_payload, None)
    }

    /// Search with optional router_top_m override.
    pub fn search_with_options(
        &self,
        collection_name: &str,
        query: &[f32],
        k: usize,
        nprobe: usize,
        include_payload: bool,
        router_top_m_override: Option<usize>,
    ) -> QueryResult<Vec<SearchResult>> {
        let coll_arc = self.catalog.get_collection(collection_name)?;
        let coll = coll_arc.read().unwrap();

        let dim = coll.meta.config.dimension;
        let metric = coll.meta.config.metric;

        // Validate query dimension
        if query.len() != dim {
            return Err(QueryError::DimensionMismatch {
                expected: dim,
                got: query.len(),
            });
        }

        let mut all_results = Vec::new();

        // Search staging buffer (brute force)
        if !coll.staging_vectors.is_empty() {
            let staging_results = self.search_staging(
                &coll.staging_vectors,
                &coll.staging_ids,
                &coll.staging_payloads,
                query,
                metric,
                k,
                include_payload,
            );
            all_results.extend(staging_results);
        }

        // Determine which segments to search using router
        let router_top_m = router_top_m_override
            .or(Some(coll.meta.config.router_top_m))
            .filter(|&m| m > 0)
            .unwrap_or(DEFAULT_ROUTER_TOP_M);

        let segment_names = coll.meta.get_segment_names();
        let total_segments = segment_names.len();

        // Use router to select top segments if we have more segments than router_top_m
        let segments_to_search: Vec<String> = if total_segments > router_top_m
            && !coll.router.entries.is_empty()
        {
            let selected = coll.router.select_segments(query, metric, router_top_m);
            tracing::debug!(
                "Router selected {}/{} segments for search",
                selected.len(),
                total_segments
            );
            selected
        } else {
            // Search all segments if few segments or router is empty
            if coll.router.entries.is_empty() && total_segments > 0 {
                tracing::warn!(
                    "Router index empty, searching all {} segments (rebuild router recommended)",
                    total_segments
                );
            }
            segment_names
        };

        // Load selected segments
        let segments = self.load_segments_by_name(&coll, &segments_to_search)?;

        // Parallel search across selected segments
        let segment_results: Vec<Vec<SearchResult>> = segments
            .par_iter()
            .map(|seg| search_segment(seg, query, k, nprobe, include_payload))
            .collect();

        for results in segment_results {
            all_results.extend(results);
        }

        // Merge and return top k
        Ok(merge_results(all_results, k))
    }

    /// Search staging buffer using brute force.
    fn search_staging(
        &self,
        vectors: &[Vec<f32>],
        ids: &[VectorId],
        payloads: &[Option<serde_json::Value>],
        query: &[f32],
        metric: Metric,
        k: usize,
        include_payload: bool,
    ) -> Vec<SearchResult> {
        let mut distances: Vec<(usize, f32)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i, distance::distance(query, v, metric)))
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        distances
            .into_iter()
            .take(k)
            .map(|(i, dist)| {
                let payload = if include_payload {
                    payloads.get(i).cloned().flatten()
                } else {
                    None
                };
                SearchResult {
                    id: ids[i].clone(),
                    distance: dist,
                    payload,
                }
            })
            .collect()
    }

    /// Load segments for a collection (with caching).
    fn load_segments(&self, coll: &Collection) -> QueryResult<Vec<Arc<LoadedSegment>>> {
        let segment_names = coll.meta.get_segment_names();
        self.load_segments_by_name(coll, &segment_names)
    }

    /// Load specific segments by name.
    fn load_segments_by_name(
        &self,
        coll: &Collection,
        _segment_names: &[String],
    ) -> QueryResult<Vec<Arc<LoadedSegment>>> {
        let name = &coll.meta.config.name;

        // Check cache for full segment list
        {
            let cache = self.segment_cache.read().unwrap();
            if let Some(cached_segments) = cache.get(name) {
                // If cache has all segments, use it
                if cached_segments.len() == coll.meta.get_segment_names().len() {
                    return Ok(cached_segments.clone());
                }
            }
        }

        // Load all segments (cache miss or stale cache)
        let all_segment_names = coll.meta.get_segment_names();
        let segments: Vec<Arc<LoadedSegment>> = all_segment_names
            .iter()
            .filter_map(|seg_name| {
                let path = coll.segment_path(seg_name);
                match LoadedSegment::open(&path) {
                    Ok(seg) => Some(Arc::new(seg)),
                    Err(e) => {
                        tracing::error!("Failed to load segment {}: {}", seg_name, e);
                        None
                    }
                }
            })
            .collect();

        // Update cache
        {
            let mut cache = self.segment_cache.write().unwrap();
            cache.insert(name.clone(), segments.clone());
        }

        Ok(segments)
    }

    /// Rebuild router index from existing segments.
    pub fn rebuild_router(&self, collection_name: &str) -> QueryResult<usize> {
        let coll_arc = self.catalog.get_collection(collection_name)?;
        let mut coll = coll_arc.write().unwrap();

        let segment_names = coll.meta.get_segment_names();
        let mut rebuilt_count = 0;

        for seg_name in &segment_names {
            let path = coll.segment_path(seg_name);
            match LoadedSegment::open(&path) {
                Ok(seg) => {
                    // Compute centroid from segment vectors
                    let num_vectors = seg.num_vectors();
                    let dim = seg.dimension();

                    // Accumulate all vectors to compute centroid
                    let mut centroid = vec![0.0f64; dim];
                    for i in 0..num_vectors {
                        let vec = seg.get_vector(i);
                        for (j, &v) in vec.iter().enumerate() {
                            centroid[j] += v as f64;
                        }
                    }

                    let segment_centroid: Vec<f32> = centroid
                        .iter()
                        .map(|&c| (c / num_vectors as f64) as f32)
                        .collect();

                    // Get level from metadata or default to 0
                    let level = coll
                        .meta
                        .segment_metas
                        .iter()
                        .find(|s| s.name == *seg_name)
                        .map(|s| s.level)
                        .unwrap_or(0);

                    coll.router.upsert(RouterEntry {
                        segment_id: seg_name.clone(),
                        level,
                        segment_centroid,
                        num_vectors,
                    });

                    rebuilt_count += 1;
                }
                Err(e) => {
                    tracing::error!("Failed to load segment {} for router rebuild: {}", seg_name, e);
                }
            }
        }

        coll.save_router()?;
        tracing::info!("Rebuilt router index with {} segments", rebuilt_count);

        // Invalidate segment cache
        let mut cache = self.segment_cache.write().unwrap();
        cache.remove(collection_name);

        Ok(rebuilt_count)
    }

    /// Get collection statistics.
    pub fn stats(&self, collection_name: &str) -> QueryResult<CollectionStats> {
        let coll_arc = self.catalog.get_collection(collection_name)?;
        let coll = coll_arc.read().unwrap();

        let segments = self.load_segments(&coll)?;

        let segment_stats: Vec<SegmentStats> = segments
            .iter()
            .zip(coll.meta.get_segment_names().iter())
            .map(|(seg, name)| {
                let level = coll
                    .meta
                    .segment_metas
                    .iter()
                    .find(|s| s.name == *name)
                    .map(|s| s.level)
                    .unwrap_or(0);

                SegmentStats {
                    name: name.clone(),
                    num_vectors: seg.num_vectors(),
                    num_clusters: seg.clusters.len(),
                    level,
                }
            })
            .collect();

        Ok(CollectionStats {
            name: coll.meta.config.name.clone(),
            dimension: coll.meta.config.dimension,
            metric: coll.meta.config.metric,
            total_vectors: coll.meta.total_vectors + coll.staging_vectors.len(),
            staging_vectors: coll.staging_vectors.len(),
            segments: segment_stats,
            router_entries: coll.router.entries.len(),
        })
    }
}

/// Compute optimal number of clusters based on segment size.
/// Rule: nlist ≈ sqrt(N), with min/max bounds.
fn compute_optimal_clusters(num_vectors: usize, min_clusters: usize) -> usize {
    let sqrt_n = (num_vectors as f64).sqrt() as usize;
    // Use sqrt(N) but ensure at least min_clusters and at most num_vectors
    sqrt_n.max(min_clusters).min(num_vectors)
}

/// Collection statistics.
#[derive(Debug, Clone, serde::Serialize)]
pub struct CollectionStats {
    pub name: String,
    pub dimension: usize,
    pub metric: Metric,
    pub total_vectors: usize,
    pub staging_vectors: usize,
    pub segments: Vec<SegmentStats>,
    pub router_entries: usize,
}

/// Segment statistics.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SegmentStats {
    pub name: String,
    pub num_vectors: usize,
    pub num_clusters: usize,
    pub level: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use puffer_storage::CollectionConfig;
    use tempfile::tempdir;

    fn test_config(name: &str) -> CollectionConfig {
        CollectionConfig {
            name: name.to_string(),
            dimension: 4,
            metric: Metric::L2,
            staging_threshold: 100,
            num_clusters: 4,
            router_top_m: 5,
            l0_max_segments: 10,
            segment_target_size: 100_000,
        }
    }

    fn create_test_engine() -> (tempfile::TempDir, Arc<QueryEngine>) {
        let dir = tempdir().unwrap();
        let catalog = Arc::new(Catalog::open(dir.path()).unwrap());
        let engine = Arc::new(QueryEngine::new(catalog.clone()));

        // Create a test collection
        catalog.create_collection(test_config("test")).unwrap();

        (dir, engine)
    }

    #[test]
    fn test_insert_and_search() {
        let (_dir, engine) = create_test_engine();

        // Insert some vectors
        let records: Vec<VectorRecord> = (0..10)
            .map(|i| {
                VectorRecord::new(
                    format!("vec_{}", i),
                    vec![i as f32, 0.0, 0.0, 0.0],
                )
            })
            .collect();

        engine.insert("test", records).unwrap();

        // Search for vector closest to [5, 0, 0, 0]
        let results = engine
            .search("test", &[5.0, 0.0, 0.0, 0.0], 3, 4, false)
            .unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].id.as_str(), "vec_5");
    }

    #[test]
    fn test_flush_to_segment() {
        let (_dir, engine) = create_test_engine();

        // Insert enough vectors to trigger flush (all 150 will be flushed)
        let records: Vec<VectorRecord> = (0..150)
            .map(|i| {
                VectorRecord::new(
                    format!("vec_{}", i),
                    vec![i as f32, 0.0, 0.0, 0.0],
                )
            })
            .collect();

        engine.insert("test", records).unwrap();

        // Check stats - all 150 vectors should be in the segment
        let stats = engine.stats("test").unwrap();
        assert_eq!(stats.segments.len(), 1);
        assert_eq!(stats.staging_vectors, 0); // All flushed to segment
        assert_eq!(stats.total_vectors, 150);
        assert_eq!(stats.router_entries, 1); // Router should have entry for segment
    }

    #[test]
    fn test_search_with_payload() {
        let (_dir, engine) = create_test_engine();

        let records: Vec<VectorRecord> = (0..5)
            .map(|i| {
                VectorRecord::new(
                    format!("vec_{}", i),
                    vec![i as f32, 0.0, 0.0, 0.0],
                )
                .with_payload(serde_json::json!({"index": i}))
            })
            .collect();

        engine.insert("test", records).unwrap();

        let results = engine
            .search("test", &[2.0, 0.0, 0.0, 0.0], 1, 4, true)
            .unwrap();

        assert_eq!(results.len(), 1);
        assert!(results[0].payload.is_some());
        assert_eq!(results[0].payload.as_ref().unwrap()["index"], 2);
    }

    #[test]
    fn test_rebuild_router() {
        let (_dir, engine) = create_test_engine();

        // Insert vectors to create segments
        let records: Vec<VectorRecord> = (0..150)
            .map(|i| {
                VectorRecord::new(
                    format!("vec_{}", i),
                    vec![i as f32, 0.0, 0.0, 0.0],
                )
            })
            .collect();

        engine.insert("test", records).unwrap();

        // Rebuild router
        let count = engine.rebuild_router("test").unwrap();
        assert_eq!(count, 1);

        // Verify stats show router entry
        let stats = engine.stats("test").unwrap();
        assert_eq!(stats.router_entries, 1);
    }
}
