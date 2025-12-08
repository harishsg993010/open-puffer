//! Query execution engine.

use crate::error::{QueryError, QueryResult};
use puffer_core::{distance, Metric, VectorId, VectorRecord};
use puffer_index::{
    kmeans::{build_cluster_data, kmeans, KMeansConfig},
    search::{merge_results, search_segment, SearchResult},
};
use puffer_storage::{Catalog, Collection, LoadedSegment, SegmentBuilder};
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

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
        let num_clusters = coll
            .meta
            .config
            .num_clusters
            .min(coll.staging_vectors.len());

        tracing::info!(
            "Flushing {} vectors to segment with {} clusters",
            coll.staging_vectors.len(),
            num_clusters
        );

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

        // Update metadata
        coll.meta.total_vectors += coll.staging_vectors.len();
        coll.meta.segments.push(segment_name);
        coll.meta.staging_count = 0;

        // Clear staging buffer
        coll.staging_vectors.clear();
        coll.staging_ids.clear();
        coll.staging_payloads.clear();
        coll.clear_staging_file()?;

        // Invalidate segment cache for this collection
        let mut cache = self.segment_cache.write().unwrap();
        cache.remove(&coll.meta.config.name);

        Ok(())
    }

    /// Force flush staging buffer (even if below threshold).
    pub fn force_flush(&self, collection_name: &str) -> QueryResult<()> {
        let coll_arc = self.catalog.get_collection(collection_name)?;
        let mut coll = coll_arc.write().unwrap();
        self.flush_staging_to_segment(&mut coll)?;
        coll.save_meta()?;
        Ok(())
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

        // Load and search segments
        let segments = self.load_segments(&coll)?;

        // Parallel search across segments
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
        let name = &coll.meta.config.name;

        // Check cache
        {
            let cache = self.segment_cache.read().unwrap();
            if let Some(segments) = cache.get(name) {
                if segments.len() == coll.meta.segments.len() {
                    return Ok(segments.clone());
                }
            }
        }

        // Load segments
        let segments: Vec<Arc<LoadedSegment>> = coll
            .meta
            .segments
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

    /// Get collection statistics.
    pub fn stats(&self, collection_name: &str) -> QueryResult<CollectionStats> {
        let coll_arc = self.catalog.get_collection(collection_name)?;
        let coll = coll_arc.read().unwrap();

        let segments = self.load_segments(&coll)?;

        let segment_stats: Vec<SegmentStats> = segments
            .iter()
            .zip(coll.meta.segments.iter())
            .map(|(seg, name)| SegmentStats {
                name: name.clone(),
                num_vectors: seg.num_vectors(),
                num_clusters: seg.clusters.len(),
            })
            .collect();

        Ok(CollectionStats {
            name: coll.meta.config.name.clone(),
            dimension: coll.meta.config.dimension,
            metric: coll.meta.config.metric,
            total_vectors: coll.meta.total_vectors + coll.staging_vectors.len(),
            staging_vectors: coll.staging_vectors.len(),
            segments: segment_stats,
        })
    }
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
}

/// Segment statistics.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SegmentStats {
    pub name: String,
    pub num_vectors: usize,
    pub num_clusters: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use puffer_storage::CollectionConfig;
    use tempfile::tempdir;

    fn create_test_engine() -> (tempfile::TempDir, Arc<QueryEngine>) {
        let dir = tempdir().unwrap();
        let catalog = Arc::new(Catalog::open(dir.path()).unwrap());
        let engine = Arc::new(QueryEngine::new(catalog.clone()));

        // Create a test collection
        catalog
            .create_collection(CollectionConfig {
                name: "test".to_string(),
                dimension: 4,
                metric: Metric::L2,
                staging_threshold: 100,
                num_clusters: 4,
            })
            .unwrap();

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
}
