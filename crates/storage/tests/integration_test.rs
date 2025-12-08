//! Integration tests for Puffer vector database.

use puffer_core::{Metric, VectorId, VectorRecord};
use puffer_index::{
    kmeans::{build_cluster_data, kmeans, KMeansConfig},
    search::search_segment,
};
use puffer_query::QueryEngine;
use puffer_storage::{Catalog, CollectionConfig, LoadedSegment, SegmentBuilder};
use std::sync::Arc;
use tempfile::tempdir;

fn random_vector(dim: usize) -> Vec<f32> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect()
}

#[test]
fn test_full_pipeline() {
    // Create catalog
    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).unwrap());

    // Create collection
    catalog
        .create_collection(CollectionConfig {
            name: "test_collection".to_string(),
            dimension: 128,
            metric: Metric::Cosine,
            staging_threshold: 100,
            num_clusters: 10,
        })
        .unwrap();

    // Create query engine
    let engine = QueryEngine::new(catalog.clone());

    // Insert vectors
    let records: Vec<VectorRecord> = (0..50)
        .map(|i| VectorRecord::new(format!("vec_{}", i), random_vector(128)))
        .collect();

    let inserted = engine.insert("test_collection", records).unwrap();
    assert_eq!(inserted, 50);

    // Search (in staging)
    let query = random_vector(128);
    let results = engine
        .search("test_collection", &query, 5, 10, false)
        .unwrap();

    assert_eq!(results.len(), 5);

    // Insert more to trigger flush
    let records: Vec<VectorRecord> = (50..200)
        .map(|i| VectorRecord::new(format!("vec_{}", i), random_vector(128)))
        .collect();

    engine.insert("test_collection", records).unwrap();

    // Search (across segment + staging)
    let results = engine
        .search("test_collection", &query, 10, 10, false)
        .unwrap();

    assert_eq!(results.len(), 10);

    // Check stats
    let stats = engine.stats("test_collection").unwrap();
    assert_eq!(stats.total_vectors, 200);
    assert!(!stats.segments.is_empty());
}

#[test]
fn test_segment_roundtrip() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("test.seg");

    let dim = 64;
    let num_vectors = 1000;
    let num_clusters = 10;

    // Generate vectors
    let vectors: Vec<Vec<f32>> = (0..num_vectors).map(|_| random_vector(dim)).collect();
    let ids: Vec<VectorId> = (0..num_vectors)
        .map(|i| VectorId::new(format!("v{}", i)))
        .collect();
    let payloads: Vec<Option<serde_json::Value>> = (0..num_vectors)
        .map(|i| Some(serde_json::json!({"idx": i})))
        .collect();

    // Cluster
    let config = KMeansConfig {
        num_clusters,
        max_iterations: 20,
        convergence_threshold: 0.001,
        seed: Some(42),
    };
    let result = kmeans(&vectors, &config);
    let cluster_data = build_cluster_data(&result.centroids, &result.assignments);

    // Build segment
    let builder = SegmentBuilder::new(dim, Metric::L2);
    builder
        .build(cluster_data, &vectors, &ids, &payloads, &path)
        .unwrap();

    // Load and verify
    let segment = LoadedSegment::open(&path).unwrap();
    assert_eq!(segment.num_vectors(), num_vectors);
    assert_eq!(segment.dimension(), dim);
    assert_eq!(segment.clusters.len(), num_clusters);

    // Search
    let query = &vectors[0];
    let results = search_segment(&segment, query, 5, num_clusters, true);

    assert!(!results.is_empty());
    // First result should be very close to query (might be the query itself)
    assert!(results[0].distance < 0.01);

    // Verify payload
    assert!(results[0].payload.is_some());
}

#[test]
fn test_cosine_vs_l2() {
    let dir = tempdir().unwrap();

    // Vectors where cosine and L2 give different rankings
    let vectors = vec![
        vec![1.0, 0.0, 0.0],  // unit vector along x
        vec![0.5, 0.0, 0.0],  // same direction, half magnitude
        vec![0.0, 1.0, 0.0],  // orthogonal unit vector
    ];

    let ids: Vec<VectorId> = vec!["x1".into(), "x0.5".into(), "y1".into()];
    let payloads: Vec<Option<serde_json::Value>> = vec![None, None, None];

    let cluster_data = vec![(vec![0.5, 0.33, 0.0], vec![0, 1, 2])];

    // Build L2 segment
    let l2_path = dir.path().join("l2.seg");
    let l2_builder = SegmentBuilder::new(3, Metric::L2);
    l2_builder
        .build(cluster_data.clone(), &vectors, &ids, &payloads, &l2_path)
        .unwrap();

    // Build Cosine segment
    let cos_path = dir.path().join("cos.seg");
    let cos_builder = SegmentBuilder::new(3, Metric::Cosine);
    cos_builder
        .build(cluster_data, &vectors, &ids, &payloads, &cos_path)
        .unwrap();

    // Query: [1.0, 0.0, 0.0] (same as first vector)
    let query = vec![1.0, 0.0, 0.0];

    // L2: x1 (dist=0), x0.5 (dist=0.25), y1 (dist=1.0)
    let l2_seg = LoadedSegment::open(&l2_path).unwrap();
    let l2_results = search_segment(&l2_seg, &query, 3, 1, false);
    assert_eq!(l2_results[0].id.as_str(), "x1");
    assert_eq!(l2_results[1].id.as_str(), "x0.5");

    // Cosine: x1 and x0.5 have same direction (dist=0), y1 is orthogonal (dist=1)
    let cos_seg = LoadedSegment::open(&cos_path).unwrap();
    let cos_results = search_segment(&cos_seg, &query, 3, 1, false);
    // Both x1 and x0.5 should have distance 0
    assert!(cos_results[0].distance < 0.001);
    assert!(cos_results[1].distance < 0.001);
}

#[test]
fn test_empty_collection_search() {
    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).unwrap());

    catalog
        .create_collection(CollectionConfig {
            name: "empty".to_string(),
            dimension: 32,
            metric: Metric::Cosine,
            staging_threshold: 1000,
            num_clusters: 10,
        })
        .unwrap();

    let engine = QueryEngine::new(catalog);

    let query = random_vector(32);
    let results = engine.search("empty", &query, 10, 4, false).unwrap();

    assert!(results.is_empty());
}

#[test]
fn test_dimension_mismatch() {
    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).unwrap());

    catalog
        .create_collection(CollectionConfig {
            name: "dim64".to_string(),
            dimension: 64,
            metric: Metric::L2,
            staging_threshold: 1000,
            num_clusters: 10,
        })
        .unwrap();

    let engine = QueryEngine::new(catalog);

    // Try to insert wrong dimension
    let records = vec![VectorRecord::new("wrong", random_vector(32))];
    let result = engine.insert("dim64", records);

    assert!(result.is_err());
}

#[test]
fn test_collection_not_found() {
    let dir = tempdir().unwrap();
    let catalog = Arc::new(Catalog::open(dir.path()).unwrap());
    let engine = QueryEngine::new(catalog);

    let result = engine.search("nonexistent", &random_vector(64), 10, 4, false);
    assert!(result.is_err());
}
