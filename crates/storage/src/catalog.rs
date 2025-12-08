//! Catalog management for collections.

use crate::error::{StorageError, StorageResult};
use puffer_core::Metric;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

/// Configuration for a collection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionConfig {
    /// Name of the collection.
    pub name: String,
    /// Vector dimension.
    pub dimension: usize,
    /// Distance metric.
    pub metric: Metric,
    /// Maximum vectors in staging buffer before building a segment.
    #[serde(default = "default_staging_threshold")]
    pub staging_threshold: usize,
    /// Number of clusters per segment.
    #[serde(default = "default_num_clusters")]
    pub num_clusters: usize,
}

fn default_staging_threshold() -> usize {
    10_000
}

fn default_num_clusters() -> usize {
    100
}

/// Persistent collection metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionMeta {
    pub config: CollectionConfig,
    /// List of segment file names (relative to collection directory).
    pub segments: Vec<String>,
    /// Number of vectors in staging buffer.
    pub staging_count: usize,
    /// Total number of vectors across all segments.
    pub total_vectors: usize,
}

/// Runtime collection state.
pub struct Collection {
    pub meta: CollectionMeta,
    /// Path to collection directory.
    pub path: PathBuf,
    /// In-memory staging buffer for new vectors.
    pub staging_vectors: Vec<Vec<f32>>,
    pub staging_ids: Vec<puffer_core::VectorId>,
    pub staging_payloads: Vec<Option<serde_json::Value>>,
}

impl Collection {
    /// Create a new collection.
    pub fn new(config: CollectionConfig, base_path: &Path) -> StorageResult<Self> {
        let path = base_path.join(&config.name);
        fs::create_dir_all(&path)?;

        let meta = CollectionMeta {
            config,
            segments: Vec::new(),
            staging_count: 0,
            total_vectors: 0,
        };

        let collection = Self {
            meta,
            path,
            staging_vectors: Vec::new(),
            staging_ids: Vec::new(),
            staging_payloads: Vec::new(),
        };

        collection.save_meta()?;
        Ok(collection)
    }

    /// Load an existing collection.
    pub fn load(path: &Path) -> StorageResult<Self> {
        let meta_path = path.join("meta.json");
        let meta_json = fs::read_to_string(&meta_path)?;
        let meta: CollectionMeta = serde_json::from_str(&meta_json)?;

        // Load staging buffer if exists
        let staging_path = path.join("staging.json");
        let (staging_vectors, staging_ids, staging_payloads) = if staging_path.exists() {
            let staging_json = fs::read_to_string(&staging_path)?;
            let staging: StagingData = serde_json::from_str(&staging_json)?;
            (staging.vectors, staging.ids, staging.payloads)
        } else {
            (Vec::new(), Vec::new(), Vec::new())
        };

        Ok(Self {
            meta,
            path: path.to_path_buf(),
            staging_vectors,
            staging_ids,
            staging_payloads,
        })
    }

    /// Save collection metadata.
    pub fn save_meta(&self) -> StorageResult<()> {
        let meta_path = self.path.join("meta.json");
        let meta_json = serde_json::to_string_pretty(&self.meta)?;
        fs::write(&meta_path, meta_json)?;
        Ok(())
    }

    /// Save staging buffer to disk.
    pub fn save_staging(&self) -> StorageResult<()> {
        let staging_path = self.path.join("staging.json");
        let staging = StagingData {
            vectors: self.staging_vectors.clone(),
            ids: self.staging_ids.clone(),
            payloads: self.staging_payloads.clone(),
        };
        let staging_json = serde_json::to_string(&staging)?;
        fs::write(&staging_path, staging_json)?;
        Ok(())
    }

    /// Clear staging buffer file.
    pub fn clear_staging_file(&self) -> StorageResult<()> {
        let staging_path = self.path.join("staging.json");
        if staging_path.exists() {
            fs::remove_file(&staging_path)?;
        }
        Ok(())
    }

    /// Get full path to a segment file.
    pub fn segment_path(&self, segment_name: &str) -> PathBuf {
        self.path.join(segment_name)
    }

    /// Generate a new segment filename.
    pub fn new_segment_name(&self) -> String {
        let id = uuid::Uuid::new_v4();
        format!("{}.seg", id)
    }

    /// Get collection statistics.
    pub fn stats(&self) -> CollectionStats {
        CollectionStats {
            name: self.meta.config.name.clone(),
            dimension: self.meta.config.dimension,
            metric: self.meta.config.metric,
            num_segments: self.meta.segments.len(),
            total_vectors: self.meta.total_vectors,
            staging_vectors: self.staging_vectors.len(),
        }
    }
}

/// Staging buffer serialization format.
#[derive(Debug, Serialize, Deserialize)]
struct StagingData {
    vectors: Vec<Vec<f32>>,
    ids: Vec<puffer_core::VectorId>,
    payloads: Vec<Option<serde_json::Value>>,
}

/// Collection statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionStats {
    pub name: String,
    pub dimension: usize,
    pub metric: Metric,
    pub num_segments: usize,
    pub total_vectors: usize,
    pub staging_vectors: usize,
}

/// Catalog storing all collections.
#[derive(Debug, Serialize, Deserialize)]
pub struct CatalogMeta {
    pub collections: Vec<String>,
}

/// Thread-safe catalog manager.
pub struct Catalog {
    base_path: PathBuf,
    collections: RwLock<HashMap<String, Arc<RwLock<Collection>>>>,
}

impl Catalog {
    /// Open or create a catalog at the given path.
    pub fn open(path: &Path) -> StorageResult<Self> {
        fs::create_dir_all(path)?;

        let catalog_path = path.join("catalog.json");
        let collections = if catalog_path.exists() {
            let json = fs::read_to_string(&catalog_path)?;
            let meta: CatalogMeta = serde_json::from_str(&json)?;

            let mut map = HashMap::new();
            for name in meta.collections {
                let coll_path = path.join(&name);
                if coll_path.exists() {
                    match Collection::load(&coll_path) {
                        Ok(coll) => {
                            map.insert(name, Arc::new(RwLock::new(coll)));
                        }
                        Err(e) => {
                            tracing::warn!("Failed to load collection {}: {}", name, e);
                        }
                    }
                }
            }
            map
        } else {
            HashMap::new()
        };

        Ok(Self {
            base_path: path.to_path_buf(),
            collections: RwLock::new(collections),
        })
    }

    /// Save catalog metadata.
    pub fn save(&self) -> StorageResult<()> {
        let collections = self.collections.read().unwrap();
        let meta = CatalogMeta {
            collections: collections.keys().cloned().collect(),
        };
        let json = serde_json::to_string_pretty(&meta)?;
        let catalog_path = self.base_path.join("catalog.json");
        fs::write(&catalog_path, json)?;
        Ok(())
    }

    /// Create a new collection.
    pub fn create_collection(&self, config: CollectionConfig) -> StorageResult<()> {
        let mut collections = self.collections.write().unwrap();

        if collections.contains_key(&config.name) {
            return Err(StorageError::CollectionAlreadyExists(config.name));
        }

        let collection = Collection::new(config.clone(), &self.base_path)?;
        collections.insert(config.name, Arc::new(RwLock::new(collection)));
        drop(collections);

        self.save()?;
        Ok(())
    }

    /// Get a collection by name.
    pub fn get_collection(&self, name: &str) -> StorageResult<Arc<RwLock<Collection>>> {
        let collections = self.collections.read().unwrap();
        collections
            .get(name)
            .cloned()
            .ok_or_else(|| StorageError::CollectionNotFound(name.to_string()))
    }

    /// List all collections.
    pub fn list_collections(&self) -> Vec<String> {
        let collections = self.collections.read().unwrap();
        collections.keys().cloned().collect()
    }

    /// Drop a collection.
    pub fn drop_collection(&self, name: &str) -> StorageResult<()> {
        let mut collections = self.collections.write().unwrap();

        if !collections.contains_key(name) {
            return Err(StorageError::CollectionNotFound(name.to_string()));
        }

        // Remove from map
        collections.remove(name);

        // Delete directory
        let coll_path = self.base_path.join(name);
        if coll_path.exists() {
            fs::remove_dir_all(&coll_path)?;
        }

        drop(collections);
        self.save()?;
        Ok(())
    }

    /// Get base path.
    pub fn base_path(&self) -> &Path {
        &self.base_path
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_create_collection() {
        let dir = tempdir().unwrap();
        let catalog = Catalog::open(dir.path()).unwrap();

        let config = CollectionConfig {
            name: "test".to_string(),
            dimension: 128,
            metric: Metric::Cosine,
            staging_threshold: 1000,
            num_clusters: 10,
        };

        catalog.create_collection(config).unwrap();

        let collections = catalog.list_collections();
        assert_eq!(collections.len(), 1);
        assert!(collections.contains(&"test".to_string()));

        // Test duplicate creation
        let config2 = CollectionConfig {
            name: "test".to_string(),
            dimension: 128,
            metric: Metric::Cosine,
            staging_threshold: 1000,
            num_clusters: 10,
        };
        assert!(catalog.create_collection(config2).is_err());
    }

    #[test]
    fn test_catalog_persistence() {
        let dir = tempdir().unwrap();

        // Create catalog and collection
        {
            let catalog = Catalog::open(dir.path()).unwrap();
            let config = CollectionConfig {
                name: "persist_test".to_string(),
                dimension: 64,
                metric: Metric::L2,
                staging_threshold: 500,
                num_clusters: 5,
            };
            catalog.create_collection(config).unwrap();
        }

        // Reopen and verify
        {
            let catalog = Catalog::open(dir.path()).unwrap();
            let collections = catalog.list_collections();
            assert!(collections.contains(&"persist_test".to_string()));

            let coll = catalog.get_collection("persist_test").unwrap();
            let coll = coll.read().unwrap();
            assert_eq!(coll.meta.config.dimension, 64);
            assert_eq!(coll.meta.config.metric, Metric::L2);
        }
    }

    #[test]
    fn test_drop_collection() {
        let dir = tempdir().unwrap();
        let catalog = Catalog::open(dir.path()).unwrap();

        let config = CollectionConfig {
            name: "to_drop".to_string(),
            dimension: 32,
            metric: Metric::Cosine,
            staging_threshold: 100,
            num_clusters: 2,
        };

        catalog.create_collection(config).unwrap();
        assert!(catalog.list_collections().contains(&"to_drop".to_string()));

        catalog.drop_collection("to_drop").unwrap();
        assert!(!catalog.list_collections().contains(&"to_drop".to_string()));

        // Verify directory is deleted
        assert!(!dir.path().join("to_drop").exists());
    }
}
