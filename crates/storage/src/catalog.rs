//! Catalog management for collections.

use crate::error::{StorageError, StorageResult};
use crate::router::{RouterEntry, RouterIndex};
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
    /// Number of clusters per segment (for L0 segments).
    /// For larger segments, this will be scaled based on sqrt(num_vectors).
    #[serde(default = "default_num_clusters")]
    pub num_clusters: usize,
    /// Number of top segments to search via router (0 = search all).
    #[serde(default = "default_router_top_m")]
    pub router_top_m: usize,
    /// Maximum L0 segments before triggering compaction.
    #[serde(default = "default_l0_max_segments")]
    pub l0_max_segments: usize,
    /// Target size for L1 (compacted) segments.
    #[serde(default = "default_segment_target_size")]
    pub segment_target_size: usize,
}

fn default_staging_threshold() -> usize {
    10_000
}

fn default_num_clusters() -> usize {
    100
}

fn default_router_top_m() -> usize {
    5
}

fn default_l0_max_segments() -> usize {
    10
}

fn default_segment_target_size() -> usize {
    100_000
}

/// Metadata for a single segment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentMeta {
    /// Segment filename.
    pub name: String,
    /// LSM level (0 = small/new, 1 = compacted, etc.)
    pub level: u32,
    /// Number of vectors in this segment.
    pub num_vectors: usize,
}

/// Persistent collection metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionMeta {
    pub config: CollectionConfig,
    /// List of segment metadata (replaces old Vec<String>).
    /// For backwards compatibility, we support both formats.
    #[serde(default)]
    pub segment_metas: Vec<SegmentMeta>,
    /// Legacy: List of segment file names (for backwards compatibility).
    #[serde(default)]
    pub segments: Vec<String>,
    /// Number of vectors in staging buffer.
    pub staging_count: usize,
    /// Total number of vectors across all segments.
    pub total_vectors: usize,
}

impl CollectionMeta {
    /// Get all segment names (handles both old and new format).
    pub fn get_segment_names(&self) -> Vec<String> {
        if !self.segment_metas.is_empty() {
            self.segment_metas.iter().map(|s| s.name.clone()).collect()
        } else {
            self.segments.clone()
        }
    }

    /// Add a new segment.
    pub fn add_segment(&mut self, name: String, level: u32, num_vectors: usize) {
        self.segment_metas.push(SegmentMeta {
            name: name.clone(),
            level,
            num_vectors,
        });
        // Also update legacy field for compatibility
        self.segments.push(name);
    }

    /// Remove segments by name.
    pub fn remove_segments(&mut self, names: &[String]) {
        self.segment_metas.retain(|s| !names.contains(&s.name));
        self.segments.retain(|s| !names.contains(s));
    }

    /// Get L0 segments (level 0).
    pub fn get_l0_segments(&self) -> Vec<&SegmentMeta> {
        self.segment_metas.iter().filter(|s| s.level == 0).collect()
    }

    /// Count L0 segments.
    pub fn l0_segment_count(&self) -> usize {
        self.segment_metas.iter().filter(|s| s.level == 0).count()
    }
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
    /// Router index for segment selection.
    pub router: RouterIndex,
}

impl Collection {
    /// Create a new collection.
    pub fn new(config: CollectionConfig, base_path: &Path) -> StorageResult<Self> {
        let path = base_path.join(&config.name);
        fs::create_dir_all(&path)?;

        let meta = CollectionMeta {
            config,
            segment_metas: Vec::new(),
            segments: Vec::new(),
            staging_count: 0,
            total_vectors: 0,
        };

        let router = RouterIndex::new();

        let collection = Self {
            meta,
            path,
            staging_vectors: Vec::new(),
            staging_ids: Vec::new(),
            staging_payloads: Vec::new(),
            router,
        };

        collection.save_meta()?;
        collection.save_router()?;
        Ok(collection)
    }

    /// Load an existing collection.
    pub fn load(path: &Path) -> StorageResult<Self> {
        let meta_path = path.join("meta.json");
        let meta_json = fs::read_to_string(&meta_path)?;
        let mut meta: CollectionMeta = serde_json::from_str(&meta_json)?;

        // Migrate old format to new format if needed
        if meta.segment_metas.is_empty() && !meta.segments.is_empty() {
            tracing::info!("Migrating {} legacy segments to new format", meta.segments.len());
            for name in &meta.segments {
                meta.segment_metas.push(SegmentMeta {
                    name: name.clone(),
                    level: 0,
                    num_vectors: 0, // Will be populated on next operation
                });
            }
        }

        // Load staging buffer if exists
        let staging_path = path.join("staging.json");
        let (staging_vectors, staging_ids, staging_payloads) = if staging_path.exists() {
            let staging_json = fs::read_to_string(&staging_path)?;
            let staging: StagingData = serde_json::from_str(&staging_json)?;
            (staging.vectors, staging.ids, staging.payloads)
        } else {
            (Vec::new(), Vec::new(), Vec::new())
        };

        // Load router index
        let router_path = path.join("router_index.json");
        let router = RouterIndex::load(&router_path).unwrap_or_else(|_| {
            tracing::warn!("Router index not found or corrupted, creating new one");
            RouterIndex::new()
        });

        Ok(Self {
            meta,
            path: path.to_path_buf(),
            staging_vectors,
            staging_ids,
            staging_payloads,
            router,
        })
    }

    /// Save collection metadata.
    pub fn save_meta(&self) -> StorageResult<()> {
        let meta_path = self.path.join("meta.json");
        let meta_json = serde_json::to_string_pretty(&self.meta)?;
        fs::write(&meta_path, meta_json)?;
        Ok(())
    }

    /// Save router index to disk.
    pub fn save_router(&self) -> StorageResult<()> {
        let router_path = self.path.join("router_index.json");
        self.router.save(&router_path)
    }

    /// Update router entry for a segment.
    pub fn update_router_entry(&mut self, entry: RouterEntry) -> StorageResult<()> {
        self.router.upsert(entry);
        self.save_router()
    }

    /// Remove segment from router.
    pub fn remove_from_router(&mut self, segment_id: &str) -> StorageResult<()> {
        self.router.remove(segment_id);
        self.save_router()
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

    /// Get path to router index file.
    pub fn router_path(&self) -> PathBuf {
        self.path.join("router_index.json")
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
            num_segments: self.meta.get_segment_names().len(),
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

    fn test_config(name: &str) -> CollectionConfig {
        CollectionConfig {
            name: name.to_string(),
            dimension: 128,
            metric: Metric::Cosine,
            staging_threshold: 1000,
            num_clusters: 10,
            router_top_m: 5,
            l0_max_segments: 10,
            segment_target_size: 100_000,
        }
    }

    #[test]
    fn test_create_collection() {
        let dir = tempdir().unwrap();
        let catalog = Catalog::open(dir.path()).unwrap();

        let config = test_config("test");
        catalog.create_collection(config).unwrap();

        let collections = catalog.list_collections();
        assert_eq!(collections.len(), 1);
        assert!(collections.contains(&"test".to_string()));

        // Test duplicate creation
        let config2 = test_config("test");
        assert!(catalog.create_collection(config2).is_err());
    }

    #[test]
    fn test_catalog_persistence() {
        let dir = tempdir().unwrap();

        // Create catalog and collection
        {
            let catalog = Catalog::open(dir.path()).unwrap();
            let mut config = test_config("persist_test");
            config.dimension = 64;
            config.metric = Metric::L2;
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
            // Verify router was loaded
            assert!(coll.router.entries.is_empty());
        }
    }

    #[test]
    fn test_drop_collection() {
        let dir = tempdir().unwrap();
        let catalog = Catalog::open(dir.path()).unwrap();

        let config = test_config("to_drop");
        catalog.create_collection(config).unwrap();
        assert!(catalog.list_collections().contains(&"to_drop".to_string()));

        catalog.drop_collection("to_drop").unwrap();
        assert!(!catalog.list_collections().contains(&"to_drop".to_string()));

        // Verify directory is deleted
        assert!(!dir.path().join("to_drop").exists());
    }

    #[test]
    fn test_router_persistence() {
        let dir = tempdir().unwrap();

        // Create catalog and collection, add router entry
        {
            let catalog = Catalog::open(dir.path()).unwrap();
            let config = test_config("router_test");
            catalog.create_collection(config).unwrap();

            let coll = catalog.get_collection("router_test").unwrap();
            let mut coll = coll.write().unwrap();

            // Add a router entry
            coll.update_router_entry(RouterEntry {
                segment_id: "test_segment.seg".to_string(),
                level: 0,
                segment_centroid: vec![1.0, 2.0, 3.0],
                num_vectors: 1000,
            }).unwrap();
        }

        // Reopen and verify router was persisted
        {
            let catalog = Catalog::open(dir.path()).unwrap();
            let coll = catalog.get_collection("router_test").unwrap();
            let coll = coll.read().unwrap();

            assert_eq!(coll.router.entries.len(), 1);
            assert_eq!(coll.router.entries[0].segment_id, "test_segment.seg");
            assert_eq!(coll.router.entries[0].num_vectors, 1000);
        }
    }
}
