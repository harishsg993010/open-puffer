//! Application state.

use parking_lot::RwLock;
use puffer_embed::{EmbedConfig, ModelType, PufferEmbedder};
use puffer_fts::config::FtsConfig;
use puffer_fts::index::FtsIndex;
use puffer_query::QueryEngine;
use puffer_storage::Catalog;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

/// Shared application state.
#[derive(Clone)]
pub struct AppState {
    pub catalog: Arc<Catalog>,
    pub engine: Arc<QueryEngine>,
    /// FTS indices per collection.
    fts_indices: Arc<RwLock<HashMap<String, Arc<FtsIndex>>>>,
    /// Embedder instance (lazily initialized).
    embedder: Arc<RwLock<Option<Arc<PufferEmbedder>>>>,
    /// Embedder config.
    embedder_config: Arc<RwLock<Option<EmbedConfig>>>,
    /// Base data directory.
    data_dir: PathBuf,
}

impl AppState {
    pub fn new(catalog: Arc<Catalog>) -> Self {
        let data_dir = catalog.base_path().to_path_buf();
        let engine = Arc::new(QueryEngine::new(catalog.clone()));
        Self {
            catalog,
            engine,
            fts_indices: Arc::new(RwLock::new(HashMap::new())),
            embedder: Arc::new(RwLock::new(None)),
            embedder_config: Arc::new(RwLock::new(None)),
            data_dir,
        }
    }

    /// Get or create FTS index for a collection.
    pub fn get_fts_index(&self, collection_name: &str) -> Option<Arc<FtsIndex>> {
        // Check if already loaded
        {
            let indices = self.fts_indices.read();
            if let Some(index) = indices.get(collection_name) {
                return Some(index.clone());
            }
        }

        // Try to open or create
        let fts_path = self.data_dir.join(collection_name).join("fts");
        let config = FtsConfig::enabled()
            .with_fields(vec!["text".to_string(), "title".to_string()]);

        let index = match FtsIndex::open_or_create(&fts_path, config) {
            Ok(idx) => Arc::new(idx),
            Err(e) => {
                tracing::error!("Failed to open FTS index for {}: {}", collection_name, e);
                return None;
            }
        };

        // Cache it
        {
            let mut indices = self.fts_indices.write();
            indices.insert(collection_name.to_string(), index.clone());
        }

        Some(index)
    }

    /// Enable FTS for a collection with custom config.
    pub fn enable_fts(&self, collection_name: &str, config: FtsConfig) -> Result<(), String> {
        let fts_path = self.data_dir.join(collection_name).join("fts");

        let index = FtsIndex::open_or_create(&fts_path, config)
            .map_err(|e| format!("Failed to create FTS index: {}", e))?;

        let mut indices = self.fts_indices.write();
        indices.insert(collection_name.to_string(), Arc::new(index));

        Ok(())
    }

    /// Disable FTS for a collection.
    pub fn disable_fts(&self, collection_name: &str) {
        let mut indices = self.fts_indices.write();
        indices.remove(collection_name);
    }

    /// Check if FTS is enabled for a collection.
    pub fn has_fts(&self, collection_name: &str) -> bool {
        let indices = self.fts_indices.read();
        indices.contains_key(collection_name)
    }

    /// Configure the embedder with a specific model.
    pub fn configure_embedder(&self, config: EmbedConfig) {
        let mut cfg = self.embedder_config.write();
        *cfg = Some(config);
        // Clear any existing embedder so it gets re-initialized
        let mut emb = self.embedder.write();
        *emb = None;
    }

    /// Get or initialize the embedder.
    pub async fn get_embedder(&self) -> Result<Arc<PufferEmbedder>, String> {
        // Check if already initialized
        {
            let emb = self.embedder.read();
            if let Some(ref embedder) = *emb {
                return Ok(embedder.clone());
            }
        }

        // Get config or use default
        let config = {
            let cfg = self.embedder_config.read();
            cfg.clone().unwrap_or_else(|| EmbedConfig::new(ModelType::Jina))
        };

        // Initialize embedder
        let embedder = PufferEmbedder::new(config)
            .await
            .map_err(|e| format!("Failed to initialize embedder: {}", e))?;

        let embedder = Arc::new(embedder);

        // Cache it
        {
            let mut emb = self.embedder.write();
            *emb = Some(embedder.clone());
        }

        Ok(embedder)
    }

    /// Check if embedder is configured.
    pub fn has_embedder(&self) -> bool {
        let emb = self.embedder.read();
        emb.is_some()
    }

    /// Get current embedder config.
    pub fn get_embedder_config(&self) -> Option<EmbedConfig> {
        let cfg = self.embedder_config.read();
        cfg.clone()
    }
}
