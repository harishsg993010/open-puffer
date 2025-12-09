//! FTS configuration.

use serde::{Deserialize, Serialize};

/// Configuration for full-text search index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FtsConfig {
    /// Whether FTS is enabled for this collection.
    pub enabled: bool,

    /// Fields to index for text search.
    pub indexed_fields: Vec<String>,

    /// Default field for queries without field prefix.
    pub default_field: String,

    /// Heap size for Tantivy writer (in bytes).
    pub writer_heap_size: usize,

    /// Number of threads for indexing.
    pub num_indexing_threads: usize,

    /// Auto-commit after this many documents.
    pub auto_commit_threshold: usize,

    /// Enable stemming for text analysis.
    pub enable_stemming: bool,

    /// Language for stemming (e.g., "english", "german").
    pub stemmer_language: String,

    /// Store original text for retrieval.
    pub store_text: bool,

    /// Enable position indexing for phrase queries.
    pub enable_positions: bool,

    /// BM25 k1 parameter (term frequency saturation).
    pub bm25_k1: f32,

    /// BM25 b parameter (document length normalization).
    pub bm25_b: f32,
}

impl Default for FtsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            indexed_fields: vec!["text".to_string()],
            default_field: "text".to_string(),
            writer_heap_size: 50_000_000, // 50MB
            num_indexing_threads: 1,
            auto_commit_threshold: 1000,
            enable_stemming: true,
            stemmer_language: "english".to_string(),
            store_text: true,
            enable_positions: true,
            bm25_k1: 1.2,
            bm25_b: 0.75,
        }
    }
}

impl FtsConfig {
    /// Create a new config with FTS enabled.
    pub fn enabled() -> Self {
        Self {
            enabled: true,
            ..Default::default()
        }
    }

    /// Set indexed fields.
    pub fn with_fields(mut self, fields: Vec<String>) -> Self {
        self.indexed_fields = fields;
        self
    }

    /// Set default search field.
    pub fn with_default_field(mut self, field: String) -> Self {
        self.default_field = field;
        self
    }

    /// Set writer heap size.
    pub fn with_heap_size(mut self, size: usize) -> Self {
        self.writer_heap_size = size;
        self
    }

    /// Set auto-commit threshold.
    pub fn with_auto_commit(mut self, threshold: usize) -> Self {
        self.auto_commit_threshold = threshold;
        self
    }

    /// Disable stemming.
    pub fn without_stemming(mut self) -> Self {
        self.enable_stemming = false;
        self
    }

    /// Set BM25 parameters.
    pub fn with_bm25_params(mut self, k1: f32, b: f32) -> Self {
        self.bm25_k1 = k1;
        self.bm25_b = b;
        self
    }

    /// Validate configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.indexed_fields.is_empty() {
            return Err("At least one indexed field required".to_string());
        }
        if !self.indexed_fields.contains(&self.default_field) {
            return Err(format!(
                "Default field '{}' must be in indexed_fields",
                self.default_field
            ));
        }
        if self.bm25_k1 < 0.0 {
            return Err("BM25 k1 must be >= 0".to_string());
        }
        if self.bm25_b < 0.0 || self.bm25_b > 1.0 {
            return Err("BM25 b must be in [0, 1]".to_string());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_validation() {
        let config = FtsConfig::enabled();
        assert!(config.validate().is_ok());

        let bad_config = FtsConfig {
            indexed_fields: vec![],
            ..Default::default()
        };
        assert!(bad_config.validate().is_err());
    }

    #[test]
    fn test_config_builder() {
        let config = FtsConfig::enabled()
            .with_fields(vec!["title".to_string(), "body".to_string()])
            .with_default_field("body".to_string())
            .with_bm25_params(1.5, 0.8);

        assert!(config.enabled);
        assert_eq!(config.indexed_fields.len(), 2);
        assert_eq!(config.bm25_k1, 1.5);
    }
}
