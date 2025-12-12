//! Configuration for embedding models.

use serde::{Deserialize, Serialize};

/// Type of embedding model to use.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ModelType {
    /// Jina AI embeddings (multilingual, high quality).
    #[default]
    Jina,
    /// BERT-based embeddings.
    Bert,
    /// Sentence Transformers (all-MiniLM, etc.).
    SentenceTransformer,
    /// CLIP for text-image embeddings.
    Clip,
    /// OpenAI embeddings (requires API key).
    OpenAI,
    /// Cohere embeddings (requires API key).
    Cohere,
}

impl ModelType {
    /// Get the default model ID for this model type.
    pub fn default_model_id(&self) -> &'static str {
        match self {
            ModelType::Jina => "jinaai/jina-embeddings-v2-small-en",
            ModelType::Bert => "sentence-transformers/all-MiniLM-L6-v2",
            ModelType::SentenceTransformer => "sentence-transformers/all-MiniLM-L6-v2",
            ModelType::Clip => "openai/clip-vit-base-patch32",
            ModelType::OpenAI => "text-embedding-3-small",
            ModelType::Cohere => "embed-english-v3.0",
        }
    }

    /// Get the expected embedding dimension for default models.
    pub fn default_dimension(&self) -> usize {
        match self {
            ModelType::Jina => 512,
            ModelType::Bert => 384,
            ModelType::SentenceTransformer => 384,
            ModelType::Clip => 512,
            ModelType::OpenAI => 1536,
            ModelType::Cohere => 1024,
        }
    }

    /// Check if this model requires an API key.
    pub fn requires_api_key(&self) -> bool {
        matches!(self, ModelType::OpenAI | ModelType::Cohere)
    }

    /// Get the embed_anything architecture name.
    pub fn architecture(&self) -> &'static str {
        match self {
            ModelType::Jina => "jina",
            ModelType::Bert => "bert",
            ModelType::SentenceTransformer => "bert",
            ModelType::Clip => "clip",
            ModelType::OpenAI => "openai",
            ModelType::Cohere => "cohere",
        }
    }
}

/// Configuration for the embedding service.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedConfig {
    /// Type of model to use.
    pub model_type: ModelType,

    /// Specific model ID (overrides default).
    pub model_id: Option<String>,

    /// API key for cloud models.
    pub api_key: Option<String>,

    /// Maximum sequence length for text.
    pub max_seq_length: usize,

    /// Batch size for embedding multiple texts.
    pub batch_size: usize,

    /// Whether to normalize embeddings.
    pub normalize: bool,

    /// Chunk size for long documents.
    pub chunk_size: usize,

    /// Chunk overlap for long documents.
    pub chunk_overlap: usize,
}

impl Default for EmbedConfig {
    fn default() -> Self {
        Self {
            model_type: ModelType::default(),
            model_id: None,
            api_key: None,
            max_seq_length: 512,
            batch_size: 32,
            normalize: true,
            chunk_size: 256,
            chunk_overlap: 32,
        }
    }
}

impl EmbedConfig {
    /// Create a new config with the specified model type.
    pub fn new(model_type: ModelType) -> Self {
        Self {
            model_type,
            ..Default::default()
        }
    }

    /// Set a custom model ID.
    pub fn with_model_id(mut self, model_id: impl Into<String>) -> Self {
        self.model_id = Some(model_id.into());
        self
    }

    /// Set the API key for cloud models.
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Set the batch size.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set chunking parameters.
    pub fn with_chunking(mut self, chunk_size: usize, overlap: usize) -> Self {
        self.chunk_size = chunk_size;
        self.chunk_overlap = overlap;
        self
    }

    /// Get the model ID to use.
    pub fn get_model_id(&self) -> &str {
        self.model_id
            .as_deref()
            .unwrap_or_else(|| self.model_type.default_model_id())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = EmbedConfig::default();
        assert!(matches!(config.model_type, ModelType::Jina));
        assert_eq!(config.batch_size, 32);
    }

    #[test]
    fn test_config_builder() {
        let config = EmbedConfig::new(ModelType::OpenAI)
            .with_api_key("test-key")
            .with_batch_size(64);

        assert!(matches!(config.model_type, ModelType::OpenAI));
        assert_eq!(config.api_key, Some("test-key".to_string()));
        assert_eq!(config.batch_size, 64);
    }
}
