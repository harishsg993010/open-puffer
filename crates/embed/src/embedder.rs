//! Embedder implementation using embed_anything.

use crate::config::{EmbedConfig, ModelType};
use crate::error::{EmbedError, EmbedResult};
use embed_anything::config::TextEmbedConfig;
use embed_anything::embeddings::embed::Embedder;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info};

/// Embedding result containing the vector and metadata.
#[derive(Debug, Clone)]
pub struct EmbeddingResult {
    /// The embedding vector.
    pub embedding: Vec<f32>,
    /// Original text (if available).
    pub text: Option<String>,
    /// Chunk index for chunked documents.
    pub chunk_index: Option<usize>,
}

/// Puffer embedder wrapping embed_anything.
pub struct PufferEmbedder {
    /// The underlying embedder.
    embedder: Arc<RwLock<Embedder>>,
    /// Configuration.
    config: EmbedConfig,
    /// Embedding dimension.
    dimension: usize,
}

impl PufferEmbedder {
    /// Create a new embedder with the given configuration.
    pub async fn new(config: EmbedConfig) -> EmbedResult<Self> {
        info!(
            "Initializing embedder with model: {} ({})",
            config.get_model_id(),
            config.model_type.architecture()
        );

        let embedder = if config.model_type.requires_api_key() {
            // Cloud-based embedder
            let api_key = config.api_key.clone().ok_or_else(|| {
                EmbedError::ConfigError(format!(
                    "{:?} requires an API key",
                    config.model_type
                ))
            })?;

            Embedder::from_pretrained_cloud(
                config.model_type.architecture(),
                config.get_model_id(),
                Some(api_key),
            )
            .map_err(|e| EmbedError::ModelError(e.to_string()))?
        } else {
            // Local embedder from HuggingFace
            // from_pretrained_hf(model, model_id, revision, token, dtype)
            Embedder::from_pretrained_hf(
                config.model_type.architecture(),
                config.get_model_id(),
                None,  // revision
                None,  // token
                None,  // dtype
            )
            .map_err(|e| EmbedError::ModelError(e.to_string()))?
        };

        let dimension = config.model_type.default_dimension();

        Ok(Self {
            embedder: Arc::new(RwLock::new(embedder)),
            config,
            dimension,
        })
    }

    /// Create an embedder with default Jina model.
    pub async fn default_jina() -> EmbedResult<Self> {
        Self::new(EmbedConfig::new(ModelType::Jina)).await
    }

    /// Create an embedder with sentence transformers.
    pub async fn default_sentence_transformer() -> EmbedResult<Self> {
        Self::new(EmbedConfig::new(ModelType::SentenceTransformer)).await
    }

    /// Get the embedding dimension.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get the configuration.
    pub fn config(&self) -> &EmbedConfig {
        &self.config
    }

    /// Embed a single text string.
    pub async fn embed_text(&self, text: &str) -> EmbedResult<Vec<f32>> {
        let results = self.embed_texts(&[text.to_string()]).await?;
        results
            .into_iter()
            .next()
            .map(|r| r.embedding)
            .ok_or_else(|| EmbedError::EmbeddingFailed("No embedding generated".into()))
    }

    /// Embed multiple texts.
    pub async fn embed_texts(&self, texts: &[String]) -> EmbedResult<Vec<EmbeddingResult>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        debug!("Embedding {} texts", texts.len());

        let embedder = self.embedder.read().await;

        let mut results = Vec::with_capacity(texts.len());

        // Process in batches
        for (batch_idx, batch) in texts.chunks(self.config.batch_size).enumerate() {
            debug!("Processing batch {} ({} texts)", batch_idx, batch.len());

            // Convert to &str slice
            let text_refs: Vec<&str> = batch.iter().map(|s| s.as_str()).collect();

            // Use embed_anything's text embedding - embed is async and takes &[&str]
            // embed(text_batch, batch_size, late_chunking)
            match embedder.embed(&text_refs, Some(self.config.batch_size), None).await {
                Ok(embed_results) => {
                    for (idx, embed_result) in embed_results.into_iter().enumerate() {
                        // EmbeddingResult is an enum, extract dense vector
                        let embedding = embed_result.to_dense()
                            .map_err(|e| EmbedError::EmbeddingFailed(e.to_string()))?;

                        results.push(EmbeddingResult {
                            embedding,
                            text: Some(batch[idx].clone()),
                            chunk_index: Some(batch_idx * self.config.batch_size + idx),
                        });
                    }
                }
                Err(e) => {
                    return Err(EmbedError::EmbeddingFailed(format!(
                        "Failed to embed text: {}",
                        e
                    )));
                }
            }
        }

        Ok(results)
    }

    /// Embed a file (PDF, text, etc.).
    pub async fn embed_file(&self, path: impl AsRef<Path>) -> EmbedResult<Vec<EmbeddingResult>> {
        let path = path.as_ref();

        if !path.exists() {
            return Err(EmbedError::FileError(format!(
                "File not found: {}",
                path.display()
            )));
        }

        info!("Embedding file: {}", path.display());

        let embedder = self.embedder.read().await;
        let embed_config = TextEmbedConfig::default();

        let path_str = path.to_string_lossy().to_string();

        // Use embed_anything's file embedding
        // Returns Result<Option<Vec<EmbedData>>>
        let embed_data_opt = embed_anything::embed_file(&path_str, &embedder, Some(&embed_config), None)
            .await
            .map_err(|e| EmbedError::EmbeddingFailed(format!("Failed to embed file: {}", e)))?;

        let embed_data = embed_data_opt.ok_or_else(|| {
            EmbedError::EmbeddingFailed("No embeddings generated for file".into())
        })?;

        let mut results = Vec::with_capacity(embed_data.len());
        for (idx, data) in embed_data.into_iter().enumerate() {
            // EmbedData has embedding: EmbeddingResult which is an enum
            let embedding = data.embedding.to_dense()
                .map_err(|e| EmbedError::EmbeddingFailed(e.to_string()))?;

            results.push(EmbeddingResult {
                embedding,
                text: data.text,
                chunk_index: Some(idx),
            });
        }

        Ok(results)
    }

    /// Embed a directory of files.
    pub async fn embed_directory(
        &self,
        path: impl AsRef<Path>,
        extensions: Option<&[&str]>,
    ) -> EmbedResult<Vec<(String, Vec<EmbeddingResult>)>> {
        let path = path.as_ref();

        if !path.is_dir() {
            return Err(EmbedError::FileError(format!(
                "Not a directory: {}",
                path.display()
            )));
        }

        let mut results = Vec::new();

        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            let file_path = entry.path();

            if file_path.is_file() {
                // Check extension filter
                if let Some(exts) = extensions {
                    let ext = file_path
                        .extension()
                        .and_then(|e| e.to_str())
                        .unwrap_or("");
                    if !exts.contains(&ext) {
                        continue;
                    }
                }

                match self.embed_file(&file_path).await {
                    Ok(embeddings) => {
                        results.push((file_path.to_string_lossy().to_string(), embeddings));
                    }
                    Err(e) => {
                        tracing::warn!("Failed to embed {}: {}", file_path.display(), e);
                    }
                }
            }
        }

        Ok(results)
    }

    /// Chunk text and embed each chunk.
    pub async fn embed_chunked(&self, text: &str) -> EmbedResult<Vec<EmbeddingResult>> {
        let chunks = self.chunk_text(text);
        let chunk_strings: Vec<String> = chunks.into_iter().map(|s| s.to_string()).collect();
        self.embed_texts(&chunk_strings).await
    }

    /// Simple text chunking by words.
    fn chunk_text<'a>(&self, text: &'a str) -> Vec<&'a str> {
        let words: Vec<&str> = text.split_whitespace().collect();
        let chunk_size = self.config.chunk_size;
        let overlap = self.config.chunk_overlap;

        if words.len() <= chunk_size {
            return vec![text];
        }

        let mut chunks = Vec::new();
        let mut start = 0;

        while start < words.len() {
            let end = (start + chunk_size).min(words.len());

            // Find byte positions for the chunk
            let chunk_words = &words[start..end];
            if !chunk_words.is_empty() {
                // Reconstruct chunk from words (simplified)
                let first_word = chunk_words[0];
                let last_word = chunk_words[chunk_words.len() - 1];

                let start_pos = text.find(first_word).unwrap_or(0);
                let end_pos = text
                    .rfind(last_word)
                    .map(|p| p + last_word.len())
                    .unwrap_or(text.len());

                if start_pos < end_pos && end_pos <= text.len() {
                    chunks.push(&text[start_pos..end_pos]);
                }
            }

            if end >= words.len() {
                break;
            }

            start = end - overlap;
        }

        chunks
    }
}

/// Utility to compute embedding similarity (cosine).
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);

        let c = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &c).abs() < 1e-6);
    }

    #[test]
    fn test_chunk_text() {
        let config = EmbedConfig::default().with_chunking(10, 2);
        // We can't easily test chunking without creating an embedder
        // Just verify the config
        assert_eq!(config.chunk_size, 10);
        assert_eq!(config.chunk_overlap, 2);
    }
}
