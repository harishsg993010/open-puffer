//! Error types for the embed module.

use thiserror::Error;

/// Result type for embedding operations.
pub type EmbedResult<T> = Result<T, EmbedError>;

/// Errors that can occur during embedding operations.
#[derive(Debug, Error)]
pub enum EmbedError {
    /// Model not found or failed to load.
    #[error("Model error: {0}")]
    ModelError(String),

    /// Invalid input provided.
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// File not found or cannot be read.
    #[error("File error: {0}")]
    FileError(String),

    /// Embedding generation failed.
    #[error("Embedding failed: {0}")]
    EmbeddingFailed(String),

    /// Configuration error.
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Unsupported model type.
    #[error("Unsupported model type: {0}")]
    UnsupportedModel(String),

    /// IO error.
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}
