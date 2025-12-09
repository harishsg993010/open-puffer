//! FTS error types.

use thiserror::Error;

/// FTS-related errors.
#[derive(Error, Debug)]
pub enum FtsError {
    #[error("Index not found: {0}")]
    IndexNotFound(String),

    #[error("Index already exists: {0}")]
    IndexAlreadyExists(String),

    #[error("Document not found: {0}")]
    DocumentNotFound(String),

    #[error("Invalid query: {0}")]
    InvalidQuery(String),

    #[error("Schema error: {0}")]
    SchemaError(String),

    #[error("Tantivy error: {0}")]
    Tantivy(#[from] tantivy::TantivyError),

    #[error("Query parser error: {0}")]
    QueryParser(#[from] tantivy::query::QueryParserError),

    #[error("Directory error: {0}")]
    Directory(#[from] tantivy::directory::error::OpenDirectoryError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Index not ready")]
    NotReady,

    #[error("Configuration error: {0}")]
    Config(String),
}

pub type FtsResult<T> = Result<T, FtsError>;
