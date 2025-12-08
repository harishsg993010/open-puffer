//! Query error types.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum QueryError {
    #[error("Storage error: {0}")]
    Storage(#[from] puffer_storage::StorageError),

    #[error("Collection not found: {0}")]
    CollectionNotFound(String),

    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("Invalid query: {0}")]
    InvalidQuery(String),
}

pub type QueryResult<T> = Result<T, QueryError>;
