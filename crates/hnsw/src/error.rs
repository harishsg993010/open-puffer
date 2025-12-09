//! HNSW error types.

use thiserror::Error;

/// HNSW-related errors.
#[derive(Error, Debug)]
pub enum HnswError {
    #[error("Invalid parameters: {0}")]
    InvalidParams(String),

    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("Index is empty")]
    EmptyIndex,

    #[error("Node not found: {0}")]
    NodeNotFound(usize),

    #[error("Invalid layer: {layer} > max {max}")]
    InvalidLayer { layer: usize, max: usize },

    #[error("Invalid data: {0}")]
    InvalidData(String),

    #[error("Index not built")]
    NotBuilt,

    #[error("Vector storage not available")]
    VectorsNotStored,

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub type HnswResult<T> = Result<T, HnswError>;
