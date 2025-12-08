//! Storage error types.

use thiserror::Error;

/// Storage-related errors.
#[derive(Error, Debug)]
pub enum StorageError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Invalid segment magic bytes")]
    InvalidMagic,

    #[error("Unsupported segment version: {0}")]
    UnsupportedVersion(u32),

    #[error("Invalid metric byte: {0}")]
    InvalidMetric(u8),

    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("Collection not found: {0}")]
    CollectionNotFound(String),

    #[error("Collection already exists: {0}")]
    CollectionAlreadyExists(String),

    #[error("Segment not found: {0}")]
    SegmentNotFound(String),

    #[error("Invalid segment file: {0}")]
    InvalidSegment(String),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Catalog is locked")]
    CatalogLocked,

    #[error("Invalid ID: {0}")]
    InvalidId(String),
}

pub type StorageResult<T> = Result<T, StorageError>;
