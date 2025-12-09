//! PQ error types.

use thiserror::Error;

/// PQ-related errors.
#[derive(Error, Debug)]
pub enum PqError {
    #[error("Invalid parameters: {0}")]
    InvalidParams(String),

    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("Not enough training samples: need at least {min}, got {got}")]
    InsufficientSamples { min: usize, got: usize },

    #[error("Codebook not trained")]
    CodebookNotTrained,

    #[error("Invalid code index: {0}")]
    InvalidCode(usize),

    #[error("Subvector count {subvectors} does not divide dimension {dim} evenly")]
    SubvectorDimensionMismatch { subvectors: usize, dim: usize },

    #[error("OPQ rotation matrix invalid: expected {expected}x{expected}, got {rows}x{cols}")]
    InvalidRotationMatrix { expected: usize, rows: usize, cols: usize },

    #[error("Training failed to converge after {iterations} iterations")]
    ConvergenceFailure { iterations: usize },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

pub type PqResult<T> = Result<T, PqError>;
