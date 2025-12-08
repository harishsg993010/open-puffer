//! Query execution for Puffer vector database.
//!
//! Handles multi-segment search, staging buffer search, and result merging.

pub mod engine;
pub mod error;

pub use engine::QueryEngine;
pub use error::{QueryError, QueryResult};
