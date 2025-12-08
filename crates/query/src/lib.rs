//! Query execution for Puffer vector database.
//!
//! Handles multi-segment search, staging buffer search, and result merging.

pub mod compaction;
pub mod engine;
pub mod error;

pub use compaction::{compact_collection, compact_until_done, needs_compaction, CompactionConfig};
pub use engine::{CompactionResult, QueryEngine};
pub use error::{QueryError, QueryResult};
