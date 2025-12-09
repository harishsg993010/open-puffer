//! Query execution for Puffer vector database.
//!
//! Handles multi-segment search, staging buffer search, and result merging.
//! Also provides:
//! - Async second-pass refinement for improved recall
//! - Multi-segment prefetching and mmap warming

pub mod compaction;
pub mod engine;
pub mod error;
pub mod refinement;
pub mod prefetch;

pub use compaction::{compact_collection, compact_until_done, needs_compaction, CompactionConfig};
pub use engine::{CompactionResult, QueryEngine, CollectionStats, SegmentStats};
pub use error::{QueryError, QueryResult};
pub use refinement::{
    AsyncRefiner, ExactReranker, RefinedSearchResult, RefinementConfig, StreamingRefiner,
    StreamingUpdate,
};
pub use prefetch::{
    PrefetchConfig, PrefetchManager, PrefetchStats, WarmSegmentCache, MmapAdvisor,
};
