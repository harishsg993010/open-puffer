//! Full-text search (BM25) for Puffer using Tantivy.
//!
//! This module provides:
//! - Per-collection Tantivy indexes for BM25 text search
//! - Document indexing with vector ID mapping
//! - BM25 search with configurable fields
//! - Hybrid search fusion (BM25 + vector similarity)
//!
//! # Example
//! ```ignore
//! use puffer_fts::{FtsIndex, FtsConfig, TextDocument};
//!
//! let config = FtsConfig::default();
//! let mut index = FtsIndex::create("./data/fts", config)?;
//!
//! // Index documents
//! index.add_document(TextDocument {
//!     vector_id: "vec_0".to_string(),
//!     text: "Machine learning is a subset of AI".to_string(),
//!     title: Some("ML Intro".to_string()),
//!     ..Default::default()
//! })?;
//!
//! // Search
//! let results = index.search("machine learning", 10)?;
//! ```

pub mod config;
pub mod error;
pub mod index;
pub mod schema;
pub mod search;
pub mod hybrid;

pub use config::FtsConfig;
pub use error::{FtsError, FtsResult};
pub use index::FtsIndex;
pub use schema::{FtsSchema, TextDocument};
pub use search::{search_bm25, FtsSearchResult};
pub use hybrid::{hybrid_search, HybridConfig, HybridResult, FusionMethod};
