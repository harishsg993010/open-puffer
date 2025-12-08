//! Indexing algorithms for Puffer vector database.
//!
//! Implements IVF-Flat (Inverted File with Flat quantization) indexing.

pub mod kmeans;
pub mod search;

pub use kmeans::{kmeans, KMeansConfig};
pub use search::{search_segment, SearchResult};
