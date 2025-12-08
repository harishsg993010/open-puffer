//! Core vector operations and distance metrics for Puffer vector database.

pub mod distance;
pub mod metric;
pub mod types;

pub use distance::{cosine_distance, cosine_similarity, l2_distance, l2_distance_squared};
pub use metric::Metric;
pub use types::{VectorId, VectorRecord};
