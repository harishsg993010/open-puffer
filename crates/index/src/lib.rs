//! Indexing algorithms for Puffer vector database.
//!
//! Implements:
//! - IVF-Flat (Inverted File with Flat quantization) indexing
//! - IVF-HNSW (IVF with HNSW intra-cluster graphs)
//! - IVF-PQ (IVF with Product Quantization compression)

pub mod kmeans;
pub mod search;
pub mod ivf_hnsw;
pub mod ivf_pq;

pub use kmeans::{kmeans, KMeansConfig, KMeansResult, build_cluster_data};
pub use search::{search_segment, SearchResult, merge_results, brute_force_search};
pub use ivf_hnsw::{IvfHnswIndex, IvfHnswConfig, IvfHnswSearchResult, IvfHnswStats};
pub use ivf_pq::{IvfPqIndex, IvfPqConfig, IvfPqSearchResult, IvfPqStats};
