//! Product Quantization (PQ) and Optimized Product Quantization (OPQ) for Puffer.
//!
//! This module implements:
//! - PQ: Splits vectors into M subvectors, quantizes each to K centroids
//! - OPQ: Applies a learned rotation matrix before PQ for better compression
//! - ADC (Asymmetric Distance Computation) for fast approximate search
//!
//! # Example
//! ```ignore
//! use puffer_pq::{PqCodebook, PqParams, train_pq, encode_vectors, search_pq};
//!
//! let params = PqParams::new(8, 256); // 8 subvectors, 256 centroids each
//! let codebook = train_pq(&training_vectors, &params)?;
//! let codes = encode_vectors(&codebook, &vectors);
//! let results = search_pq(&codebook, &codes, &query, 10);
//! ```

pub mod codebook;
pub mod encoding;
pub mod search;
pub mod opq;
pub mod error;
pub mod config;

pub use codebook::{PqCodebook, train_pq, train_pq_parallel};
pub use encoding::{encode_vector, encode_vectors, decode_vector};
pub use search::{compute_distance_table, search_pq, search_pq_with_refinement, PqSearchResult};
pub use opq::{OpqTransform, train_opq, apply_opq_rotation};
pub use error::{PqError, PqResult};
pub use config::{PqParams, OpqParams, QuantizationMode, QuantizationConfig};
