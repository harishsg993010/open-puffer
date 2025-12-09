//! HNSW (Hierarchical Navigable Small World) graph index for Puffer.
//!
//! This module implements HNSW for approximate nearest neighbor search.
//! It can be used standalone or as part of IVF+HNSW hybrid indexing.
//!
//! Reference: "Efficient and robust approximate nearest neighbor search using
//! Hierarchical Navigable Small World graphs" by Malkov & Yashunin, 2016
//!
//! # Features
//! - Multi-layer graph with exponentially decaying node distribution
//! - Efficient greedy search with backtracking
//! - Configurable construction parameters (M, ef_construction)
//! - Serialization for persistent storage
//!
//! # Example
//! ```ignore
//! use puffer_hnsw::{HnswIndex, HnswConfig};
//!
//! let config = HnswConfig::default();
//! let mut index = HnswIndex::new(128, config);
//!
//! // Add vectors
//! for (id, vector) in vectors.iter().enumerate() {
//!     index.add(id, vector);
//! }
//!
//! // Search
//! let results = index.search(&query, 10);
//! ```

pub mod config;
pub mod error;
pub mod graph;
pub mod search;
pub mod builder;

pub use config::HnswConfig;
pub use error::{HnswError, HnswResult};
pub use graph::{HnswGraph, HnswNode};
pub use search::{search_hnsw, HnswSearchResult};
pub use builder::HnswBuilder;

use puffer_core::Metric;
use serde::{Deserialize, Serialize};

/// A complete HNSW index with vectors and graph structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswIndex {
    /// Configuration parameters.
    pub config: HnswConfig,

    /// The graph structure.
    pub graph: HnswGraph,

    /// Vector dimension.
    pub dim: usize,

    /// Distance metric.
    pub metric: Metric,

    /// Stored vectors (optional, can be external).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vectors: Option<Vec<Vec<f32>>>,

    /// Entry point (node with highest layer).
    pub entry_point: Option<usize>,

    /// Maximum layer currently in use.
    pub max_layer: usize,

    /// Number of vectors in the index.
    pub num_vectors: usize,
}

impl HnswIndex {
    /// Create a new empty HNSW index.
    pub fn new(dim: usize, config: HnswConfig) -> Self {
        Self {
            config,
            graph: HnswGraph::new(),
            dim,
            metric: Metric::L2,
            vectors: Some(Vec::new()),
            entry_point: None,
            max_layer: 0,
            num_vectors: 0,
        }
    }

    /// Create a new HNSW index with a specific metric.
    pub fn with_metric(dim: usize, metric: Metric, config: HnswConfig) -> Self {
        Self {
            config,
            graph: HnswGraph::new(),
            dim,
            metric,
            vectors: Some(Vec::new()),
            entry_point: None,
            max_layer: 0,
            num_vectors: 0,
        }
    }

    /// Create an HNSW index without storing vectors (for external storage).
    pub fn without_vectors(dim: usize, metric: Metric, config: HnswConfig) -> Self {
        Self {
            config,
            graph: HnswGraph::new(),
            dim,
            metric,
            vectors: None,
            entry_point: None,
            max_layer: 0,
            num_vectors: 0,
        }
    }

    /// Check if the index stores vectors internally.
    pub fn has_vectors(&self) -> bool {
        self.vectors.is_some()
    }

    /// Get a vector by index.
    pub fn get_vector(&self, id: usize) -> Option<&[f32]> {
        self.vectors.as_ref()?.get(id).map(|v| v.as_slice())
    }

    /// Get the number of vectors in the index.
    pub fn len(&self) -> usize {
        self.num_vectors
    }

    /// Check if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.num_vectors == 0
    }

    /// Serialize the index to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        // Format:
        // [dim: u32][metric: u8][num_vectors: u32][max_layer: u32][entry_point: i64]
        // [config bytes][graph bytes][optional vectors]
        let mut bytes = Vec::new();

        // Header
        bytes.extend_from_slice(&(self.dim as u32).to_le_bytes());
        bytes.push(self.metric.to_byte());
        bytes.extend_from_slice(&(self.num_vectors as u32).to_le_bytes());
        bytes.extend_from_slice(&(self.max_layer as u32).to_le_bytes());

        let entry_point = self.entry_point.map(|e| e as i64).unwrap_or(-1);
        bytes.extend_from_slice(&entry_point.to_le_bytes());

        // Config
        let config_bytes = self.config.to_bytes();
        bytes.extend_from_slice(&(config_bytes.len() as u32).to_le_bytes());
        bytes.extend(config_bytes);

        // Graph
        let graph_bytes = self.graph.to_bytes();
        bytes.extend_from_slice(&(graph_bytes.len() as u32).to_le_bytes());
        bytes.extend(graph_bytes);

        // Vectors (if stored)
        if let Some(ref vectors) = self.vectors {
            bytes.push(1u8);
            bytes.extend_from_slice(&(vectors.len() as u32).to_le_bytes());
            for vec in vectors {
                for &v in vec {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
            }
        } else {
            bytes.push(0u8);
        }

        bytes
    }

    /// Deserialize from bytes.
    pub fn from_bytes(data: &[u8]) -> HnswResult<Self> {
        if data.len() < 22 {
            return Err(HnswError::InvalidData("Data too short".into()));
        }

        let dim = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
        let metric = Metric::from_byte(data[4])
            .ok_or_else(|| HnswError::InvalidData("Invalid metric".into()))?;
        let num_vectors = u32::from_le_bytes(data[5..9].try_into().unwrap()) as usize;
        let max_layer = u32::from_le_bytes(data[9..13].try_into().unwrap()) as usize;
        let entry_point_raw = i64::from_le_bytes(data[13..21].try_into().unwrap());
        let entry_point = if entry_point_raw >= 0 {
            Some(entry_point_raw as usize)
        } else {
            None
        };

        let mut offset = 21;

        // Config
        let config_len = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;
        let config = HnswConfig::from_bytes(&data[offset..offset + config_len])?;
        offset += config_len;

        // Graph
        let graph_len = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;
        let graph = HnswGraph::from_bytes(&data[offset..offset + graph_len])?;
        offset += graph_len;

        // Vectors
        let has_vectors = data[offset] != 0;
        offset += 1;

        let vectors = if has_vectors {
            let vec_count = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
            offset += 4;

            let mut vectors = Vec::with_capacity(vec_count);
            for _ in 0..vec_count {
                let vec: Vec<f32> = (0..dim)
                    .map(|i| {
                        let start = offset + i * 4;
                        f32::from_le_bytes(data[start..start + 4].try_into().unwrap())
                    })
                    .collect();
                offset += dim * 4;
                vectors.push(vec);
            }
            Some(vectors)
        } else {
            None
        };

        Ok(Self {
            config,
            graph,
            dim,
            metric,
            vectors,
            entry_point,
            max_layer,
            num_vectors,
        })
    }
}
