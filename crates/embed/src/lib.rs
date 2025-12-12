//! Embedding generation module for Puffer vector database.
//!
//! This module provides text and file embedding capabilities using various models
//! through the `embed_anything` library, plus text chunking via `chunkr`.

pub mod chunker;
pub mod config;
pub mod embedder;
pub mod error;

pub use chunker::{ChunkResult, ChunkingStrategy, PufferChunker};
pub use config::{EmbedConfig, ModelType};
pub use embedder::PufferEmbedder;
pub use error::{EmbedError, EmbedResult};
