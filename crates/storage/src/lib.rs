//! Storage layer for Puffer vector database.
//!
//! Provides segment file format, catalog management, and disk I/O.

pub mod catalog;
pub mod error;
pub mod segment;

pub use catalog::{Catalog, Collection, CollectionConfig};
pub use error::{StorageError, StorageResult};
pub use segment::{ClusterMeta, LoadedSegment, SegmentBuilder, SegmentHeader};
