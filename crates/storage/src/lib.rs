//! Storage layer for Puffer vector database.
//!
//! Provides segment file format, catalog management, and disk I/O.

pub mod catalog;
pub mod error;
pub mod router;
pub mod segment;

pub use catalog::{Catalog, Collection, CollectionConfig, CollectionMeta, SegmentMeta};
pub use error::{StorageError, StorageResult};
pub use router::{compute_centroid, compute_centroid_from_slice, RouterEntry, RouterIndex};
pub use segment::{ClusterMeta, LoadedSegment, SegmentBuilder, SegmentHeader};
