//! Segment file format and operations.
//!
//! Segment file layout:
//! ```text
//! HEADER (fixed size):
//!   magic: "PRSEG1\0" (7 bytes)
//!   version: u32
//!   dimension: u32
//!   metric: u8
//!   num_vectors: u32
//!   num_clusters: u32
//!   cluster_meta_offset: u64
//!   vector_data_offset: u64
//!   id_table_offset: u64
//!   payload_offset_table_offset: u64
//!   payload_blob_offset: u64
//!
//! CLUSTER METADATA:
//!   For each cluster:
//!     centroid[dim]: f32 array
//!     start_index: u32
//!     length: u32
//!
//! VECTOR DATA:
//!   [num_vectors * dim] f32 values, in cluster order
//!
//! ID TABLE:
//!   For each vector: length-prefixed string (u8 len + bytes)
//!
//! PAYLOAD OFFSET TABLE:
//!   For each vector: (offset: u64, length: u32)
//!
//! PAYLOAD BLOB:
//!   Concatenated JSON payloads
//! ```

use crate::error::{StorageError, StorageResult};
use memmap2::Mmap;
use puffer_core::{Metric, VectorId};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

/// Magic bytes for segment files.
pub const SEGMENT_MAGIC: &[u8; 7] = b"PRSEG1\0";

/// Current segment format version.
pub const SEGMENT_VERSION: u32 = 1;

/// Size of the fixed header in bytes.
pub const HEADER_SIZE: usize = 7 + 4 + 4 + 1 + 4 + 4 + 8 * 5; // 64 bytes

/// Segment file header.
#[derive(Debug, Clone)]
pub struct SegmentHeader {
    pub version: u32,
    pub dimension: u32,
    pub metric: Metric,
    pub num_vectors: u32,
    pub num_clusters: u32,
    pub cluster_meta_offset: u64,
    pub vector_data_offset: u64,
    pub id_table_offset: u64,
    pub payload_offset_table_offset: u64,
    pub payload_blob_offset: u64,
}

impl SegmentHeader {
    /// Parse header from bytes.
    pub fn from_bytes(data: &[u8]) -> StorageResult<Self> {
        if data.len() < HEADER_SIZE {
            return Err(StorageError::InvalidSegment("Header too small".into()));
        }

        // Check magic
        if &data[0..7] != SEGMENT_MAGIC {
            return Err(StorageError::InvalidMagic);
        }

        let version = u32::from_le_bytes(data[7..11].try_into().unwrap());
        if version != SEGMENT_VERSION {
            return Err(StorageError::UnsupportedVersion(version));
        }

        let dimension = u32::from_le_bytes(data[11..15].try_into().unwrap());
        let metric_byte = data[15];
        let metric = Metric::from_byte(metric_byte)
            .ok_or(StorageError::InvalidMetric(metric_byte))?;
        let num_vectors = u32::from_le_bytes(data[16..20].try_into().unwrap());
        let num_clusters = u32::from_le_bytes(data[20..24].try_into().unwrap());
        let cluster_meta_offset = u64::from_le_bytes(data[24..32].try_into().unwrap());
        let vector_data_offset = u64::from_le_bytes(data[32..40].try_into().unwrap());
        let id_table_offset = u64::from_le_bytes(data[40..48].try_into().unwrap());
        let payload_offset_table_offset = u64::from_le_bytes(data[48..56].try_into().unwrap());
        let payload_blob_offset = u64::from_le_bytes(data[56..64].try_into().unwrap());

        Ok(Self {
            version,
            dimension,
            metric,
            num_vectors,
            num_clusters,
            cluster_meta_offset,
            vector_data_offset,
            id_table_offset,
            payload_offset_table_offset,
            payload_blob_offset,
        })
    }

    /// Serialize header to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(HEADER_SIZE);
        buf.extend_from_slice(SEGMENT_MAGIC);
        buf.extend_from_slice(&self.version.to_le_bytes());
        buf.extend_from_slice(&self.dimension.to_le_bytes());
        buf.push(self.metric.to_byte());
        buf.extend_from_slice(&self.num_vectors.to_le_bytes());
        buf.extend_from_slice(&self.num_clusters.to_le_bytes());
        buf.extend_from_slice(&self.cluster_meta_offset.to_le_bytes());
        buf.extend_from_slice(&self.vector_data_offset.to_le_bytes());
        buf.extend_from_slice(&self.id_table_offset.to_le_bytes());
        buf.extend_from_slice(&self.payload_offset_table_offset.to_le_bytes());
        buf.extend_from_slice(&self.payload_blob_offset.to_le_bytes());
        buf
    }
}

/// Metadata for a single cluster.
#[derive(Debug, Clone)]
pub struct ClusterMeta {
    /// Centroid vector for this cluster.
    pub centroid: Vec<f32>,
    /// Start index in the vector data array.
    pub start_index: u32,
    /// Number of vectors in this cluster.
    pub length: u32,
}

impl ClusterMeta {
    /// Size in bytes for a cluster with given dimension.
    pub fn byte_size(dim: usize) -> usize {
        dim * 4 + 4 + 4 // centroid + start_index + length
    }

    /// Parse cluster metadata from bytes.
    pub fn from_bytes(data: &[u8], dim: usize) -> StorageResult<Self> {
        let expected_size = Self::byte_size(dim);
        if data.len() < expected_size {
            return Err(StorageError::InvalidSegment("Cluster meta too small".into()));
        }

        let centroid: Vec<f32> = (0..dim)
            .map(|i| {
                let start = i * 4;
                f32::from_le_bytes(data[start..start + 4].try_into().unwrap())
            })
            .collect();

        let offset = dim * 4;
        let start_index = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
        let length = u32::from_le_bytes(data[offset + 4..offset + 8].try_into().unwrap());

        Ok(Self {
            centroid,
            start_index,
            length,
        })
    }

    /// Serialize cluster metadata to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(Self::byte_size(self.centroid.len()));
        for &v in &self.centroid {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        buf.extend_from_slice(&self.start_index.to_le_bytes());
        buf.extend_from_slice(&self.length.to_le_bytes());
        buf
    }
}

/// Payload offset entry.
#[derive(Debug, Clone, Copy)]
pub struct PayloadOffset {
    pub offset: u64,
    pub length: u32,
}

impl PayloadOffset {
    pub const SIZE: usize = 12; // u64 + u32

    pub fn from_bytes(data: &[u8]) -> Self {
        Self {
            offset: u64::from_le_bytes(data[0..8].try_into().unwrap()),
            length: u32::from_le_bytes(data[8..12].try_into().unwrap()),
        }
    }

    pub fn to_bytes(&self) -> [u8; 12] {
        let mut buf = [0u8; 12];
        buf[0..8].copy_from_slice(&self.offset.to_le_bytes());
        buf[8..12].copy_from_slice(&self.length.to_le_bytes());
        buf
    }
}

/// A loaded segment with memory-mapped file.
pub struct LoadedSegment {
    pub header: SegmentHeader,
    pub clusters: Vec<ClusterMeta>,
    mmap: Mmap,
    /// Cached ID offsets for faster lookup
    id_offsets: Vec<usize>,
}

impl LoadedSegment {
    /// Open a segment file.
    pub fn open(path: &Path) -> StorageResult<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        let header = SegmentHeader::from_bytes(&mmap)?;

        // Parse cluster metadata
        let dim = header.dimension as usize;
        let cluster_size = ClusterMeta::byte_size(dim);
        let mut clusters = Vec::with_capacity(header.num_clusters as usize);

        for i in 0..header.num_clusters as usize {
            let offset = header.cluster_meta_offset as usize + i * cluster_size;
            let cluster = ClusterMeta::from_bytes(&mmap[offset..], dim)?;
            clusters.push(cluster);
        }

        // Build ID offset table for fast lookups
        let id_offsets = Self::build_id_offsets(&mmap, &header)?;

        Ok(Self {
            header,
            clusters,
            mmap,
            id_offsets,
        })
    }

    fn build_id_offsets(mmap: &[u8], header: &SegmentHeader) -> StorageResult<Vec<usize>> {
        let mut offsets = Vec::with_capacity(header.num_vectors as usize);
        let mut pos = header.id_table_offset as usize;

        for _ in 0..header.num_vectors {
            offsets.push(pos);
            if pos >= mmap.len() {
                return Err(StorageError::InvalidSegment("ID table overflow".into()));
            }
            let len = mmap[pos] as usize;
            pos += 1 + len;
        }

        Ok(offsets)
    }

    /// Get vector data for a specific index.
    pub fn get_vector(&self, index: usize) -> &[f32] {
        let dim = self.header.dimension as usize;
        let offset = self.header.vector_data_offset as usize + index * dim * 4;
        let slice = &self.mmap[offset..offset + dim * 4];
        // Safety: we're interpreting bytes as f32, which is safe for aligned data
        unsafe {
            std::slice::from_raw_parts(slice.as_ptr() as *const f32, dim)
        }
    }

    /// Get ID for a specific index.
    pub fn get_id(&self, index: usize) -> StorageResult<VectorId> {
        let offset = self.id_offsets[index];
        let (id, _) = VectorId::from_bytes(&self.mmap[offset..])
            .ok_or_else(|| StorageError::InvalidId(format!("Invalid ID at index {}", index)))?;
        Ok(id)
    }

    /// Get payload for a specific index.
    pub fn get_payload(&self, index: usize) -> StorageResult<Option<serde_json::Value>> {
        let po_offset = self.header.payload_offset_table_offset as usize + index * PayloadOffset::SIZE;
        let po = PayloadOffset::from_bytes(&self.mmap[po_offset..]);

        if po.length == 0 {
            return Ok(None);
        }

        let blob_offset = self.header.payload_blob_offset as usize + po.offset as usize;
        let payload_bytes = &self.mmap[blob_offset..blob_offset + po.length as usize];
        let payload: serde_json::Value = serde_json::from_slice(payload_bytes)?;
        Ok(Some(payload))
    }

    /// Get number of vectors in segment.
    pub fn num_vectors(&self) -> usize {
        self.header.num_vectors as usize
    }

    /// Get vector dimension.
    pub fn dimension(&self) -> usize {
        self.header.dimension as usize
    }

    /// Get the metric used.
    pub fn metric(&self) -> Metric {
        self.header.metric
    }
}

/// Builder for creating segment files.
pub struct SegmentBuilder {
    dimension: usize,
    metric: Metric,
}

impl SegmentBuilder {
    pub fn new(dimension: usize, metric: Metric) -> Self {
        Self { dimension, metric }
    }

    /// Build a segment from clustered data.
    ///
    /// - `cluster_data`: Vec of (centroid, vector_indices)
    /// - `vectors`: Original vectors in insertion order
    /// - `ids`: Vector IDs in insertion order
    /// - `payloads`: Optional payloads in insertion order
    /// - `output_path`: Where to write the segment
    pub fn build(
        &self,
        cluster_data: Vec<(Vec<f32>, Vec<usize>)>,
        vectors: &[Vec<f32>],
        ids: &[VectorId],
        payloads: &[Option<serde_json::Value>],
        output_path: &Path,
    ) -> StorageResult<()> {
        let num_vectors = vectors.len();
        let num_clusters = cluster_data.len();

        // Validate dimensions
        for v in vectors.iter() {
            if v.len() != self.dimension {
                return Err(StorageError::DimensionMismatch {
                    expected: self.dimension,
                    got: v.len(),
                });
            }
        }

        // Build mapping from original index to cluster-ordered index
        let mut cluster_order: Vec<usize> = Vec::with_capacity(num_vectors);
        let mut cluster_metas: Vec<ClusterMeta> = Vec::with_capacity(num_clusters);

        let mut current_start = 0u32;
        for (centroid, indices) in &cluster_data {
            cluster_metas.push(ClusterMeta {
                centroid: centroid.clone(),
                start_index: current_start,
                length: indices.len() as u32,
            });
            cluster_order.extend(indices);
            current_start += indices.len() as u32;
        }

        // Calculate offsets
        let cluster_meta_offset = HEADER_SIZE as u64;
        let cluster_meta_size = num_clusters * ClusterMeta::byte_size(self.dimension);
        let vector_data_offset = cluster_meta_offset + cluster_meta_size as u64;
        let vector_data_size = num_vectors * self.dimension * 4;
        let id_table_offset = vector_data_offset + vector_data_size as u64;

        // Calculate ID table size
        let mut id_table_size = 0usize;
        for &orig_idx in &cluster_order {
            id_table_size += ids[orig_idx].to_bytes().len();
        }

        let payload_offset_table_offset = id_table_offset + id_table_size as u64;
        let payload_offset_table_size = num_vectors * PayloadOffset::SIZE;
        let payload_blob_offset = payload_offset_table_offset + payload_offset_table_size as u64;

        // Build payload blob and offset table
        let mut payload_blob = Vec::new();
        let mut payload_offsets: Vec<PayloadOffset> = Vec::with_capacity(num_vectors);

        for &orig_idx in &cluster_order {
            if let Some(Some(payload)) = payloads.get(orig_idx) {
                let json_bytes = serde_json::to_vec(payload)?;
                payload_offsets.push(PayloadOffset {
                    offset: payload_blob.len() as u64,
                    length: json_bytes.len() as u32,
                });
                payload_blob.extend(json_bytes);
            } else {
                payload_offsets.push(PayloadOffset { offset: 0, length: 0 });
            }
        }

        // Create header
        let header = SegmentHeader {
            version: SEGMENT_VERSION,
            dimension: self.dimension as u32,
            metric: self.metric,
            num_vectors: num_vectors as u32,
            num_clusters: num_clusters as u32,
            cluster_meta_offset,
            vector_data_offset,
            id_table_offset,
            payload_offset_table_offset,
            payload_blob_offset,
        };

        // Write to temporary file first
        let tmp_path = output_path.with_extension("tmp");
        {
            let file = File::create(&tmp_path)?;
            let mut writer = BufWriter::new(file);

            // Write header
            writer.write_all(&header.to_bytes())?;

            // Write cluster metadata
            for cm in &cluster_metas {
                writer.write_all(&cm.to_bytes())?;
            }

            // Write vector data in cluster order
            for &orig_idx in &cluster_order {
                for &v in &vectors[orig_idx] {
                    writer.write_all(&v.to_le_bytes())?;
                }
            }

            // Write ID table
            for &orig_idx in &cluster_order {
                writer.write_all(&ids[orig_idx].to_bytes())?;
            }

            // Write payload offset table
            for po in &payload_offsets {
                writer.write_all(&po.to_bytes())?;
            }

            // Write payload blob
            writer.write_all(&payload_blob)?;

            writer.flush()?;
        }

        // Atomic rename
        fs::rename(&tmp_path, output_path)?;

        tracing::info!(
            "Built segment: {} vectors, {} clusters, {} bytes",
            num_vectors,
            num_clusters,
            payload_blob_offset + payload_blob.len() as u64
        );

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_header_roundtrip() {
        let header = SegmentHeader {
            version: SEGMENT_VERSION,
            dimension: 128,
            metric: Metric::Cosine,
            num_vectors: 1000,
            num_clusters: 10,
            cluster_meta_offset: 64,
            vector_data_offset: 1000,
            id_table_offset: 50000,
            payload_offset_table_offset: 60000,
            payload_blob_offset: 70000,
        };

        let bytes = header.to_bytes();
        let parsed = SegmentHeader::from_bytes(&bytes).unwrap();

        assert_eq!(parsed.version, header.version);
        assert_eq!(parsed.dimension, header.dimension);
        assert_eq!(parsed.metric, header.metric);
        assert_eq!(parsed.num_vectors, header.num_vectors);
        assert_eq!(parsed.num_clusters, header.num_clusters);
    }

    #[test]
    fn test_cluster_meta_roundtrip() {
        let cm = ClusterMeta {
            centroid: vec![1.0, 2.0, 3.0, 4.0],
            start_index: 100,
            length: 50,
        };

        let bytes = cm.to_bytes();
        let parsed = ClusterMeta::from_bytes(&bytes, 4).unwrap();

        assert_eq!(parsed.centroid, cm.centroid);
        assert_eq!(parsed.start_index, cm.start_index);
        assert_eq!(parsed.length, cm.length);
    }

    #[test]
    fn test_segment_build_and_load() {
        let dir = tempdir().unwrap();
        let segment_path = dir.path().join("test.seg");

        let dim = 4;
        let vectors = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ];
        let ids: Vec<VectorId> = (0..4).map(|i| VectorId::new(format!("vec_{}", i))).collect();
        let payloads = vec![
            Some(serde_json::json!({"idx": 0})),
            None,
            Some(serde_json::json!({"idx": 2})),
            None,
        ];

        // Single cluster containing all vectors
        let cluster_data = vec![
            (vec![0.25, 0.25, 0.25, 0.25], vec![0, 1, 2, 3]),
        ];

        let builder = SegmentBuilder::new(dim, Metric::L2);
        builder.build(cluster_data, &vectors, &ids, &payloads, &segment_path).unwrap();

        // Load and verify
        let segment = LoadedSegment::open(&segment_path).unwrap();
        assert_eq!(segment.num_vectors(), 4);
        assert_eq!(segment.dimension(), 4);
        assert_eq!(segment.clusters.len(), 1);

        // Check vectors
        let v0 = segment.get_vector(0);
        assert_eq!(v0, &[1.0, 0.0, 0.0, 0.0]);

        // Check IDs
        let id0 = segment.get_id(0).unwrap();
        assert_eq!(id0.as_str(), "vec_0");

        // Check payloads
        let p0 = segment.get_payload(0).unwrap();
        assert!(p0.is_some());
        assert_eq!(p0.unwrap()["idx"], 0);

        let p1 = segment.get_payload(1).unwrap();
        assert!(p1.is_none());
    }
}
