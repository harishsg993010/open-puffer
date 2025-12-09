//! Unified configuration for Puffer vector database features.
//!
//! This module provides a single configuration structure that encompasses
//! all the advanced features: PQ, HNSW, FTS, hybrid search, etc.

use serde::{Deserialize, Serialize};

/// Master configuration for a Puffer collection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PufferConfig {
    /// Basic collection settings.
    #[serde(default)]
    pub collection: CollectionSettings,

    /// Product Quantization settings.
    #[serde(default)]
    pub pq: PqSettings,

    /// HNSW index settings.
    #[serde(default)]
    pub hnsw: HnswSettings,

    /// Full-text search settings.
    #[serde(default)]
    pub fts: FtsSettings,

    /// Hybrid search settings.
    #[serde(default)]
    pub hybrid: HybridSettings,

    /// Query execution settings.
    #[serde(default)]
    pub query: QuerySettings,

    /// Performance tuning settings.
    #[serde(default)]
    pub performance: PerformanceSettings,
}

impl Default for PufferConfig {
    fn default() -> Self {
        Self {
            collection: CollectionSettings::default(),
            pq: PqSettings::default(),
            hnsw: HnswSettings::default(),
            fts: FtsSettings::default(),
            hybrid: HybridSettings::default(),
            query: QuerySettings::default(),
            performance: PerformanceSettings::default(),
        }
    }
}

/// Basic collection settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionSettings {
    /// Staging buffer size before flushing to segment.
    pub staging_threshold: usize,

    /// Base number of clusters for IVF partitioning.
    pub num_clusters: usize,

    /// Maximum L0 segments before triggering compaction.
    pub l0_max_segments: usize,

    /// Target size for compacted segments.
    pub segment_target_size: usize,

    /// Number of top segments to search via router.
    pub router_top_m: usize,
}

impl Default for CollectionSettings {
    fn default() -> Self {
        Self {
            staging_threshold: 10_000,
            num_clusters: 100,
            l0_max_segments: 10,
            segment_target_size: 100_000,
            router_top_m: 5,
        }
    }
}

/// Product Quantization settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PqSettings {
    /// Enable PQ compression.
    pub enabled: bool,

    /// Number of subvectors (dimension / subvector_dim).
    pub num_subvectors: usize,

    /// Codebook size per subvector (typically 256).
    pub codebook_size: usize,

    /// Enable OPQ (rotation optimization).
    pub enable_opq: bool,

    /// Number of OPQ training iterations.
    pub opq_iterations: usize,

    /// Training samples for codebook (None = use all).
    pub training_samples: Option<usize>,

    /// Use residual coding (subtract centroid before encoding).
    pub use_residual: bool,

    /// Number of candidates to refine with exact distances.
    pub refine_candidates: Option<usize>,
}

impl Default for PqSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            num_subvectors: 8,
            codebook_size: 256,
            enable_opq: false,
            opq_iterations: 10,
            training_samples: Some(10_000),
            use_residual: true,
            refine_candidates: Some(100),
        }
    }
}

/// HNSW index settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswSettings {
    /// Enable HNSW for intra-cluster search.
    pub enabled: bool,

    /// Maximum connections per node per layer.
    pub m: usize,

    /// Maximum connections on layer 0.
    pub m_max_0: usize,

    /// Size of dynamic candidate list during construction.
    pub ef_construction: usize,

    /// Size of dynamic candidate list during search.
    pub ef_search: usize,

    /// Level multiplier for layer assignment.
    pub ml: f64,

    /// Minimum cluster size to build HNSW (smaller clusters use brute-force).
    pub min_cluster_size: usize,
}

impl Default for HnswSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            m: 16,
            m_max_0: 32,
            ef_construction: 100,
            ef_search: 50,
            ml: 1.0 / (16.0_f64).ln(),
            min_cluster_size: 32,
        }
    }
}

/// Full-text search settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FtsSettings {
    /// Enable full-text search.
    pub enabled: bool,

    /// Fields to index for text search.
    pub indexed_fields: Vec<String>,

    /// Default field for queries.
    pub default_field: String,

    /// Enable stemming.
    pub enable_stemming: bool,

    /// Stemmer language.
    pub stemmer_language: String,

    /// Store original text for retrieval.
    pub store_text: bool,

    /// Enable phrase queries.
    pub enable_positions: bool,

    /// BM25 k1 parameter.
    pub bm25_k1: f32,

    /// BM25 b parameter.
    pub bm25_b: f32,

    /// Writer heap size in bytes.
    pub writer_heap_size: usize,

    /// Auto-commit after this many documents.
    pub auto_commit_threshold: usize,
}

impl Default for FtsSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            indexed_fields: vec!["text".to_string()],
            default_field: "text".to_string(),
            enable_stemming: true,
            stemmer_language: "english".to_string(),
            store_text: true,
            enable_positions: true,
            bm25_k1: 1.2,
            bm25_b: 0.75,
            writer_heap_size: 50_000_000,
            auto_commit_threshold: 1000,
        }
    }
}

/// Hybrid search settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridSettings {
    /// Default lambda for text/vector weighting (0=vector only, 1=text only).
    pub default_lambda: f32,

    /// Default fusion method.
    pub fusion_method: FusionMethodConfig,

    /// Number of candidates to retrieve from each source.
    pub candidates_per_source: usize,
}

impl Default for HybridSettings {
    fn default() -> Self {
        Self {
            default_lambda: 0.5,
            fusion_method: FusionMethodConfig::WeightedSum,
            candidates_per_source: 100,
        }
    }
}

/// Fusion method configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum FusionMethodConfig {
    WeightedSum,
    ReciprocalRankFusion { k: f32 },
    NormalizedWeightedSum,
    SoftmaxFusion { temperature: f32 },
}

impl Default for FusionMethodConfig {
    fn default() -> Self {
        Self::WeightedSum
    }
}

/// Query execution settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuerySettings {
    /// Default number of clusters to probe.
    pub default_nprobe: usize,

    /// Enable async refinement for improved recall.
    pub enable_refinement: bool,

    /// Initial clusters for fast path.
    pub refinement_initial_nprobe: usize,

    /// Additional clusters for refinement.
    pub refinement_extra_nprobe: usize,

    /// Initial timeout in ms.
    pub refinement_initial_timeout_ms: u64,

    /// Total refinement timeout in ms.
    pub refinement_total_timeout_ms: u64,
}

impl Default for QuerySettings {
    fn default() -> Self {
        Self {
            default_nprobe: 10,
            enable_refinement: false,
            refinement_initial_nprobe: 5,
            refinement_extra_nprobe: 15,
            refinement_initial_timeout_ms: 50,
            refinement_total_timeout_ms: 200,
        }
    }
}

/// Performance tuning settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSettings {
    /// Enable segment prefetching.
    pub enable_prefetching: bool,

    /// Number of segments to prefetch ahead.
    pub prefetch_ahead: usize,

    /// Maximum segments in warm cache.
    pub max_warm_segments: usize,

    /// Warm mmap pages on segment load.
    pub warm_on_load: bool,

    /// Page size for mmap warming.
    pub page_size: usize,

    /// Number of prefetch workers.
    pub prefetch_workers: usize,

    /// Enable parallel search across segments.
    pub parallel_search: bool,

    /// Maximum parallel search threads.
    pub max_search_threads: usize,
}

impl Default for PerformanceSettings {
    fn default() -> Self {
        Self {
            enable_prefetching: true,
            prefetch_ahead: 3,
            max_warm_segments: 10,
            warm_on_load: true,
            page_size: 4096,
            prefetch_workers: 2,
            parallel_search: true,
            max_search_threads: 0, // 0 = use all available cores
        }
    }
}

impl PufferConfig {
    /// Create a minimal config (no advanced features).
    pub fn minimal() -> Self {
        Self::default()
    }

    /// Create a config optimized for high recall.
    pub fn high_recall() -> Self {
        Self {
            query: QuerySettings {
                default_nprobe: 20,
                enable_refinement: true,
                refinement_extra_nprobe: 30,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// Create a config optimized for low latency.
    pub fn low_latency() -> Self {
        Self {
            pq: PqSettings {
                enabled: true,
                ..Default::default()
            },
            hnsw: HnswSettings {
                enabled: true,
                ef_search: 32,
                ..Default::default()
            },
            query: QuerySettings {
                default_nprobe: 5,
                enable_refinement: false,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// Create a config with full-text search enabled.
    pub fn with_fts() -> Self {
        Self {
            fts: FtsSettings {
                enabled: true,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// Builder: enable PQ.
    pub fn with_pq(mut self) -> Self {
        self.pq.enabled = true;
        self
    }

    /// Builder: enable HNSW.
    pub fn with_hnsw(mut self) -> Self {
        self.hnsw.enabled = true;
        self
    }

    /// Builder: enable refinement.
    pub fn with_refinement(mut self) -> Self {
        self.query.enable_refinement = true;
        self
    }

    /// Builder: enable full-text search.
    pub fn enable_fts(mut self) -> Self {
        self.fts.enabled = true;
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.collection.num_clusters == 0 {
            return Err("num_clusters must be > 0".to_string());
        }

        if self.pq.enabled && self.pq.num_subvectors == 0 {
            return Err("PQ num_subvectors must be > 0".to_string());
        }

        if self.hnsw.enabled && self.hnsw.m == 0 {
            return Err("HNSW m must be > 0".to_string());
        }

        if self.fts.enabled && self.fts.indexed_fields.is_empty() {
            return Err("FTS requires at least one indexed field".to_string());
        }

        if self.hybrid.default_lambda < 0.0 || self.hybrid.default_lambda > 1.0 {
            return Err("Hybrid lambda must be in [0, 1]".to_string());
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = PufferConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_high_recall_config() {
        let config = PufferConfig::high_recall();
        assert!(config.query.enable_refinement);
        assert_eq!(config.query.default_nprobe, 20);
    }

    #[test]
    fn test_low_latency_config() {
        let config = PufferConfig::low_latency();
        assert!(config.pq.enabled);
        assert!(config.hnsw.enabled);
    }

    #[test]
    fn test_builder_pattern() {
        let config = PufferConfig::default()
            .with_pq()
            .with_hnsw()
            .with_refinement()
            .enable_fts();

        assert!(config.pq.enabled);
        assert!(config.hnsw.enabled);
        assert!(config.query.enable_refinement);
        assert!(config.fts.enabled);
    }

    #[test]
    fn test_serialization() {
        let config = PufferConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let parsed: PufferConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.collection.num_clusters, config.collection.num_clusters);
    }
}
