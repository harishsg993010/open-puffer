//! PQ configuration types.

use serde::{Deserialize, Serialize};

/// Parameters for Product Quantization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PqParams {
    /// Number of subvectors (M). The dimension must be divisible by M.
    /// Typical values: 4, 8, 16, 32
    pub num_subvectors: usize,

    /// Number of centroids per subvector (K).
    /// Typically 256 (fits in u8) or 65536 (fits in u16).
    pub codebook_size: usize,

    /// Number of training samples to use. If None, use all provided vectors.
    pub training_samples: Option<usize>,

    /// Maximum iterations for k-means training.
    pub max_iterations: usize,

    /// Convergence threshold for k-means.
    pub convergence_threshold: f64,

    /// Random seed for reproducibility.
    pub seed: Option<u64>,
}

impl PqParams {
    /// Create new PQ parameters with defaults.
    pub fn new(num_subvectors: usize, codebook_size: usize) -> Self {
        Self {
            num_subvectors,
            codebook_size,
            training_samples: None,
            max_iterations: 25,
            convergence_threshold: 0.001,
            seed: None,
        }
    }

    /// Set number of training samples.
    pub fn with_training_samples(mut self, samples: usize) -> Self {
        self.training_samples = Some(samples);
        self
    }

    /// Set maximum iterations.
    pub fn with_max_iterations(mut self, iterations: usize) -> Self {
        self.max_iterations = iterations;
        self
    }

    /// Set random seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Validate parameters against a dimension.
    pub fn validate(&self, dim: usize) -> Result<(), String> {
        if self.num_subvectors == 0 {
            return Err("num_subvectors must be > 0".to_string());
        }
        if self.codebook_size == 0 {
            return Err("codebook_size must be > 0".to_string());
        }
        if dim % self.num_subvectors != 0 {
            return Err(format!(
                "dimension {} is not divisible by num_subvectors {}",
                dim, self.num_subvectors
            ));
        }
        if self.codebook_size > 65536 {
            return Err("codebook_size must be <= 65536 (fits in u16)".to_string());
        }
        Ok(())
    }

    /// Get the subvector dimension.
    pub fn subvector_dim(&self, dim: usize) -> usize {
        dim / self.num_subvectors
    }
}

impl Default for PqParams {
    fn default() -> Self {
        Self::new(8, 256)
    }
}

/// Parameters for Optimized Product Quantization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpqParams {
    /// Base PQ parameters.
    pub pq: PqParams,

    /// Number of OPQ iterations (alternating between rotation and PQ).
    pub opq_iterations: usize,

    /// Whether to use a full rotation matrix or block-diagonal.
    pub full_rotation: bool,
}

impl OpqParams {
    /// Create new OPQ parameters.
    pub fn new(pq: PqParams) -> Self {
        Self {
            pq,
            opq_iterations: 10,
            full_rotation: true,
        }
    }

    /// Set number of OPQ iterations.
    pub fn with_opq_iterations(mut self, iterations: usize) -> Self {
        self.opq_iterations = iterations;
        self
    }

    /// Use block-diagonal rotation instead of full rotation.
    pub fn with_block_rotation(mut self) -> Self {
        self.full_rotation = false;
        self
    }
}

impl Default for OpqParams {
    fn default() -> Self {
        Self::new(PqParams::default())
    }
}

/// Quantization mode for a collection.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum QuantizationMode {
    /// No quantization, store full vectors.
    #[default]
    None,

    /// Product Quantization.
    PQ(PqParams),

    /// Optimized Product Quantization.
    OPQ(OpqParams),
}

impl QuantizationMode {
    /// Check if quantization is enabled.
    pub fn is_enabled(&self) -> bool {
        !matches!(self, QuantizationMode::None)
    }

    /// Get PQ parameters if using PQ or OPQ.
    pub fn pq_params(&self) -> Option<&PqParams> {
        match self {
            QuantizationMode::None => None,
            QuantizationMode::PQ(p) => Some(p),
            QuantizationMode::OPQ(o) => Some(&o.pq),
        }
    }
}

/// Full quantization configuration including storage options.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QuantizationConfig {
    /// The quantization mode to use.
    pub mode: QuantizationMode,

    /// Whether to store full vectors alongside PQ codes for refinement.
    pub store_full_vectors: bool,

    /// Number of candidates to refine with full vectors (0 = no refinement).
    pub refinement_candidates: usize,

    /// Use SIMD for distance computation when available.
    pub use_simd: bool,
}

impl QuantizationConfig {
    /// Create a new configuration with PQ.
    pub fn pq(params: PqParams) -> Self {
        Self {
            mode: QuantizationMode::PQ(params),
            store_full_vectors: true,
            refinement_candidates: 0,
            use_simd: true,
        }
    }

    /// Create a new configuration with OPQ.
    pub fn opq(params: OpqParams) -> Self {
        Self {
            mode: QuantizationMode::OPQ(params),
            store_full_vectors: true,
            refinement_candidates: 0,
            use_simd: true,
        }
    }

    /// Enable refinement with the given number of candidates.
    pub fn with_refinement(mut self, candidates: usize) -> Self {
        self.refinement_candidates = candidates;
        self.store_full_vectors = true;
        self
    }

    /// Disable storing full vectors (PQ-only mode).
    pub fn without_full_vectors(mut self) -> Self {
        self.store_full_vectors = false;
        self.refinement_candidates = 0;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pq_params_validation() {
        let params = PqParams::new(8, 256);
        assert!(params.validate(128).is_ok());
        assert!(params.validate(64).is_ok());
        assert!(params.validate(100).is_err()); // Not divisible by 8
    }

    #[test]
    fn test_subvector_dim() {
        let params = PqParams::new(8, 256);
        assert_eq!(params.subvector_dim(128), 16);
        assert_eq!(params.subvector_dim(768), 96);
    }

    #[test]
    fn test_quantization_mode_serde() {
        let mode = QuantizationMode::PQ(PqParams::new(8, 256));
        let json = serde_json::to_string(&mode).unwrap();
        let parsed: QuantizationMode = serde_json::from_str(&json).unwrap();
        assert!(matches!(parsed, QuantizationMode::PQ(_)));
    }
}
