//! HNSW configuration.

use crate::error::{HnswError, HnswResult};
use serde::{Deserialize, Serialize};

/// Configuration for HNSW index construction and search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswConfig {
    /// Maximum number of connections per node at layer 0.
    /// Higher values = better recall but slower construction and more memory.
    /// Typical range: 12-48, default: 16
    pub m: usize,

    /// Maximum number of connections per node at layers > 0.
    /// Usually M, computed as m * 2.
    pub m_max: usize,

    /// Maximum number of connections at layer 0.
    /// Usually 2 * m.
    pub m_max_0: usize,

    /// Size of dynamic candidate list during construction.
    /// Higher values = better graph quality but slower construction.
    /// Typical range: 100-500, default: 200
    pub ef_construction: usize,

    /// Size of dynamic candidate list during search.
    /// Higher values = better recall but slower search.
    /// Typical range: 10-500, default: 50
    pub ef_search: usize,

    /// Level multiplier for probabilistic layer selection.
    /// Computed as 1 / ln(M).
    pub ml: f64,

    /// Random seed for reproducibility.
    pub seed: Option<u64>,

    /// Enable parallel construction.
    pub parallel: bool,

    /// Number of threads for parallel construction (0 = auto).
    pub num_threads: usize,
}

impl HnswConfig {
    /// Create new config with default values.
    pub fn new(m: usize) -> Self {
        let ml = 1.0 / (m as f64).ln();
        Self {
            m,
            m_max: m,
            m_max_0: m * 2,
            ef_construction: 200,
            ef_search: 50,
            ml,
            seed: None,
            parallel: true,
            num_threads: 0,
        }
    }

    /// Set ef_construction.
    pub fn with_ef_construction(mut self, ef: usize) -> Self {
        self.ef_construction = ef;
        self
    }

    /// Set ef_search.
    pub fn with_ef_search(mut self, ef: usize) -> Self {
        self.ef_search = ef;
        self
    }

    /// Set random seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Disable parallel construction.
    pub fn sequential(mut self) -> Self {
        self.parallel = false;
        self
    }

    /// Validate configuration.
    pub fn validate(&self) -> HnswResult<()> {
        if self.m == 0 {
            return Err(HnswError::InvalidParams("M must be > 0".into()));
        }
        if self.ef_construction < self.m {
            return Err(HnswError::InvalidParams(
                "ef_construction must be >= M".into(),
            ));
        }
        if self.ef_search == 0 {
            return Err(HnswError::InvalidParams("ef_search must be > 0".into()));
        }
        Ok(())
    }

    /// Serialize to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(self.m as u32).to_le_bytes());
        bytes.extend_from_slice(&(self.m_max as u32).to_le_bytes());
        bytes.extend_from_slice(&(self.m_max_0 as u32).to_le_bytes());
        bytes.extend_from_slice(&(self.ef_construction as u32).to_le_bytes());
        bytes.extend_from_slice(&(self.ef_search as u32).to_le_bytes());
        bytes.extend_from_slice(&self.ml.to_le_bytes());

        let seed = self.seed.unwrap_or(u64::MAX);
        bytes.extend_from_slice(&seed.to_le_bytes());

        bytes.push(self.parallel as u8);
        bytes.extend_from_slice(&(self.num_threads as u32).to_le_bytes());
        bytes
    }

    /// Deserialize from bytes.
    pub fn from_bytes(data: &[u8]) -> HnswResult<Self> {
        // 4+4+4+4+4+8+8+1+4 = 41 bytes
        if data.len() < 41 {
            return Err(HnswError::InvalidData("Config data too short".into()));
        }

        let m = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
        let m_max = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;
        let m_max_0 = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;
        let ef_construction = u32::from_le_bytes(data[12..16].try_into().unwrap()) as usize;
        let ef_search = u32::from_le_bytes(data[16..20].try_into().unwrap()) as usize;
        let ml = f64::from_le_bytes(data[20..28].try_into().unwrap());
        let seed_raw = u64::from_le_bytes(data[28..36].try_into().unwrap());
        let seed = if seed_raw == u64::MAX { None } else { Some(seed_raw) };
        let parallel = data[36] != 0;
        let num_threads = u32::from_le_bytes(data[37..41].try_into().unwrap()) as usize;

        Ok(Self {
            m,
            m_max,
            m_max_0,
            ef_construction,
            ef_search,
            ml,
            seed,
            parallel,
            num_threads,
        })
    }

    /// Get max connections for a given layer.
    pub fn max_connections(&self, layer: usize) -> usize {
        if layer == 0 {
            self.m_max_0
        } else {
            self.m_max
        }
    }
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self::new(16)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_serialization() {
        let config = HnswConfig::new(24)
            .with_ef_construction(300)
            .with_ef_search(100)
            .with_seed(42);

        let bytes = config.to_bytes();
        let restored = HnswConfig::from_bytes(&bytes).unwrap();

        assert_eq!(config.m, restored.m);
        assert_eq!(config.ef_construction, restored.ef_construction);
        assert_eq!(config.ef_search, restored.ef_search);
        assert_eq!(config.seed, restored.seed);
    }

    #[test]
    fn test_validation() {
        let config = HnswConfig::new(16);
        assert!(config.validate().is_ok());

        let bad_config = HnswConfig {
            m: 0,
            ..Default::default()
        };
        assert!(bad_config.validate().is_err());
    }
}
