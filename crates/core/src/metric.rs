//! Distance metric definitions.

use serde::{Deserialize, Serialize};

/// Supported distance metrics for vector similarity search.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Metric {
    /// Euclidean (L2) distance - lower is more similar
    L2,
    /// Cosine distance (1 - cosine_similarity) - lower is more similar
    Cosine,
}

impl Metric {
    /// Convert metric to byte representation for storage.
    pub fn to_byte(self) -> u8 {
        match self {
            Metric::L2 => 0,
            Metric::Cosine => 1,
        }
    }

    /// Parse metric from byte representation.
    pub fn from_byte(b: u8) -> Option<Self> {
        match b {
            0 => Some(Metric::L2),
            1 => Some(Metric::Cosine),
            _ => None,
        }
    }
}

impl Default for Metric {
    fn default() -> Self {
        Metric::Cosine
    }
}

impl std::fmt::Display for Metric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Metric::L2 => write!(f, "l2"),
            Metric::Cosine => write!(f, "cosine"),
        }
    }
}

impl std::str::FromStr for Metric {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "l2" | "euclidean" => Ok(Metric::L2),
            "cosine" => Ok(Metric::Cosine),
            _ => Err(format!("Unknown metric: {}", s)),
        }
    }
}
