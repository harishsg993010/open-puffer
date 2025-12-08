//! Core type definitions.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Unique identifier for a vector.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VectorId(pub String);

impl VectorId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Convert to bytes for storage (max 256 bytes).
    pub fn to_bytes(&self) -> Vec<u8> {
        let bytes = self.0.as_bytes();
        let len = bytes.len().min(255) as u8;
        let mut result = vec![len];
        result.extend_from_slice(&bytes[..len as usize]);
        result
    }

    /// Parse from bytes.
    pub fn from_bytes(data: &[u8]) -> Option<(Self, usize)> {
        if data.is_empty() {
            return None;
        }
        let len = data[0] as usize;
        if data.len() < 1 + len {
            return None;
        }
        let s = std::str::from_utf8(&data[1..1 + len]).ok()?;
        Some((Self(s.to_string()), 1 + len))
    }
}

impl fmt::Display for VectorId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<String> for VectorId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for VectorId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

/// A vector record with ID, vector data, and optional payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorRecord {
    /// Unique identifier for this vector.
    pub id: VectorId,
    /// The vector data.
    pub vector: Vec<f32>,
    /// Optional JSON payload.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payload: Option<serde_json::Value>,
}

impl VectorRecord {
    pub fn new(id: impl Into<VectorId>, vector: Vec<f32>) -> Self {
        Self {
            id: id.into(),
            vector,
            payload: None,
        }
    }

    pub fn with_payload(mut self, payload: serde_json::Value) -> Self {
        self.payload = Some(payload);
        self
    }

    pub fn dimension(&self) -> usize {
        self.vector.len()
    }
}
