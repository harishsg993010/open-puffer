//! Optimized Product Quantization (OPQ) with learned rotation matrix.
//!
//! OPQ improves upon PQ by learning a rotation matrix R that minimizes
//! quantization error. The optimization alternates between:
//! 1. Fixing R, optimizing PQ codebooks
//! 2. Fixing codebooks, optimizing R via SVD
//!
//! Reference: "Optimized Product Quantization" by Ge et al., CVPR 2013

use crate::codebook::{train_pq, PqCodebook};
use crate::config::{OpqParams, PqParams};
use crate::encoding::{decode_vector, encode_vectors};
use crate::error::{PqError, PqResult};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// OPQ transformation with rotation matrix and PQ codebook.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpqTransform {
    /// The learned rotation matrix (dim × dim), stored row-major.
    pub rotation_matrix: Vec<f32>,

    /// The PQ codebook trained on rotated vectors.
    pub codebook: PqCodebook,

    /// Original vector dimension.
    pub dim: usize,

    /// Whether to use full rotation or block-diagonal.
    pub full_rotation: bool,
}

impl OpqTransform {
    /// Get the rotation matrix as a 2D slice.
    pub fn rotation(&self) -> &[f32] {
        &self.rotation_matrix
    }

    /// Apply rotation to a single vector: y = R * x
    pub fn rotate(&self, vector: &[f32]) -> Vec<f32> {
        apply_rotation(&self.rotation_matrix, vector, self.dim)
    }

    /// Apply inverse rotation (R^T * y since R is orthogonal): x = R^T * y
    pub fn rotate_inverse(&self, vector: &[f32]) -> Vec<f32> {
        apply_rotation_transpose(&self.rotation_matrix, vector, self.dim)
    }

    /// Rotate and encode a vector.
    pub fn encode(&self, vector: &[f32]) -> PqResult<Vec<u16>> {
        let rotated = self.rotate(vector);
        crate::encoding::encode_vector(&self.codebook, &rotated)
    }

    /// Decode and inverse-rotate a code.
    pub fn decode(&self, codes: &[u16]) -> PqResult<Vec<f32>> {
        let decoded = decode_vector(&self.codebook, codes)?;
        Ok(self.rotate_inverse(&decoded))
    }

    /// Serialize to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // Header
        bytes.extend_from_slice(&(self.dim as u32).to_le_bytes());
        bytes.push(self.full_rotation as u8);

        // Rotation matrix
        let rotation_size = if self.full_rotation {
            self.dim * self.dim
        } else {
            self.dim // Block diagonal stores only diagonal blocks
        };
        bytes.extend_from_slice(&(rotation_size as u32).to_le_bytes());
        for &r in &self.rotation_matrix[..rotation_size] {
            bytes.extend_from_slice(&r.to_le_bytes());
        }

        // Codebook
        let codebook_bytes = self.codebook.to_bytes();
        bytes.extend_from_slice(&(codebook_bytes.len() as u32).to_le_bytes());
        bytes.extend(codebook_bytes);

        bytes
    }

    /// Deserialize from bytes.
    pub fn from_bytes(data: &[u8]) -> PqResult<Self> {
        if data.len() < 9 {
            return Err(PqError::InvalidParams("OPQ data too short".into()));
        }

        let dim = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
        let full_rotation = data[4] != 0;
        let rotation_size = u32::from_le_bytes(data[5..9].try_into().unwrap()) as usize;

        let rotation_end = 9 + rotation_size * 4;
        if data.len() < rotation_end + 4 {
            return Err(PqError::InvalidParams("OPQ data truncated".into()));
        }

        let rotation_matrix: Vec<f32> = (0..rotation_size)
            .map(|i| {
                let offset = 9 + i * 4;
                f32::from_le_bytes(data[offset..offset + 4].try_into().unwrap())
            })
            .collect();

        let codebook_len = u32::from_le_bytes(
            data[rotation_end..rotation_end + 4].try_into().unwrap()
        ) as usize;

        let codebook_start = rotation_end + 4;
        if data.len() < codebook_start + codebook_len {
            return Err(PqError::InvalidParams("OPQ codebook data truncated".into()));
        }

        let codebook = PqCodebook::from_bytes(&data[codebook_start..codebook_start + codebook_len])?;

        Ok(Self {
            rotation_matrix,
            codebook,
            dim,
            full_rotation,
        })
    }
}

/// Train OPQ transformation.
///
/// Uses alternating optimization:
/// 1. Initialize R as identity
/// 2. Repeat:
///    a. Rotate vectors: X' = X * R
///    b. Train PQ on X'
///    c. Compute residuals: E = X' - decode(encode(X'))
///    d. Update R via SVD of X^T * E
pub fn train_opq(vectors: &[Vec<f32>], params: &OpqParams) -> PqResult<OpqTransform> {
    if vectors.is_empty() {
        return Err(PqError::InsufficientSamples { min: 1, got: 0 });
    }

    let dim = vectors[0].len();
    params.pq.validate(dim).map_err(PqError::InvalidParams)?;

    // Sample training vectors if specified
    let training_vectors: Vec<Vec<f32>> = match params.pq.training_samples {
        Some(n) if n < vectors.len() => {
            use rand::prelude::*;
            let mut rng = match params.pq.seed {
                Some(s) => StdRng::seed_from_u64(s),
                None => StdRng::from_entropy(),
            };
            let mut indices: Vec<usize> = (0..vectors.len()).collect();
            indices.shuffle(&mut rng);
            indices.truncate(n);
            indices.iter().map(|&i| vectors[i].clone()).collect()
        }
        _ => vectors.to_vec(),
    };

    // Initialize rotation matrix as identity
    let mut rotation_matrix = vec![0.0f32; dim * dim];
    for i in 0..dim {
        rotation_matrix[i * dim + i] = 1.0;
    }

    let mut best_codebook: Option<PqCodebook> = None;
    let mut best_error = f32::MAX;

    for iter in 0..params.opq_iterations {
        // Rotate training vectors
        let rotated_vectors: Vec<Vec<f32>> = training_vectors
            .par_iter()
            .map(|v| apply_rotation(&rotation_matrix, v, dim))
            .collect();

        // Train PQ on rotated vectors
        let codebook = train_pq(&rotated_vectors, &params.pq)?;

        // Encode and decode to get reconstructions
        let codes = encode_vectors(&codebook, &rotated_vectors)?;
        let reconstructed: Vec<Vec<f32>> = codes
            .par_iter()
            .map(|c| decode_vector(&codebook, c).unwrap())
            .collect();

        // Compute reconstruction error
        let error = compute_reconstruction_error(&rotated_vectors, &reconstructed);

        tracing::debug!(
            "OPQ iteration {}: reconstruction error = {:.6}",
            iter + 1,
            error
        );

        if error < best_error {
            best_error = error;
            best_codebook = Some(codebook.clone());
        }

        // Update rotation matrix if not the last iteration
        if iter + 1 < params.opq_iterations {
            rotation_matrix = update_rotation_matrix(
                &training_vectors,
                &reconstructed,
                &rotation_matrix,
                dim,
            );
        }
    }

    let codebook = best_codebook.ok_or(PqError::ConvergenceFailure {
        iterations: params.opq_iterations,
    })?;

    Ok(OpqTransform {
        rotation_matrix,
        codebook,
        dim,
        full_rotation: params.full_rotation,
    })
}

/// Apply rotation matrix: y = R * x
fn apply_rotation(rotation: &[f32], vector: &[f32], dim: usize) -> Vec<f32> {
    let mut result = vec![0.0f32; dim];

    for i in 0..dim {
        let mut sum = 0.0f32;
        for j in 0..dim {
            sum += rotation[i * dim + j] * vector[j];
        }
        result[i] = sum;
    }

    result
}

/// Apply rotation matrix transpose: y = R^T * x
fn apply_rotation_transpose(rotation: &[f32], vector: &[f32], dim: usize) -> Vec<f32> {
    let mut result = vec![0.0f32; dim];

    for i in 0..dim {
        let mut sum = 0.0f32;
        for j in 0..dim {
            sum += rotation[j * dim + i] * vector[j];
        }
        result[i] = sum;
    }

    result
}

/// Compute mean squared reconstruction error.
fn compute_reconstruction_error(original: &[Vec<f32>], reconstructed: &[Vec<f32>]) -> f32 {
    if original.is_empty() {
        return 0.0;
    }

    let total_error: f64 = original
        .par_iter()
        .zip(reconstructed.par_iter())
        .map(|(o, r)| {
            o.iter()
                .zip(r.iter())
                .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
                .sum::<f64>()
        })
        .sum();

    (total_error / original.len() as f64) as f32
}

/// Update rotation matrix using gradient descent on orthogonal manifold.
///
/// This is a simplified version - full OPQ would use SVD-based Procrustes solution.
fn update_rotation_matrix(
    original: &[Vec<f32>],
    reconstructed: &[Vec<f32>],
    current_rotation: &[f32],
    dim: usize,
) -> Vec<f32> {
    // Compute X^T * Y where X is original and Y is reconstructed
    let mut xty = vec![0.0f64; dim * dim];

    for (x, y) in original.iter().zip(reconstructed.iter()) {
        // Apply current rotation to get original in rotated space
        let x_rotated = apply_rotation(current_rotation, x, dim);

        for i in 0..dim {
            for j in 0..dim {
                xty[i * dim + j] += x_rotated[i] as f64 * y[j] as f64;
            }
        }
    }

    // Simple orthogonalization via Gram-Schmidt
    // For production, use SVD-based Procrustes solution
    let mut new_rotation = vec![0.0f32; dim * dim];

    for i in 0..dim {
        // Start with the i-th row of XTY
        let mut row: Vec<f64> = (0..dim).map(|j| xty[i * dim + j]).collect();

        // Subtract projections onto previous rows
        for k in 0..i {
            let prev_row: Vec<f64> = (0..dim)
                .map(|j| new_rotation[k * dim + j] as f64)
                .collect();

            let dot: f64 = row.iter().zip(prev_row.iter()).map(|(a, b)| a * b).sum();
            for j in 0..dim {
                row[j] -= dot * prev_row[j];
            }
        }

        // Normalize
        let norm: f64 = row.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-10 {
            for j in 0..dim {
                new_rotation[i * dim + j] = (row[j] / norm) as f32;
            }
        } else {
            // Fallback to identity row
            new_rotation[i * dim + i] = 1.0;
        }
    }

    new_rotation
}

/// Apply OPQ rotation to a batch of vectors.
pub fn apply_opq_rotation(transform: &OpqTransform, vectors: &[Vec<f32>]) -> Vec<Vec<f32>> {
    vectors
        .par_iter()
        .map(|v| transform.rotate(v))
        .collect()
}

/// Encode vectors with OPQ.
pub fn encode_opq(transform: &OpqTransform, vectors: &[Vec<f32>]) -> PqResult<Vec<Vec<u16>>> {
    let rotated = apply_opq_rotation(transform, vectors);
    encode_vectors(&transform.codebook, &rotated)
}

/// Decode OPQ codes back to vectors.
pub fn decode_opq(transform: &OpqTransform, codes: &[Vec<u16>]) -> PqResult<Vec<Vec<f32>>> {
    codes
        .par_iter()
        .map(|c| transform.decode(c))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_random_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        use rand::prelude::*;
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect()
    }

    #[test]
    fn test_rotation_identity() {
        let dim = 4;
        let mut identity = vec![0.0f32; dim * dim];
        for i in 0..dim {
            identity[i * dim + i] = 1.0;
        }

        let vector = vec![1.0, 2.0, 3.0, 4.0];
        let rotated = apply_rotation(&identity, &vector, dim);

        for (a, b) in vector.iter().zip(rotated.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_rotation_roundtrip() {
        let dim = 4;
        // 90-degree rotation in xy plane
        let mut rotation = vec![0.0f32; dim * dim];
        rotation[0 * dim + 1] = 1.0;  // x -> y
        rotation[1 * dim + 0] = -1.0; // y -> -x
        rotation[2 * dim + 2] = 1.0;
        rotation[3 * dim + 3] = 1.0;

        let vector = vec![1.0, 0.0, 0.0, 0.0];
        let rotated = apply_rotation(&rotation, &vector, dim);
        let restored = apply_rotation_transpose(&rotation, &rotated, dim);

        for (a, b) in vector.iter().zip(restored.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_train_opq() {
        let vectors = generate_random_vectors(500, 16, 42);
        let pq_params = PqParams::new(4, 16).with_max_iterations(10);
        let opq_params = OpqParams::new(pq_params).with_opq_iterations(3);

        let transform = train_opq(&vectors, &opq_params).unwrap();

        assert_eq!(transform.dim, 16);
        assert!(transform.codebook.trained);
    }

    #[test]
    fn test_opq_encode_decode() {
        let vectors = generate_random_vectors(200, 16, 42);
        let pq_params = PqParams::new(4, 16).with_max_iterations(10);
        let opq_params = OpqParams::new(pq_params).with_opq_iterations(3);

        let transform = train_opq(&vectors, &opq_params).unwrap();

        let codes = transform.encode(&vectors[0]).unwrap();
        let decoded = transform.decode(&codes).unwrap();

        assert_eq!(codes.len(), 4);
        assert_eq!(decoded.len(), 16);
    }

    #[test]
    fn test_opq_serialization() {
        let vectors = generate_random_vectors(200, 16, 42);
        let pq_params = PqParams::new(4, 8).with_max_iterations(5);
        let opq_params = OpqParams::new(pq_params).with_opq_iterations(2);

        let transform = train_opq(&vectors, &opq_params).unwrap();
        let bytes = transform.to_bytes();
        let restored = OpqTransform::from_bytes(&bytes).unwrap();

        assert_eq!(transform.dim, restored.dim);
        assert_eq!(transform.rotation_matrix.len(), restored.rotation_matrix.len());
    }
}
