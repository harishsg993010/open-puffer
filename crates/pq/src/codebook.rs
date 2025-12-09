//! PQ codebook training using k-means clustering.

use crate::config::PqParams;
use crate::error::{PqError, PqResult};
use puffer_core::distance::l2_distance_squared;
use rand::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// A trained Product Quantization codebook.
///
/// The codebook contains M sets of K centroids, where:
/// - M = number of subvectors
/// - K = codebook_size (centroids per subvector)
/// - D/M = subvector dimension
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PqCodebook {
    /// Number of subvectors (M).
    pub num_subvectors: usize,

    /// Number of centroids per subvector (K).
    pub codebook_size: usize,

    /// Original vector dimension.
    pub dim: usize,

    /// Dimension of each subvector (dim / num_subvectors).
    pub subvector_dim: usize,

    /// Centroids stored as [M][K][subvector_dim].
    /// Flattened layout: centroids[m * K * subvector_dim + k * subvector_dim + d]
    pub centroids: Vec<f32>,

    /// Whether the codebook has been trained.
    pub trained: bool,
}

impl PqCodebook {
    /// Create an empty (untrained) codebook.
    pub fn new(dim: usize, params: &PqParams) -> PqResult<Self> {
        params.validate(dim).map_err(PqError::InvalidParams)?;

        let subvector_dim = dim / params.num_subvectors;
        let total_centroids = params.num_subvectors * params.codebook_size * subvector_dim;

        Ok(Self {
            num_subvectors: params.num_subvectors,
            codebook_size: params.codebook_size,
            dim,
            subvector_dim,
            centroids: vec![0.0; total_centroids],
            trained: false,
        })
    }

    /// Get centroid for subvector m, code k.
    #[inline]
    pub fn get_centroid(&self, m: usize, k: usize) -> &[f32] {
        let offset = (m * self.codebook_size + k) * self.subvector_dim;
        &self.centroids[offset..offset + self.subvector_dim]
    }

    /// Get mutable centroid for subvector m, code k.
    #[inline]
    pub fn get_centroid_mut(&mut self, m: usize, k: usize) -> &mut [f32] {
        let offset = (m * self.codebook_size + k) * self.subvector_dim;
        &mut self.centroids[offset..offset + self.subvector_dim]
    }

    /// Get all centroids for subvector m as a contiguous slice.
    #[inline]
    pub fn get_subvector_centroids(&self, m: usize) -> &[f32] {
        let offset = m * self.codebook_size * self.subvector_dim;
        let size = self.codebook_size * self.subvector_dim;
        &self.centroids[offset..offset + size]
    }

    /// Get the starting index in the original vector for subvector m.
    #[inline]
    pub fn subvector_start(&self, m: usize) -> usize {
        m * self.subvector_dim
    }

    /// Extract subvector m from a vector.
    #[inline]
    pub fn extract_subvector<'a>(&self, vector: &'a [f32], m: usize) -> &'a [f32] {
        let start = self.subvector_start(m);
        &vector[start..start + self.subvector_dim]
    }

    /// Compute reconstruction error for a set of vectors.
    pub fn reconstruction_error(&self, vectors: &[Vec<f32>], codes: &[Vec<u16>]) -> f32 {
        if vectors.is_empty() {
            return 0.0;
        }

        let total_error: f32 = vectors
            .par_iter()
            .zip(codes.par_iter())
            .map(|(vec, code)| {
                let mut error = 0.0f32;
                for m in 0..self.num_subvectors {
                    let subvec = self.extract_subvector(vec, m);
                    let centroid = self.get_centroid(m, code[m] as usize);
                    error += l2_distance_squared(subvec, centroid);
                }
                error
            })
            .sum();

        total_error / vectors.len() as f32
    }

    /// Serialize codebook to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        // Format: [num_subvectors: u32][codebook_size: u32][dim: u32][subvector_dim: u32]
        //         [trained: u8][centroids: f32 array]
        let mut bytes = Vec::new();

        bytes.extend_from_slice(&(self.num_subvectors as u32).to_le_bytes());
        bytes.extend_from_slice(&(self.codebook_size as u32).to_le_bytes());
        bytes.extend_from_slice(&(self.dim as u32).to_le_bytes());
        bytes.extend_from_slice(&(self.subvector_dim as u32).to_le_bytes());
        bytes.push(self.trained as u8);

        for &c in &self.centroids {
            bytes.extend_from_slice(&c.to_le_bytes());
        }

        bytes
    }

    /// Deserialize codebook from bytes.
    pub fn from_bytes(data: &[u8]) -> PqResult<Self> {
        if data.len() < 17 {
            return Err(PqError::InvalidParams("Data too short for codebook".into()));
        }

        let num_subvectors = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
        let codebook_size = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;
        let dim = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;
        let subvector_dim = u32::from_le_bytes(data[12..16].try_into().unwrap()) as usize;
        let trained = data[16] != 0;

        let expected_centroids = num_subvectors * codebook_size * subvector_dim;
        let centroid_bytes = expected_centroids * 4;

        if data.len() < 17 + centroid_bytes {
            return Err(PqError::InvalidParams("Data too short for centroids".into()));
        }

        let centroids: Vec<f32> = (0..expected_centroids)
            .map(|i| {
                let offset = 17 + i * 4;
                f32::from_le_bytes(data[offset..offset + 4].try_into().unwrap())
            })
            .collect();

        Ok(Self {
            num_subvectors,
            codebook_size,
            dim,
            subvector_dim,
            centroids,
            trained,
        })
    }

    /// Get the size in bytes when serialized.
    pub fn byte_size(&self) -> usize {
        17 + self.centroids.len() * 4
    }
}

/// Train a PQ codebook using k-means clustering.
///
/// # Arguments
/// * `vectors` - Training vectors (should have at least codebook_size * 10 samples)
/// * `params` - PQ parameters
///
/// # Returns
/// A trained PQ codebook
pub fn train_pq(vectors: &[Vec<f32>], params: &PqParams) -> PqResult<PqCodebook> {
    if vectors.is_empty() {
        return Err(PqError::InsufficientSamples { min: 1, got: 0 });
    }

    let dim = vectors[0].len();
    params.validate(dim).map_err(PqError::InvalidParams)?;

    // Sample training vectors if specified
    let training_vectors: Vec<&Vec<f32>> = match params.training_samples {
        Some(n) if n < vectors.len() => {
            let mut rng = match params.seed {
                Some(s) => StdRng::seed_from_u64(s),
                None => StdRng::from_entropy(),
            };
            let mut indices: Vec<usize> = (0..vectors.len()).collect();
            indices.shuffle(&mut rng);
            indices.truncate(n);
            indices.iter().map(|&i| &vectors[i]).collect()
        }
        _ => vectors.iter().collect(),
    };

    let min_samples = params.codebook_size;
    if training_vectors.len() < min_samples {
        return Err(PqError::InsufficientSamples {
            min: min_samples,
            got: training_vectors.len(),
        });
    }

    let mut codebook = PqCodebook::new(dim, params)?;

    // Train each subvector's codebook independently
    for m in 0..params.num_subvectors {
        train_subvector_codebook(
            &mut codebook,
            m,
            &training_vectors,
            params,
        )?;
    }

    codebook.trained = true;
    Ok(codebook)
}

/// Train PQ codebook with parallel subvector training.
pub fn train_pq_parallel(vectors: &[Vec<f32>], params: &PqParams) -> PqResult<PqCodebook> {
    if vectors.is_empty() {
        return Err(PqError::InsufficientSamples { min: 1, got: 0 });
    }

    let dim = vectors[0].len();
    params.validate(dim).map_err(PqError::InvalidParams)?;

    // Sample training vectors if specified
    let training_vectors: Vec<&Vec<f32>> = match params.training_samples {
        Some(n) if n < vectors.len() => {
            let mut rng = match params.seed {
                Some(s) => StdRng::seed_from_u64(s),
                None => StdRng::from_entropy(),
            };
            let mut indices: Vec<usize> = (0..vectors.len()).collect();
            indices.shuffle(&mut rng);
            indices.truncate(n);
            indices.iter().map(|&i| &vectors[i]).collect()
        }
        _ => vectors.iter().collect(),
    };

    let min_samples = params.codebook_size;
    if training_vectors.len() < min_samples {
        return Err(PqError::InsufficientSamples {
            min: min_samples,
            got: training_vectors.len(),
        });
    }

    let subvector_dim = dim / params.num_subvectors;

    // Extract all subvectors for parallel processing
    let subvector_sets: Vec<Vec<Vec<f32>>> = (0..params.num_subvectors)
        .map(|m| {
            let start = m * subvector_dim;
            training_vectors
                .iter()
                .map(|v| v[start..start + subvector_dim].to_vec())
                .collect()
        })
        .collect();

    // Train each subvector codebook in parallel
    let codebook_parts: Vec<Vec<f32>> = subvector_sets
        .par_iter()
        .map(|subvecs| {
            train_subvector_kmeans(
                subvecs,
                params.codebook_size,
                subvector_dim,
                params.max_iterations,
                params.convergence_threshold,
                params.seed,
            )
        })
        .collect();

    // Assemble the full codebook
    let mut codebook = PqCodebook::new(dim, params)?;
    for (m, part) in codebook_parts.into_iter().enumerate() {
        let offset = m * params.codebook_size * subvector_dim;
        codebook.centroids[offset..offset + part.len()].copy_from_slice(&part);
    }

    codebook.trained = true;
    Ok(codebook)
}

/// Train the codebook for a single subvector using k-means.
fn train_subvector_codebook(
    codebook: &mut PqCodebook,
    m: usize,
    vectors: &[&Vec<f32>],
    params: &PqParams,
) -> PqResult<()> {
    let subvector_dim = codebook.subvector_dim;
    let start = m * subvector_dim;

    // Extract subvectors
    let subvectors: Vec<Vec<f32>> = vectors
        .iter()
        .map(|v| v[start..start + subvector_dim].to_vec())
        .collect();

    // Run k-means
    let centroids = train_subvector_kmeans(
        &subvectors,
        params.codebook_size,
        subvector_dim,
        params.max_iterations,
        params.convergence_threshold,
        params.seed.map(|s| s + m as u64),
    );

    // Copy centroids to codebook
    for k in 0..params.codebook_size {
        let src_offset = k * subvector_dim;
        let centroid = codebook.get_centroid_mut(m, k);
        centroid.copy_from_slice(&centroids[src_offset..src_offset + subvector_dim]);
    }

    Ok(())
}

/// K-means clustering for subvectors.
fn train_subvector_kmeans(
    subvectors: &[Vec<f32>],
    k: usize,
    dim: usize,
    max_iterations: usize,
    convergence_threshold: f64,
    seed: Option<u64>,
) -> Vec<f32> {
    let n = subvectors.len();
    let k = k.min(n);

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    // K-means++ initialization
    let mut centroids = kmeans_plusplus_init(subvectors, k, dim, &mut rng);
    let mut assignments = vec![0usize; n];

    for _iter in 0..max_iterations {
        // Assignment step
        let mut changes = 0usize;
        for (i, subvec) in subvectors.iter().enumerate() {
            let mut best_k = 0;
            let mut best_dist = f32::MAX;

            for (ki, chunk) in centroids.chunks(dim).enumerate() {
                let dist = l2_distance_squared(subvec, chunk);
                if dist < best_dist {
                    best_dist = dist;
                    best_k = ki;
                }
            }

            if assignments[i] != best_k {
                changes += 1;
                assignments[i] = best_k;
            }
        }

        // Check convergence
        let change_ratio = changes as f64 / n as f64;
        if change_ratio < convergence_threshold {
            break;
        }

        // Update step
        let mut new_centroids = vec![0.0f64; k * dim];
        let mut counts = vec![0usize; k];

        for (i, subvec) in subvectors.iter().enumerate() {
            let ki = assignments[i];
            counts[ki] += 1;
            let offset = ki * dim;
            for (j, &v) in subvec.iter().enumerate() {
                new_centroids[offset + j] += v as f64;
            }
        }

        for ki in 0..k {
            if counts[ki] > 0 {
                let offset = ki * dim;
                for j in 0..dim {
                    centroids[offset + j] = (new_centroids[offset + j] / counts[ki] as f64) as f32;
                }
            }
        }
    }

    centroids
}

/// K-means++ initialization.
fn kmeans_plusplus_init<R: Rng>(
    vectors: &[Vec<f32>],
    k: usize,
    dim: usize,
    rng: &mut R,
) -> Vec<f32> {
    let n = vectors.len();
    let mut centroids = vec![0.0f32; k * dim];

    // Pick first centroid uniformly at random
    let first_idx = rng.gen_range(0..n);
    centroids[..dim].copy_from_slice(&vectors[first_idx]);

    let mut min_distances: Vec<f32> = vectors
        .iter()
        .map(|v| l2_distance_squared(v, &centroids[..dim]))
        .collect();

    // Pick remaining centroids proportional to squared distance
    for ki in 1..k {
        let total_dist: f64 = min_distances.iter().map(|&d| d as f64).sum();

        if total_dist == 0.0 {
            // All remaining points are duplicates
            let idx = rng.gen_range(0..n);
            let offset = ki * dim;
            centroids[offset..offset + dim].copy_from_slice(&vectors[idx]);
            continue;
        }

        // Sample proportional to squared distance
        let threshold = rng.gen::<f64>() * total_dist;
        let mut cumsum = 0.0;
        let mut chosen_idx = 0;

        for (i, &dist) in min_distances.iter().enumerate() {
            cumsum += dist as f64;
            if cumsum >= threshold {
                chosen_idx = i;
                break;
            }
        }

        let offset = ki * dim;
        centroids[offset..offset + dim].copy_from_slice(&vectors[chosen_idx]);

        // Update minimum distances
        for (i, v) in vectors.iter().enumerate() {
            let dist = l2_distance_squared(v, &centroids[offset..offset + dim]);
            if dist < min_distances[i] {
                min_distances[i] = dist;
            }
        }
    }

    centroids
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_clustered_vectors(
        num_clusters: usize,
        points_per_cluster: usize,
        dim: usize,
    ) -> Vec<Vec<f32>> {
        let mut rng = StdRng::seed_from_u64(42);
        let mut vectors = Vec::new();

        for c in 0..num_clusters {
            let center: Vec<f32> = (0..dim).map(|_| rng.gen_range(-10.0..10.0)).collect();

            for _ in 0..points_per_cluster {
                let v: Vec<f32> = center
                    .iter()
                    .map(|&x| x + rng.gen_range(-0.5..0.5))
                    .collect();
                vectors.push(v);
            }
        }

        vectors
    }

    #[test]
    fn test_codebook_new() {
        let params = PqParams::new(8, 256);
        let codebook = PqCodebook::new(128, &params).unwrap();

        assert_eq!(codebook.num_subvectors, 8);
        assert_eq!(codebook.codebook_size, 256);
        assert_eq!(codebook.dim, 128);
        assert_eq!(codebook.subvector_dim, 16);
        assert!(!codebook.trained);
    }

    #[test]
    fn test_train_pq() {
        let vectors = generate_clustered_vectors(10, 100, 32);
        let params = PqParams::new(4, 16).with_max_iterations(10);

        let codebook = train_pq(&vectors, &params).unwrap();

        assert!(codebook.trained);
        assert_eq!(codebook.num_subvectors, 4);
        assert_eq!(codebook.codebook_size, 16);
    }

    #[test]
    fn test_codebook_serialization() {
        let vectors = generate_clustered_vectors(5, 50, 16);
        let params = PqParams::new(4, 8).with_max_iterations(5);

        let codebook = train_pq(&vectors, &params).unwrap();
        let bytes = codebook.to_bytes();
        let restored = PqCodebook::from_bytes(&bytes).unwrap();

        assert_eq!(codebook.num_subvectors, restored.num_subvectors);
        assert_eq!(codebook.codebook_size, restored.codebook_size);
        assert_eq!(codebook.dim, restored.dim);
        assert_eq!(codebook.centroids.len(), restored.centroids.len());

        for (a, b) in codebook.centroids.iter().zip(restored.centroids.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_get_centroid() {
        let params = PqParams::new(4, 8);
        let mut codebook = PqCodebook::new(16, &params).unwrap();

        // Set a centroid
        let centroid = codebook.get_centroid_mut(2, 5);
        for (i, c) in centroid.iter_mut().enumerate() {
            *c = i as f32;
        }

        // Retrieve it
        let retrieved = codebook.get_centroid(2, 5);
        for (i, &c) in retrieved.iter().enumerate() {
            assert_eq!(c, i as f32);
        }
    }
}
