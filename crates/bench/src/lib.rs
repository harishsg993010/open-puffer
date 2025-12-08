//! Benchmark utilities for Puffer vector database.

use puffer_core::VectorId;
use rand_distr::{Distribution, Normal};

/// Generate a random vector with given dimension.
pub fn random_vector(dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, 1.0).unwrap();
    (0..dim).map(|_| normal.sample(&mut rng) as f32).collect()
}

/// Generate a batch of random vectors.
pub fn random_vectors(count: usize, dim: usize) -> Vec<Vec<f32>> {
    (0..count).map(|_| random_vector(dim)).collect()
}

/// Generate vector IDs.
pub fn generate_ids(count: usize) -> Vec<VectorId> {
    (0..count)
        .map(|i| VectorId::new(format!("vec_{}", i)))
        .collect()
}
