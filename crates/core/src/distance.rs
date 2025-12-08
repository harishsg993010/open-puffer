//! SIMD-optimized distance functions.
//!
//! These functions use manual loop unrolling and compiler auto-vectorization
//! to achieve good SIMD performance on modern CPUs.

/// Compute squared L2 (Euclidean) distance between two vectors.
///
/// This is faster than L2 distance as it avoids the sqrt operation.
/// Use this for comparisons where only relative ordering matters.
#[inline]
pub fn l2_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vector dimensions must match");

    // Process in chunks of 8 for better SIMD utilization
    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    let mut sum2 = 0.0f32;
    let mut sum3 = 0.0f32;

    let a_chunks = a.chunks_exact(8);
    let b_chunks = b.chunks_exact(8);
    let a_remainder = a_chunks.remainder();
    let b_remainder = b_chunks.remainder();

    for (a_chunk, b_chunk) in a_chunks.zip(b_chunks) {
        let d0 = a_chunk[0] - b_chunk[0];
        let d1 = a_chunk[1] - b_chunk[1];
        let d2 = a_chunk[2] - b_chunk[2];
        let d3 = a_chunk[3] - b_chunk[3];
        let d4 = a_chunk[4] - b_chunk[4];
        let d5 = a_chunk[5] - b_chunk[5];
        let d6 = a_chunk[6] - b_chunk[6];
        let d7 = a_chunk[7] - b_chunk[7];

        sum0 += d0 * d0 + d4 * d4;
        sum1 += d1 * d1 + d5 * d5;
        sum2 += d2 * d2 + d6 * d6;
        sum3 += d3 * d3 + d7 * d7;
    }

    // Handle remainder
    for (a_val, b_val) in a_remainder.iter().zip(b_remainder.iter()) {
        let d = a_val - b_val;
        sum0 += d * d;
    }

    sum0 + sum1 + sum2 + sum3
}

/// Compute L2 (Euclidean) distance between two vectors.
#[inline]
pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    l2_distance_squared(a, b).sqrt()
}

/// Compute dot product of two vectors.
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vector dimensions must match");

    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    let mut sum2 = 0.0f32;
    let mut sum3 = 0.0f32;

    let a_chunks = a.chunks_exact(8);
    let b_chunks = b.chunks_exact(8);
    let a_remainder = a_chunks.remainder();
    let b_remainder = b_chunks.remainder();

    for (a_chunk, b_chunk) in a_chunks.zip(b_chunks) {
        sum0 += a_chunk[0] * b_chunk[0] + a_chunk[4] * b_chunk[4];
        sum1 += a_chunk[1] * b_chunk[1] + a_chunk[5] * b_chunk[5];
        sum2 += a_chunk[2] * b_chunk[2] + a_chunk[6] * b_chunk[6];
        sum3 += a_chunk[3] * b_chunk[3] + a_chunk[7] * b_chunk[7];
    }

    for (a_val, b_val) in a_remainder.iter().zip(b_remainder.iter()) {
        sum0 += a_val * b_val;
    }

    sum0 + sum1 + sum2 + sum3
}

/// Compute the L2 norm (magnitude) of a vector.
#[inline]
pub fn l2_norm(v: &[f32]) -> f32 {
    dot_product(v, v).sqrt()
}

/// Compute cosine similarity between two vectors.
///
/// Returns a value in [-1, 1] where 1 means identical direction,
/// 0 means orthogonal, and -1 means opposite direction.
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot = dot_product(a, b);
    let norm_a = l2_norm(a);
    let norm_b = l2_norm(b);

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

/// Compute cosine distance between two vectors.
///
/// Returns a value in [0, 2] where 0 means identical direction.
/// This is defined as 1 - cosine_similarity.
#[inline]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    1.0 - cosine_similarity(a, b)
}

/// Compute cosine distance with precomputed norms.
///
/// This is faster when norms are already known, avoiding 2 sqrt operations.
/// `norm_a` and `norm_b` should be the L2 norms of vectors `a` and `b`.
#[inline]
pub fn cosine_distance_with_norms(a: &[f32], b: &[f32], norm_a: f32, norm_b: f32) -> f32 {
    if norm_a == 0.0 || norm_b == 0.0 {
        return 1.0; // Undefined, return neutral distance
    }
    let dot = dot_product(a, b);
    1.0 - (dot / (norm_a * norm_b))
}

/// Compute cosine distance between a query (with precomputed norm) and a stored vector.
///
/// This is the common case during search: query norm is computed once,
/// and stored vector norms are precomputed.
#[inline]
pub fn cosine_distance_query(query: &[f32], query_norm: f32, stored: &[f32], stored_norm: f32) -> f32 {
    cosine_distance_with_norms(query, stored, query_norm, stored_norm)
}

/// Compute distance between two vectors using the specified metric.
#[inline]
pub fn distance(a: &[f32], b: &[f32], metric: super::Metric) -> f32 {
    match metric {
        super::Metric::L2 => l2_distance_squared(a, b),
        super::Metric::Cosine => cosine_distance(a, b),
    }
}

/// Normalize a vector to unit length (in-place).
pub fn normalize(v: &mut [f32]) {
    let norm = l2_norm(v);
    if norm > 0.0 {
        let inv_norm = 1.0 / norm;
        for x in v.iter_mut() {
            *x *= inv_norm;
        }
    }
}

/// Normalize a vector to unit length (returns new vector).
pub fn normalized(v: &[f32]) -> Vec<f32> {
    let mut result = v.to_vec();
    normalize(&mut result);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-6;

    fn assert_approx_eq(a: f32, b: f32) {
        assert!(
            (a - b).abs() < EPSILON,
            "Values not approximately equal: {} vs {}",
            a,
            b
        );
    }

    #[test]
    fn test_l2_distance_squared() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert_approx_eq(l2_distance_squared(&a, &b), 2.0);

        let c = vec![1.0, 2.0, 3.0];
        let d = vec![4.0, 5.0, 6.0];
        // (4-1)^2 + (5-2)^2 + (6-3)^2 = 9 + 9 + 9 = 27
        assert_approx_eq(l2_distance_squared(&c, &d), 27.0);
    }

    #[test]
    fn test_l2_distance() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![3.0, 4.0, 0.0];
        assert_approx_eq(l2_distance(&a, &b), 5.0);
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert_approx_eq(dot_product(&a, &b), 32.0);
    }

    #[test]
    fn test_cosine_similarity() {
        // Same direction
        let a = vec![1.0, 0.0];
        let b = vec![2.0, 0.0];
        assert_approx_eq(cosine_similarity(&a, &b), 1.0);

        // Opposite direction
        let c = vec![1.0, 0.0];
        let d = vec![-1.0, 0.0];
        assert_approx_eq(cosine_similarity(&c, &d), -1.0);

        // Orthogonal
        let e = vec![1.0, 0.0];
        let f = vec![0.0, 1.0];
        assert_approx_eq(cosine_similarity(&e, &f), 0.0);
    }

    #[test]
    fn test_cosine_distance() {
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0];
        assert_approx_eq(cosine_distance(&a, &b), 0.0);

        let c = vec![1.0, 0.0];
        let d = vec![-1.0, 0.0];
        assert_approx_eq(cosine_distance(&c, &d), 2.0);
    }

    #[test]
    fn test_normalize() {
        let mut v = vec![3.0, 4.0];
        normalize(&mut v);
        assert_approx_eq(v[0], 0.6);
        assert_approx_eq(v[1], 0.8);
        assert_approx_eq(l2_norm(&v), 1.0);
    }

    #[test]
    fn test_large_vectors() {
        // Test with vectors larger than chunk size
        let a: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..1000).map(|i| (i * 2) as f32).collect();

        // Verify it doesn't panic and gives reasonable result
        let dist = l2_distance_squared(&a, &b);
        assert!(dist > 0.0);

        let cos = cosine_similarity(&a, &b);
        assert!(cos > 0.0 && cos <= 1.0);
    }

    #[test]
    fn test_zero_vector() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_approx_eq(cosine_similarity(&a, &b), 0.0);
    }
}
