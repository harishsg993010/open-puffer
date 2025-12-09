//! PQ vector encoding and decoding.

use crate::codebook::PqCodebook;
use crate::error::{PqError, PqResult};
use puffer_core::distance::l2_distance_squared;
use rayon::prelude::*;

/// Encode a single vector into PQ codes.
///
/// Returns a vector of M codes (u16), one per subvector.
#[inline]
pub fn encode_vector(codebook: &PqCodebook, vector: &[f32]) -> PqResult<Vec<u16>> {
    if !codebook.trained {
        return Err(PqError::CodebookNotTrained);
    }

    if vector.len() != codebook.dim {
        return Err(PqError::DimensionMismatch {
            expected: codebook.dim,
            got: vector.len(),
        });
    }

    let mut codes = Vec::with_capacity(codebook.num_subvectors);

    for m in 0..codebook.num_subvectors {
        let subvec = codebook.extract_subvector(vector, m);
        let code = find_nearest_centroid(codebook, m, subvec);
        codes.push(code);
    }

    Ok(codes)
}

/// Encode multiple vectors in parallel.
pub fn encode_vectors(codebook: &PqCodebook, vectors: &[Vec<f32>]) -> PqResult<Vec<Vec<u16>>> {
    if !codebook.trained {
        return Err(PqError::CodebookNotTrained);
    }

    // Validate dimensions
    for (i, v) in vectors.iter().enumerate() {
        if v.len() != codebook.dim {
            return Err(PqError::DimensionMismatch {
                expected: codebook.dim,
                got: v.len(),
            });
        }
    }

    let codes: Vec<Vec<u16>> = vectors
        .par_iter()
        .map(|v| {
            let mut codes = Vec::with_capacity(codebook.num_subvectors);
            for m in 0..codebook.num_subvectors {
                let subvec = codebook.extract_subvector(v, m);
                let code = find_nearest_centroid(codebook, m, subvec);
                codes.push(code);
            }
            codes
        })
        .collect();

    Ok(codes)
}

/// Encode a single vector into a compact byte array.
///
/// For codebook_size <= 256, uses u8 per code.
/// For larger codebook_size, uses u16 per code.
#[inline]
pub fn encode_vector_compact(codebook: &PqCodebook, vector: &[f32]) -> PqResult<Vec<u8>> {
    let codes = encode_vector(codebook, vector)?;

    if codebook.codebook_size <= 256 {
        Ok(codes.iter().map(|&c| c as u8).collect())
    } else {
        let mut bytes = Vec::with_capacity(codes.len() * 2);
        for code in codes {
            bytes.extend_from_slice(&code.to_le_bytes());
        }
        Ok(bytes)
    }
}

/// Encode multiple vectors into compact byte arrays.
pub fn encode_vectors_compact(codebook: &PqCodebook, vectors: &[Vec<f32>]) -> PqResult<Vec<Vec<u8>>> {
    let codes = encode_vectors(codebook, vectors)?;

    if codebook.codebook_size <= 256 {
        Ok(codes
            .into_iter()
            .map(|c| c.iter().map(|&x| x as u8).collect())
            .collect())
    } else {
        Ok(codes
            .into_iter()
            .map(|c| {
                let mut bytes = Vec::with_capacity(c.len() * 2);
                for code in c {
                    bytes.extend_from_slice(&code.to_le_bytes());
                }
                bytes
            })
            .collect())
    }
}

/// Decode compact bytes back to codes.
#[inline]
pub fn decode_compact_codes(codebook: &PqCodebook, bytes: &[u8]) -> Vec<u16> {
    if codebook.codebook_size <= 256 {
        bytes.iter().map(|&b| b as u16).collect()
    } else {
        bytes
            .chunks_exact(2)
            .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
            .collect()
    }
}

/// Decode PQ codes back to an approximate vector.
///
/// Reconstructs the vector by concatenating the corresponding centroids.
pub fn decode_vector(codebook: &PqCodebook, codes: &[u16]) -> PqResult<Vec<f32>> {
    if codes.len() != codebook.num_subvectors {
        return Err(PqError::InvalidParams(format!(
            "Expected {} codes, got {}",
            codebook.num_subvectors,
            codes.len()
        )));
    }

    let mut vector = Vec::with_capacity(codebook.dim);

    for (m, &code) in codes.iter().enumerate() {
        if code as usize >= codebook.codebook_size {
            return Err(PqError::InvalidCode(code as usize));
        }
        let centroid = codebook.get_centroid(m, code as usize);
        vector.extend_from_slice(centroid);
    }

    Ok(vector)
}

/// Decode multiple code sets in parallel.
pub fn decode_vectors(codebook: &PqCodebook, codes: &[Vec<u16>]) -> PqResult<Vec<Vec<f32>>> {
    codes
        .par_iter()
        .map(|c| decode_vector(codebook, c))
        .collect()
}

/// Find the nearest centroid for a subvector.
#[inline]
fn find_nearest_centroid(codebook: &PqCodebook, m: usize, subvec: &[f32]) -> u16 {
    let mut best_k = 0u16;
    let mut best_dist = f32::MAX;

    for k in 0..codebook.codebook_size {
        let centroid = codebook.get_centroid(m, k);
        let dist = l2_distance_squared(subvec, centroid);
        if dist < best_dist {
            best_dist = dist;
            best_k = k as u16;
        }
    }

    best_k
}

/// Batch encode with progress callback.
pub fn encode_vectors_with_progress<F>(
    codebook: &PqCodebook,
    vectors: &[Vec<f32>],
    progress_callback: F,
) -> PqResult<Vec<Vec<u16>>>
where
    F: Fn(usize, usize) + Send + Sync,
{
    if !codebook.trained {
        return Err(PqError::CodebookNotTrained);
    }

    let total = vectors.len();
    let batch_size = 1000;
    let mut all_codes = Vec::with_capacity(total);

    for (batch_idx, batch) in vectors.chunks(batch_size).enumerate() {
        let codes: Vec<Vec<u16>> = batch
            .par_iter()
            .map(|v| {
                let mut codes = Vec::with_capacity(codebook.num_subvectors);
                for m in 0..codebook.num_subvectors {
                    let subvec = codebook.extract_subvector(v, m);
                    let code = find_nearest_centroid(codebook, m, subvec);
                    codes.push(code);
                }
                codes
            })
            .collect();

        all_codes.extend(codes);
        progress_callback((batch_idx + 1) * batch_size.min(total - batch_idx * batch_size), total);
    }

    Ok(all_codes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codebook::train_pq;
    use crate::config::PqParams;

    fn generate_random_vectors(n: usize, dim: usize) -> Vec<Vec<f32>> {
        use rand::prelude::*;
        let mut rng = StdRng::seed_from_u64(42);
        (0..n)
            .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect()
    }

    #[test]
    fn test_encode_decode() {
        let vectors = generate_random_vectors(500, 32);
        let params = PqParams::new(4, 16).with_max_iterations(10);
        let codebook = train_pq(&vectors, &params).unwrap();

        let test_vec = &vectors[0];
        let codes = encode_vector(&codebook, test_vec).unwrap();
        let decoded = decode_vector(&codebook, &codes).unwrap();

        assert_eq!(codes.len(), 4);
        assert_eq!(decoded.len(), 32);

        // Check reconstruction is reasonable
        let error: f32 = test_vec
            .iter()
            .zip(decoded.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        assert!(error < 10.0, "Reconstruction error too high: {}", error);
    }

    #[test]
    fn test_encode_vectors_parallel() {
        let vectors = generate_random_vectors(1000, 64);
        let params = PqParams::new(8, 32).with_max_iterations(10);
        let codebook = train_pq(&vectors, &params).unwrap();

        let codes = encode_vectors(&codebook, &vectors).unwrap();

        assert_eq!(codes.len(), 1000);
        for code in &codes {
            assert_eq!(code.len(), 8);
            for &c in code {
                assert!(c < 32);
            }
        }
    }

    #[test]
    fn test_compact_encoding() {
        let vectors = generate_random_vectors(300, 32);  // Need >= 256 for K=256
        let params = PqParams::new(4, 256).with_max_iterations(10);
        let codebook = train_pq(&vectors, &params).unwrap();

        let codes = encode_vector(&codebook, &vectors[0]).unwrap();
        let compact = encode_vector_compact(&codebook, &vectors[0]).unwrap();

        // With codebook_size=256, should use u8 (4 bytes total)
        assert_eq!(compact.len(), 4);

        let decoded_codes = decode_compact_codes(&codebook, &compact);
        assert_eq!(codes, decoded_codes);
    }

    #[test]
    fn test_large_codebook_compact() {
        let vectors = generate_random_vectors(1000, 32);
        let params = PqParams::new(4, 512).with_max_iterations(10);
        let codebook = train_pq(&vectors, &params).unwrap();

        let compact = encode_vector_compact(&codebook, &vectors[0]).unwrap();

        // With codebook_size=512 > 256, should use u16 (8 bytes total)
        assert_eq!(compact.len(), 8);
    }

    #[test]
    fn test_reconstruction_error() {
        let vectors = generate_random_vectors(500, 64);
        let params = PqParams::new(8, 64).with_max_iterations(20);
        let codebook = train_pq(&vectors, &params).unwrap();

        let codes = encode_vectors(&codebook, &vectors).unwrap();
        let error = codebook.reconstruction_error(&vectors, &codes);

        // Error should be positive but reasonable
        assert!(error > 0.0);
        assert!(error < 50.0, "Reconstruction error too high: {}", error);
    }
}
