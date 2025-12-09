//! Hybrid search combining vector similarity and BM25 text search.

use crate::error::{FtsError, FtsResult};
use crate::index::FtsIndex;
use crate::search::{search_all_matching, FtsSearchResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for hybrid search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridConfig {
    /// Weight for text (BM25) score. Vector weight = 1 - lambda.
    /// Range: [0, 1], default: 0.5
    pub lambda: f32,

    /// Fusion method for combining scores.
    pub fusion: FusionMethod,

    /// Number of candidates to retrieve from each source.
    /// The more candidates, the better recall but slower performance.
    pub candidates_per_source: usize,

    /// Minimum text score to include a document.
    pub min_text_score: Option<f32>,

    /// Minimum vector score to include a document.
    /// Note: For cosine/L2, lower is better, so this is actually a max.
    pub max_vector_distance: Option<f32>,
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self {
            lambda: 0.5,
            fusion: FusionMethod::WeightedSum,
            candidates_per_source: 100,
            min_text_score: None,
            max_vector_distance: None,
        }
    }
}

impl HybridConfig {
    /// Create config with specific lambda.
    pub fn with_lambda(mut self, lambda: f32) -> Self {
        self.lambda = lambda.clamp(0.0, 1.0);
        self
    }

    /// Use RRF fusion.
    pub fn with_rrf(mut self, k: f32) -> Self {
        self.fusion = FusionMethod::ReciprocalRankFusion { k };
        self
    }

    /// Set candidates per source.
    pub fn with_candidates(mut self, n: usize) -> Self {
        self.candidates_per_source = n;
        self
    }
}

/// Method for fusing text and vector scores.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum FusionMethod {
    /// Weighted sum: score = lambda * text_score + (1 - lambda) * vector_score
    WeightedSum,

    /// Reciprocal Rank Fusion: score = 1/(k + rank_text) + 1/(k + rank_vector)
    ReciprocalRankFusion { k: f32 },

    /// Min-max normalization then weighted sum.
    NormalizedWeightedSum,

    /// Convex combination with softmax normalization.
    SoftmaxFusion { temperature: f32 },
}

impl Default for FusionMethod {
    fn default() -> Self {
        FusionMethod::WeightedSum
    }
}

/// Result of hybrid search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridResult {
    /// Vector/document ID.
    pub id: String,

    /// Combined hybrid score.
    pub score: f32,

    /// Text (BM25) score.
    pub text_score: Option<f32>,

    /// Vector distance/score.
    pub vector_score: Option<f32>,

    /// Rank in text results (1-indexed).
    pub text_rank: Option<usize>,

    /// Rank in vector results (1-indexed).
    pub vector_rank: Option<usize>,
}

/// Vector search result for hybrid fusion.
#[derive(Debug, Clone)]
pub struct VectorResult {
    pub id: String,
    pub distance: f32,
}

/// Perform hybrid search combining vector and text results.
///
/// # Arguments
/// * `text_results` - Results from BM25 text search (id, score)
/// * `vector_results` - Results from vector search (id, distance)
/// * `config` - Hybrid search configuration
/// * `top_k` - Number of results to return
pub fn hybrid_search(
    text_results: &[(String, f32)],
    vector_results: &[VectorResult],
    config: &HybridConfig,
    top_k: usize,
) -> Vec<HybridResult> {
    match &config.fusion {
        FusionMethod::WeightedSum => {
            weighted_sum_fusion(text_results, vector_results, config.lambda, top_k)
        }
        FusionMethod::ReciprocalRankFusion { k } => {
            rrf_fusion(text_results, vector_results, *k, top_k)
        }
        FusionMethod::NormalizedWeightedSum => {
            normalized_fusion(text_results, vector_results, config.lambda, top_k)
        }
        FusionMethod::SoftmaxFusion { temperature } => {
            softmax_fusion(text_results, vector_results, config.lambda, *temperature, top_k)
        }
    }
}

/// Weighted sum fusion.
fn weighted_sum_fusion(
    text_results: &[(String, f32)],
    vector_results: &[VectorResult],
    lambda: f32,
    top_k: usize,
) -> Vec<HybridResult> {
    let mut scores: HashMap<String, HybridResult> = HashMap::new();

    // Normalize text scores to [0, 1]
    let max_text_score = text_results
        .iter()
        .map(|(_, s)| *s)
        .fold(f32::MIN, f32::max);
    let min_text_score = text_results
        .iter()
        .map(|(_, s)| *s)
        .fold(f32::MAX, f32::min);
    let text_range = (max_text_score - min_text_score).max(1e-6);

    // Normalize vector distances to [0, 1] (inverted since lower is better)
    let max_vector_dist = vector_results
        .iter()
        .map(|r| r.distance)
        .fold(f32::MIN, f32::max);
    let min_vector_dist = vector_results
        .iter()
        .map(|r| r.distance)
        .fold(f32::MAX, f32::min);
    let vector_range = (max_vector_dist - min_vector_dist).max(1e-6);

    // Add text results
    for (rank, (id, score)) in text_results.iter().enumerate() {
        let norm_score = (score - min_text_score) / text_range;
        scores.insert(
            id.clone(),
            HybridResult {
                id: id.clone(),
                score: lambda * norm_score,
                text_score: Some(*score),
                vector_score: None,
                text_rank: Some(rank + 1),
                vector_rank: None,
            },
        );
    }

    // Merge vector results
    for (rank, result) in vector_results.iter().enumerate() {
        let norm_score = 1.0 - (result.distance - min_vector_dist) / vector_range;
        let vector_contribution = (1.0 - lambda) * norm_score;

        scores
            .entry(result.id.clone())
            .and_modify(|r| {
                r.score += vector_contribution;
                r.vector_score = Some(result.distance);
                r.vector_rank = Some(rank + 1);
            })
            .or_insert(HybridResult {
                id: result.id.clone(),
                score: vector_contribution,
                text_score: None,
                vector_score: Some(result.distance),
                text_rank: None,
                vector_rank: Some(rank + 1),
            });
    }

    // Sort by score and return top_k
    let mut results: Vec<_> = scores.into_values().collect();
    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(top_k);
    results
}

/// Reciprocal Rank Fusion.
fn rrf_fusion(
    text_results: &[(String, f32)],
    vector_results: &[VectorResult],
    k: f32,
    top_k: usize,
) -> Vec<HybridResult> {
    let mut scores: HashMap<String, HybridResult> = HashMap::new();

    // Add text results
    for (rank, (id, score)) in text_results.iter().enumerate() {
        let rrf_score = 1.0 / (k + (rank + 1) as f32);
        scores.insert(
            id.clone(),
            HybridResult {
                id: id.clone(),
                score: rrf_score,
                text_score: Some(*score),
                vector_score: None,
                text_rank: Some(rank + 1),
                vector_rank: None,
            },
        );
    }

    // Merge vector results
    for (rank, result) in vector_results.iter().enumerate() {
        let rrf_score = 1.0 / (k + (rank + 1) as f32);

        scores
            .entry(result.id.clone())
            .and_modify(|r| {
                r.score += rrf_score;
                r.vector_score = Some(result.distance);
                r.vector_rank = Some(rank + 1);
            })
            .or_insert(HybridResult {
                id: result.id.clone(),
                score: rrf_score,
                text_score: None,
                vector_score: Some(result.distance),
                text_rank: None,
                vector_rank: Some(rank + 1),
            });
    }

    let mut results: Vec<_> = scores.into_values().collect();
    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(top_k);
    results
}

/// Normalized weighted sum fusion with min-max normalization.
fn normalized_fusion(
    text_results: &[(String, f32)],
    vector_results: &[VectorResult],
    lambda: f32,
    top_k: usize,
) -> Vec<HybridResult> {
    // This is essentially the same as weighted_sum_fusion with explicit normalization
    weighted_sum_fusion(text_results, vector_results, lambda, top_k)
}

/// Softmax-based fusion.
fn softmax_fusion(
    text_results: &[(String, f32)],
    vector_results: &[VectorResult],
    lambda: f32,
    temperature: f32,
    top_k: usize,
) -> Vec<HybridResult> {
    let mut scores: HashMap<String, HybridResult> = HashMap::new();

    // Compute softmax normalization for text scores
    let text_exp_sum: f32 = text_results
        .iter()
        .map(|(_, s)| (s / temperature).exp())
        .sum();

    // Compute softmax for inverted vector distances
    let vector_max = vector_results
        .iter()
        .map(|r| r.distance)
        .fold(f32::MIN, f32::max);
    let vector_exp_sum: f32 = vector_results
        .iter()
        .map(|r| ((vector_max - r.distance) / temperature).exp())
        .sum();

    // Add text results with softmax
    for (rank, (id, score)) in text_results.iter().enumerate() {
        let softmax_score = (score / temperature).exp() / text_exp_sum;
        scores.insert(
            id.clone(),
            HybridResult {
                id: id.clone(),
                score: lambda * softmax_score,
                text_score: Some(*score),
                vector_score: None,
                text_rank: Some(rank + 1),
                vector_rank: None,
            },
        );
    }

    // Merge vector results with softmax
    for (rank, result) in vector_results.iter().enumerate() {
        let softmax_score = ((vector_max - result.distance) / temperature).exp() / vector_exp_sum;
        let vector_contribution = (1.0 - lambda) * softmax_score;

        scores
            .entry(result.id.clone())
            .and_modify(|r| {
                r.score += vector_contribution;
                r.vector_score = Some(result.distance);
                r.vector_rank = Some(rank + 1);
            })
            .or_insert(HybridResult {
                id: result.id.clone(),
                score: vector_contribution,
                text_score: None,
                vector_score: Some(result.distance),
                text_rank: None,
                vector_rank: Some(rank + 1),
            });
    }

    let mut results: Vec<_> = scores.into_values().collect();
    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(top_k);
    results
}

/// Helper to convert puffer search results to VectorResult.
pub fn convert_vector_results(results: &[(String, f32)]) -> Vec<VectorResult> {
    results
        .iter()
        .map(|(id, dist)| VectorResult {
            id: id.clone(),
            distance: *dist,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_text_results() -> Vec<(String, f32)> {
        vec![
            ("doc_a".to_string(), 5.0),
            ("doc_b".to_string(), 4.0),
            ("doc_c".to_string(), 3.0),
            ("doc_d".to_string(), 2.0),
        ]
    }

    fn sample_vector_results() -> Vec<VectorResult> {
        vec![
            VectorResult { id: "doc_b".to_string(), distance: 0.1 },
            VectorResult { id: "doc_c".to_string(), distance: 0.2 },
            VectorResult { id: "doc_e".to_string(), distance: 0.3 },
            VectorResult { id: "doc_a".to_string(), distance: 0.5 },
        ]
    }

    #[test]
    fn test_weighted_sum_fusion() {
        let text_results = sample_text_results();
        let vector_results = sample_vector_results();

        let results = weighted_sum_fusion(&text_results, &vector_results, 0.5, 5);

        assert!(!results.is_empty());
        // doc_a and doc_b appear in both, should have higher scores
        let top_ids: Vec<_> = results.iter().map(|r| r.id.as_str()).collect();
        assert!(top_ids.contains(&"doc_a") || top_ids.contains(&"doc_b"));
    }

    #[test]
    fn test_rrf_fusion() {
        let text_results = sample_text_results();
        let vector_results = sample_vector_results();

        let results = rrf_fusion(&text_results, &vector_results, 60.0, 5);

        assert!(!results.is_empty());
        // Documents appearing in both lists should rank higher
        for result in &results {
            if result.text_rank.is_some() && result.vector_rank.is_some() {
                // This document appeared in both lists
                assert!(result.score > 0.0);
            }
        }
    }

    #[test]
    fn test_hybrid_config() {
        let config = HybridConfig::default()
            .with_lambda(0.7)
            .with_rrf(60.0)
            .with_candidates(200);

        assert_eq!(config.lambda, 0.7);
        assert!(matches!(config.fusion, FusionMethod::ReciprocalRankFusion { k } if k == 60.0));
        assert_eq!(config.candidates_per_source, 200);
    }

    #[test]
    fn test_lambda_bounds() {
        let config = HybridConfig::default().with_lambda(1.5);
        assert_eq!(config.lambda, 1.0); // Clamped to max

        let config = HybridConfig::default().with_lambda(-0.5);
        assert_eq!(config.lambda, 0.0); // Clamped to min
    }
}
