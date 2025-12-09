//! Async second-pass refinement for improved recall on long-tail queries.
//!
//! This module implements:
//! - Streaming refinement that returns initial results quickly
//! - Background refinement to improve recall by searching more clusters
//! - Configurable refinement depth and timeouts

use crate::error::{QueryError, QueryResult};
use puffer_core::{distance, Metric, VectorId};
use puffer_index::SearchResult;
use parking_lot::Mutex;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

/// Configuration for async refinement.
#[derive(Debug, Clone)]
pub struct RefinementConfig {
    /// Maximum time to wait for initial results (ms).
    pub initial_timeout_ms: u64,

    /// Maximum time for background refinement (ms).
    pub refinement_timeout_ms: u64,

    /// Number of initial clusters to probe (fast path).
    pub initial_nprobe: usize,

    /// Additional clusters to probe in refinement (slow path).
    pub refinement_nprobe: usize,

    /// Whether to continue refining after initial results.
    pub enable_refinement: bool,

    /// Minimum score improvement to include refinement results.
    pub min_score_improvement: f32,

    /// Maximum candidates to consider during refinement.
    pub max_refinement_candidates: usize,
}

impl Default for RefinementConfig {
    fn default() -> Self {
        Self {
            initial_timeout_ms: 50,
            refinement_timeout_ms: 200,
            initial_nprobe: 5,
            refinement_nprobe: 20,
            enable_refinement: true,
            min_score_improvement: 0.01,
            max_refinement_candidates: 1000,
        }
    }
}

/// State for tracking refinement progress.
pub struct RefinementState {
    /// Best results found so far.
    results: Mutex<Vec<SearchResult>>,

    /// IDs already seen (for deduplication).
    seen_ids: Mutex<HashSet<String>>,

    /// Number of results requested.
    k: usize,

    /// Query start time.
    start_time: Instant,

    /// Whether initial results have been returned.
    initial_complete: Mutex<bool>,

    /// Configuration.
    config: RefinementConfig,
}

impl RefinementState {
    /// Create new refinement state.
    pub fn new(k: usize, config: RefinementConfig) -> Self {
        Self {
            results: Mutex::new(Vec::with_capacity(k * 2)),
            seen_ids: Mutex::new(HashSet::new()),
            k,
            start_time: Instant::now(),
            initial_complete: Mutex::new(false),
            config,
        }
    }

    /// Add results from a search pass.
    pub fn add_results(&self, new_results: Vec<SearchResult>) -> Vec<SearchResult> {
        let mut results = self.results.lock();
        let mut seen = self.seen_ids.lock();

        for result in new_results {
            let id_str = result.id.as_str().to_string();
            if !seen.contains(&id_str) {
                seen.insert(id_str);
                results.push(result);
            }
        }

        // Sort by distance
        results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(Ordering::Equal));

        // Return current top-k
        results.iter().take(self.k).cloned().collect()
    }

    /// Get current best results.
    pub fn get_results(&self) -> Vec<SearchResult> {
        let results = self.results.lock();
        results.iter().take(self.k).cloned().collect()
    }

    /// Mark initial phase as complete.
    pub fn mark_initial_complete(&self) {
        let mut complete = self.initial_complete.lock();
        *complete = true;
    }

    /// Check if initial phase is complete.
    pub fn is_initial_complete(&self) -> bool {
        *self.initial_complete.lock()
    }

    /// Check if refinement should continue.
    pub fn should_continue_refinement(&self) -> bool {
        if !self.config.enable_refinement {
            return false;
        }

        let elapsed = self.start_time.elapsed();
        let timeout = Duration::from_millis(
            self.config.initial_timeout_ms + self.config.refinement_timeout_ms,
        );

        elapsed < timeout
    }

    /// Get elapsed time.
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }
}

/// Result from async refinement search.
#[derive(Debug, Clone)]
pub struct RefinedSearchResult {
    /// Search results.
    pub results: Vec<SearchResult>,

    /// Whether refinement is still in progress.
    pub refinement_pending: bool,

    /// Number of clusters probed.
    pub clusters_probed: usize,

    /// Time taken for initial results.
    pub initial_time_ms: u64,

    /// Total time taken.
    pub total_time_ms: u64,
}

/// Async refinement search context.
pub struct AsyncRefiner {
    config: RefinementConfig,
}

impl AsyncRefiner {
    /// Create a new async refiner.
    pub fn new(config: RefinementConfig) -> Self {
        Self { config }
    }

    /// Perform search with async refinement.
    ///
    /// Returns initial results quickly, then optionally refines in background.
    pub async fn search_with_refinement<F, Fut>(
        &self,
        k: usize,
        search_clusters: F,
    ) -> RefinedSearchResult
    where
        F: Fn(Vec<usize>) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Vec<SearchResult>> + Send,
    {
        let start = Instant::now();
        let state = Arc::new(RefinementState::new(k, self.config.clone()));

        // Phase 1: Initial fast search
        let initial_clusters: Vec<usize> = (0..self.config.initial_nprobe).collect();
        let initial_results = search_clusters(initial_clusters.clone()).await;
        let current_results = state.add_results(initial_results);
        let clusters_probed = self.config.initial_nprobe;

        let initial_time = start.elapsed();
        state.mark_initial_complete();

        // Check if we need refinement
        if !self.config.enable_refinement
            || current_results.len() >= k
            || initial_time.as_millis() as u64 >= self.config.refinement_timeout_ms
        {
            return RefinedSearchResult {
                results: current_results,
                refinement_pending: false,
                clusters_probed,
                initial_time_ms: initial_time.as_millis() as u64,
                total_time_ms: start.elapsed().as_millis() as u64,
            };
        }

        // Phase 2: Refinement (search additional clusters)
        let refinement_clusters: Vec<usize> = (self.config.initial_nprobe
            ..self.config.initial_nprobe + self.config.refinement_nprobe)
            .collect();

        let timeout = Duration::from_millis(self.config.refinement_timeout_ms);
        let remaining = timeout.saturating_sub(initial_time);

        let refinement_result = tokio::time::timeout(remaining, async {
            search_clusters(refinement_clusters.clone()).await
        })
        .await;

        let (final_results, total_clusters) = match refinement_result {
            Ok(results) => {
                let refined = state.add_results(results);
                (refined, clusters_probed + refinement_clusters.len())
            }
            Err(_) => {
                // Timeout - return what we have
                (state.get_results(), clusters_probed)
            }
        };

        RefinedSearchResult {
            results: final_results,
            refinement_pending: false,
            clusters_probed: total_clusters,
            initial_time_ms: initial_time.as_millis() as u64,
            total_time_ms: start.elapsed().as_millis() as u64,
        }
    }
}

/// Streaming refinement that yields results as they're found.
pub struct StreamingRefiner {
    config: RefinementConfig,
}

impl StreamingRefiner {
    /// Create a new streaming refiner.
    pub fn new(config: RefinementConfig) -> Self {
        Self { config }
    }

    /// Start streaming search with refinement.
    ///
    /// Returns a channel receiver that yields progressively better results.
    pub fn search_streaming<F, Fut>(
        &self,
        k: usize,
        total_clusters: usize,
        search_cluster: F,
    ) -> mpsc::Receiver<StreamingUpdate>
    where
        F: Fn(usize) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Vec<SearchResult>> + Send + 'static,
    {
        let (tx, rx) = mpsc::channel(32);
        let config = self.config.clone();

        tokio::spawn(async move {
            let start = Instant::now();
            let state = Arc::new(RefinementState::new(k, config.clone()));
            let mut clusters_searched = 0;

            // Search clusters in batches
            let batch_size = config.initial_nprobe;
            let max_clusters = total_clusters.min(config.initial_nprobe + config.refinement_nprobe);

            for batch_start in (0..max_clusters).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(max_clusters);

                // Check timeout
                let elapsed = start.elapsed();
                let timeout = Duration::from_millis(
                    config.initial_timeout_ms + config.refinement_timeout_ms,
                );
                if elapsed >= timeout {
                    break;
                }

                // Search this batch of clusters
                let mut batch_results = Vec::new();
                for cluster_idx in batch_start..batch_end {
                    let results = search_cluster(cluster_idx).await;
                    batch_results.extend(results);
                    clusters_searched += 1;
                }

                // Update state and send update
                let current_results = state.add_results(batch_results);
                let is_initial = batch_start == 0;

                let update = StreamingUpdate {
                    results: current_results,
                    is_initial,
                    is_final: batch_end >= max_clusters,
                    clusters_searched,
                    elapsed_ms: start.elapsed().as_millis() as u64,
                };

                if tx.send(update).await.is_err() {
                    // Receiver dropped
                    break;
                }

                // Mark initial complete after first batch
                if is_initial {
                    state.mark_initial_complete();
                }
            }
        });

        rx
    }
}

/// Update from streaming refinement.
#[derive(Debug, Clone)]
pub struct StreamingUpdate {
    /// Current best results.
    pub results: Vec<SearchResult>,

    /// Whether this is the initial result set.
    pub is_initial: bool,

    /// Whether refinement is complete.
    pub is_final: bool,

    /// Number of clusters searched so far.
    pub clusters_searched: usize,

    /// Time elapsed since search start.
    pub elapsed_ms: u64,
}

/// Exact re-ranking of approximate results.
pub struct ExactReranker;

impl ExactReranker {
    /// Re-rank approximate results using exact distance computation.
    pub fn rerank(
        approximate_results: &[SearchResult],
        query: &[f32],
        get_vector: impl Fn(&VectorId) -> Option<Vec<f32>>,
        metric: Metric,
        k: usize,
    ) -> Vec<SearchResult> {
        let mut reranked: Vec<SearchResult> = approximate_results
            .iter()
            .filter_map(|result| {
                get_vector(&result.id).map(|vector| {
                    let exact_distance = distance::distance(query, &vector, metric);
                    SearchResult {
                        id: result.id.clone(),
                        distance: exact_distance,
                        payload: result.payload.clone(),
                    }
                })
            })
            .collect();

        reranked.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(Ordering::Equal));
        reranked.truncate(k);
        reranked
    }

    /// Re-rank with candidate expansion.
    ///
    /// Fetches more candidates than k, re-ranks, then returns top-k.
    pub fn rerank_with_expansion(
        approximate_results: &[SearchResult],
        query: &[f32],
        get_vector: impl Fn(&VectorId) -> Option<Vec<f32>>,
        metric: Metric,
        k: usize,
        expansion_factor: usize,
    ) -> Vec<SearchResult> {
        let candidates = approximate_results
            .iter()
            .take(k * expansion_factor)
            .cloned()
            .collect::<Vec<_>>();

        Self::rerank(&candidates, query, get_vector, metric, k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use puffer_core::VectorId;

    fn make_result(id: &str, distance: f32) -> SearchResult {
        SearchResult {
            id: VectorId::new(id.to_string()),
            distance,
            payload: None,
        }
    }

    #[test]
    fn test_refinement_state() {
        let config = RefinementConfig::default();
        let state = RefinementState::new(3, config);

        // Add some results
        let results1 = vec![
            make_result("a", 0.5),
            make_result("b", 0.3),
            make_result("c", 0.7),
        ];
        let top = state.add_results(results1);

        assert_eq!(top.len(), 3);
        assert_eq!(top[0].id.as_str(), "b"); // Lowest distance first

        // Add more results (with duplicates)
        let results2 = vec![
            make_result("b", 0.3), // Duplicate
            make_result("d", 0.1), // New best
            make_result("e", 0.6),
        ];
        let top = state.add_results(results2);

        assert_eq!(top.len(), 3);
        assert_eq!(top[0].id.as_str(), "d"); // New best
        assert_eq!(top[1].id.as_str(), "b");
        assert_eq!(top[2].id.as_str(), "a");
    }

    #[test]
    fn test_exact_reranker() {
        let approximate = vec![
            make_result("a", 1.0), // Approximate distance
            make_result("b", 2.0),
            make_result("c", 3.0),
        ];

        let query = vec![0.0, 0.0];

        // Mock vectors with different actual distances
        let get_vector = |id: &VectorId| -> Option<Vec<f32>> {
            match id.as_str() {
                "a" => Some(vec![2.0, 0.0]), // Actual dist = 2.0
                "b" => Some(vec![0.5, 0.0]), // Actual dist = 0.5
                "c" => Some(vec![1.0, 0.0]), // Actual dist = 1.0
                _ => None,
            }
        };

        let reranked = ExactReranker::rerank(&approximate, &query, get_vector, Metric::L2, 3);

        assert_eq!(reranked.len(), 3);
        assert_eq!(reranked[0].id.as_str(), "b"); // Now first (closest)
        assert_eq!(reranked[1].id.as_str(), "c");
        assert_eq!(reranked[2].id.as_str(), "a");
    }

    #[tokio::test]
    async fn test_async_refiner() {
        let config = RefinementConfig {
            initial_nprobe: 2,
            refinement_nprobe: 3,
            enable_refinement: true,
            ..Default::default()
        };

        let refiner = AsyncRefiner::new(config);

        // Mock search function
        let search_fn = |clusters: Vec<usize>| async move {
            clusters
                .into_iter()
                .map(|i| make_result(&format!("cluster_{}", i), i as f32 * 0.1))
                .collect()
        };

        let result = refiner.search_with_refinement(3, search_fn).await;

        assert!(!result.results.is_empty());
        assert!(!result.refinement_pending);
        assert!(result.clusters_probed >= 2);
    }
}
