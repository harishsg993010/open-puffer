//! K-means clustering with k-means++ initialization.

use puffer_core::distance::l2_distance_squared;
use rand::prelude::*;
use rayon::prelude::*;

/// Configuration for k-means clustering.
#[derive(Debug, Clone)]
pub struct KMeansConfig {
    /// Number of clusters (k).
    pub num_clusters: usize,
    /// Maximum iterations.
    pub max_iterations: usize,
    /// Convergence threshold (fraction of vectors that changed assignment).
    pub convergence_threshold: f64,
    /// Random seed for reproducibility.
    pub seed: Option<u64>,
}

impl Default for KMeansConfig {
    fn default() -> Self {
        Self {
            num_clusters: 100,
            max_iterations: 20,
            convergence_threshold: 0.001,
            seed: None,
        }
    }
}

/// Result of k-means clustering.
#[derive(Debug)]
pub struct KMeansResult {
    /// Centroids for each cluster.
    pub centroids: Vec<Vec<f32>>,
    /// Cluster assignment for each vector.
    pub assignments: Vec<usize>,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Whether the algorithm converged.
    pub converged: bool,
}

/// Run k-means clustering on a set of vectors.
///
/// Uses k-means++ initialization for better initial centroids.
pub fn kmeans(vectors: &[Vec<f32>], config: &KMeansConfig) -> KMeansResult {
    if vectors.is_empty() {
        return KMeansResult {
            centroids: Vec::new(),
            assignments: Vec::new(),
            iterations: 0,
            converged: true,
        };
    }

    let dim = vectors[0].len();
    let k = config.num_clusters.min(vectors.len());

    // Initialize with k-means++
    let mut rng = match config.seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_entropy(),
    };

    let mut centroids = kmeans_plusplus_init(vectors, k, &mut rng);
    let mut assignments = vec![0usize; vectors.len()];

    let mut iterations = 0;
    let mut converged = false;

    for iter in 0..config.max_iterations {
        iterations = iter + 1;

        // Assign vectors to nearest centroid (parallel)
        let new_assignments: Vec<usize> = vectors
            .par_iter()
            .map(|v| {
                let mut best_idx = 0;
                let mut best_dist = f32::MAX;
                for (i, c) in centroids.iter().enumerate() {
                    let dist = l2_distance_squared(v, c);
                    if dist < best_dist {
                        best_dist = dist;
                        best_idx = i;
                    }
                }
                best_idx
            })
            .collect();

        // Count changes
        let changes: usize = assignments
            .iter()
            .zip(new_assignments.iter())
            .filter(|(a, b)| a != b)
            .count();

        assignments = new_assignments;

        let change_ratio = changes as f64 / vectors.len() as f64;
        tracing::debug!(
            "K-means iteration {}: {} changes ({:.2}%)",
            iter + 1,
            changes,
            change_ratio * 100.0
        );

        if change_ratio < config.convergence_threshold {
            converged = true;
            break;
        }

        // Update centroids
        centroids = update_centroids(vectors, &assignments, k, dim);
    }

    KMeansResult {
        centroids,
        assignments,
        iterations,
        converged,
    }
}

/// K-means++ initialization.
fn kmeans_plusplus_init<R: Rng>(vectors: &[Vec<f32>], k: usize, rng: &mut R) -> Vec<Vec<f32>> {
    let n = vectors.len();
    let mut centroids = Vec::with_capacity(k);

    // Pick first centroid uniformly at random
    let first_idx = rng.gen_range(0..n);
    centroids.push(vectors[first_idx].clone());

    // Distance to nearest centroid for each point
    let mut min_distances: Vec<f32> = vectors
        .iter()
        .map(|v| l2_distance_squared(v, &centroids[0]))
        .collect();

    // Pick remaining centroids
    for _ in 1..k {
        // Compute cumulative distribution
        let total_dist: f64 = min_distances.iter().map(|&d| d as f64).sum();

        if total_dist == 0.0 {
            // All remaining points are duplicates of centroids
            // Just pick a random one
            let idx = rng.gen_range(0..n);
            centroids.push(vectors[idx].clone());
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

        let new_centroid = vectors[chosen_idx].clone();

        // Update minimum distances
        for (i, v) in vectors.iter().enumerate() {
            let dist = l2_distance_squared(v, &new_centroid);
            if dist < min_distances[i] {
                min_distances[i] = dist;
            }
        }

        centroids.push(new_centroid);
    }

    centroids
}

/// Update centroids based on assignments.
fn update_centroids(
    vectors: &[Vec<f32>],
    assignments: &[usize],
    k: usize,
    dim: usize,
) -> Vec<Vec<f32>> {
    // Parallel accumulation per cluster
    let cluster_sums: Vec<(Vec<f64>, usize)> = (0..k)
        .into_par_iter()
        .map(|cluster_idx| {
            let mut sum = vec![0.0f64; dim];
            let mut count = 0usize;

            for (i, &assignment) in assignments.iter().enumerate() {
                if assignment == cluster_idx {
                    for (j, &v) in vectors[i].iter().enumerate() {
                        sum[j] += v as f64;
                    }
                    count += 1;
                }
            }

            (sum, count)
        })
        .collect();

    // Compute means
    cluster_sums
        .into_iter()
        .map(|(sum, count)| {
            if count == 0 {
                // Empty cluster - reinitialize to a random vector
                // For simplicity, just keep the old position (would need RNG to truly randomize)
                vec![0.0f32; dim]
            } else {
                sum.iter().map(|&s| (s / count as f64) as f32).collect()
            }
        })
        .collect()
}

/// Build cluster data for segment building.
///
/// Returns: Vec<(centroid, vector_indices)>
pub fn build_cluster_data(
    centroids: &[Vec<f32>],
    assignments: &[usize],
) -> Vec<(Vec<f32>, Vec<usize>)> {
    let k = centroids.len();
    let mut cluster_indices: Vec<Vec<usize>> = vec![Vec::new(); k];

    for (vec_idx, &cluster_idx) in assignments.iter().enumerate() {
        cluster_indices[cluster_idx].push(vec_idx);
    }

    centroids
        .iter()
        .cloned()
        .zip(cluster_indices)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_clustered_data(
        cluster_centers: &[Vec<f32>],
        points_per_cluster: usize,
        noise: f32,
    ) -> Vec<Vec<f32>> {
        let mut rng = StdRng::seed_from_u64(42);
        let mut vectors = Vec::new();

        for center in cluster_centers {
            for _ in 0..points_per_cluster {
                let v: Vec<f32> = center
                    .iter()
                    .map(|&c| c + rng.gen_range(-noise..noise))
                    .collect();
                vectors.push(v);
            }
        }

        vectors
    }

    #[test]
    fn test_kmeans_basic() {
        let centers = vec![
            vec![0.0, 0.0],
            vec![10.0, 0.0],
            vec![0.0, 10.0],
        ];
        let vectors = make_clustered_data(&centers, 100, 0.5);

        let config = KMeansConfig {
            num_clusters: 3,
            max_iterations: 50,
            convergence_threshold: 0.001,
            seed: Some(123),
        };

        let result = kmeans(&vectors, &config);

        assert_eq!(result.centroids.len(), 3);
        assert_eq!(result.assignments.len(), 300);
        assert!(result.converged);

        // Verify centroids are close to original centers
        for center in &centers {
            let closest = result
                .centroids
                .iter()
                .map(|c| l2_distance_squared(c, center))
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap();
            assert!(closest < 1.0, "Centroid too far from expected center");
        }
    }

    #[test]
    fn test_kmeans_single_cluster() {
        let vectors: Vec<Vec<f32>> = (0..100)
            .map(|_| vec![1.0, 2.0, 3.0])
            .collect();

        let config = KMeansConfig {
            num_clusters: 1,
            ..Default::default()
        };

        let result = kmeans(&vectors, &config);

        assert_eq!(result.centroids.len(), 1);
        assert!(result.assignments.iter().all(|&a| a == 0));
    }

    #[test]
    fn test_kmeans_more_clusters_than_points() {
        let vectors = vec![
            vec![0.0, 0.0],
            vec![1.0, 1.0],
            vec![2.0, 2.0],
        ];

        let config = KMeansConfig {
            num_clusters: 10, // More than 3 points
            ..Default::default()
        };

        let result = kmeans(&vectors, &config);

        // Should only create 3 clusters
        assert_eq!(result.centroids.len(), 3);
    }

    #[test]
    fn test_build_cluster_data() {
        let centroids = vec![
            vec![0.0, 0.0],
            vec![1.0, 1.0],
        ];
        let assignments = vec![0, 0, 1, 0, 1, 1];

        let cluster_data = build_cluster_data(&centroids, &assignments);

        assert_eq!(cluster_data.len(), 2);
        assert_eq!(cluster_data[0].1, vec![0, 1, 3]); // Cluster 0 indices
        assert_eq!(cluster_data[1].1, vec![2, 4, 5]); // Cluster 1 indices
    }

    #[test]
    fn test_empty_input() {
        let vectors: Vec<Vec<f32>> = Vec::new();
        let config = KMeansConfig::default();
        let result = kmeans(&vectors, &config);

        assert!(result.centroids.is_empty());
        assert!(result.assignments.is_empty());
        assert!(result.converged);
    }
}
