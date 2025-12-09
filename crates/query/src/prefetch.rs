//! Multi-segment prefetching and mmap warming.
//!
//! This module implements:
//! - Predictive segment prefetching based on query patterns
//! - Mmap page warming to reduce first-access latency
//! - Background prefetch workers

use puffer_core::Metric;
use puffer_storage::LoadedSegment;
use parking_lot::{Mutex, RwLock};
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

/// Configuration for prefetching.
#[derive(Debug, Clone)]
pub struct PrefetchConfig {
    /// Number of segments to prefetch ahead.
    pub prefetch_ahead: usize,

    /// Maximum segments to keep in warm cache.
    pub max_warm_segments: usize,

    /// Whether to warm mmap pages on load.
    pub warm_on_load: bool,

    /// Page size for warming (bytes).
    pub page_size: usize,

    /// Maximum pages to warm per segment.
    pub max_pages_to_warm: usize,

    /// Prefetch queue size.
    pub queue_size: usize,

    /// Background worker count.
    pub num_workers: usize,

    /// Idle timeout before evicting warm segments.
    pub idle_timeout: Duration,
}

impl Default for PrefetchConfig {
    fn default() -> Self {
        Self {
            prefetch_ahead: 3,
            max_warm_segments: 10,
            warm_on_load: true,
            page_size: 4096,
            max_pages_to_warm: 1000,
            queue_size: 32,
            num_workers: 2,
            idle_timeout: Duration::from_secs(60),
        }
    }
}

/// Statistics for prefetching.
#[derive(Debug, Clone, Default)]
pub struct PrefetchStats {
    pub segments_prefetched: usize,
    pub segments_warmed: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub pages_warmed: usize,
    pub prefetch_time_ms: u64,
}

/// Entry in the warm segment cache.
struct WarmSegmentEntry {
    segment: Arc<LoadedSegment>,
    last_access: Instant,
    access_count: usize,
}

/// Warm segment cache with LRU eviction.
pub struct WarmSegmentCache {
    /// Cached segments by path.
    cache: RwLock<HashMap<PathBuf, WarmSegmentEntry>>,

    /// Configuration.
    config: PrefetchConfig,

    /// Statistics.
    stats: Mutex<PrefetchStats>,
}

impl WarmSegmentCache {
    /// Create a new warm segment cache.
    pub fn new(config: PrefetchConfig) -> Self {
        Self {
            cache: RwLock::new(HashMap::new()),
            config,
            stats: Mutex::new(PrefetchStats::default()),
        }
    }

    /// Get a segment from cache or load it.
    pub fn get_or_load(&self, path: &Path) -> Option<Arc<LoadedSegment>> {
        let path_buf = path.to_path_buf();

        // Check cache first
        {
            let mut cache = self.cache.write();
            if let Some(entry) = cache.get_mut(&path_buf) {
                entry.last_access = Instant::now();
                entry.access_count += 1;

                let mut stats = self.stats.lock();
                stats.cache_hits += 1;

                return Some(entry.segment.clone());
            }
        }

        // Cache miss - load segment
        let mut stats = self.stats.lock();
        stats.cache_misses += 1;
        drop(stats);

        self.load_and_cache(path)
    }

    /// Load a segment and add to cache.
    fn load_and_cache(&self, path: &Path) -> Option<Arc<LoadedSegment>> {
        let start = Instant::now();

        let segment = LoadedSegment::open(path).ok()?;
        let segment = Arc::new(segment);

        // Warm mmap pages if configured
        if self.config.warm_on_load {
            let pages_warmed = self.warm_segment(&segment);
            let mut stats = self.stats.lock();
            stats.pages_warmed += pages_warmed;
            stats.segments_warmed += 1;
        }

        // Add to cache
        let path_buf = path.to_path_buf();
        {
            let mut cache = self.cache.write();

            // Evict if at capacity
            if cache.len() >= self.config.max_warm_segments {
                self.evict_lru(&mut cache);
            }

            cache.insert(
                path_buf,
                WarmSegmentEntry {
                    segment: segment.clone(),
                    last_access: Instant::now(),
                    access_count: 1,
                },
            );
        }

        let mut stats = self.stats.lock();
        stats.segments_prefetched += 1;
        stats.prefetch_time_ms += start.elapsed().as_millis() as u64;

        Some(segment)
    }

    /// Warm mmap pages for a segment.
    fn warm_segment(&self, segment: &LoadedSegment) -> usize {
        // Access vector data to trigger page faults and load into memory
        let num_vectors = segment.num_vectors();
        let dim = segment.dimension();

        // Calculate approximate bytes and pages
        let vector_bytes = num_vectors * dim * std::mem::size_of::<f32>();
        let approx_pages = vector_bytes / self.config.page_size;
        let pages_to_warm = approx_pages.min(self.config.max_pages_to_warm);

        if pages_to_warm == 0 || num_vectors == 0 {
            return 0;
        }

        // Sample vectors across the segment to warm pages
        let step = num_vectors / pages_to_warm.max(1);
        let mut warmed = 0;

        for i in (0..num_vectors).step_by(step.max(1)) {
            // Access the vector data (triggers page fault if not loaded)
            let _vector = segment.get_vector(i);
            warmed += 1;

            if warmed >= pages_to_warm {
                break;
            }
        }

        warmed
    }

    /// Evict least recently used entry.
    fn evict_lru(&self, cache: &mut HashMap<PathBuf, WarmSegmentEntry>) {
        if let Some((oldest_path, _)) = cache
            .iter()
            .min_by_key(|(_, entry)| entry.last_access)
            .map(|(k, v)| (k.clone(), v.last_access))
        {
            cache.remove(&oldest_path);
        }
    }

    /// Prefetch a segment in the background.
    pub fn prefetch(&self, path: &Path) {
        let path_buf = path.to_path_buf();

        // Skip if already cached
        {
            let cache = self.cache.read();
            if cache.contains_key(&path_buf) {
                return;
            }
        }

        // Load and cache
        let _ = self.load_and_cache(path);
    }

    /// Evict stale entries.
    pub fn evict_stale(&self) {
        let now = Instant::now();
        let mut cache = self.cache.write();

        cache.retain(|_, entry| now.duration_since(entry.last_access) < self.config.idle_timeout);
    }

    /// Get cache statistics.
    pub fn stats(&self) -> PrefetchStats {
        self.stats.lock().clone()
    }

    /// Clear the cache.
    pub fn clear(&self) {
        let mut cache = self.cache.write();
        cache.clear();
    }
}

/// Prefetch request.
#[derive(Debug, Clone)]
pub struct PrefetchRequest {
    /// Segment paths to prefetch.
    pub paths: Vec<PathBuf>,

    /// Priority (higher = more urgent).
    pub priority: u32,
}

/// Background prefetch worker.
pub struct PrefetchWorker {
    /// Warm segment cache.
    cache: Arc<WarmSegmentCache>,

    /// Request receiver.
    rx: mpsc::Receiver<PrefetchRequest>,

    /// Shutdown flag.
    shutdown: Arc<AtomicBool>,

    /// Active request count.
    active_count: Arc<AtomicUsize>,
}

impl PrefetchWorker {
    /// Run the worker.
    pub async fn run(mut self) {
        while !self.shutdown.load(Ordering::Relaxed) {
            match self.rx.recv().await {
                Some(request) => {
                    self.active_count.fetch_add(1, Ordering::Relaxed);

                    for path in &request.paths {
                        if self.shutdown.load(Ordering::Relaxed) {
                            break;
                        }
                        self.cache.prefetch(path);
                    }

                    self.active_count.fetch_sub(1, Ordering::Relaxed);
                }
                None => break,
            }
        }
    }
}

/// Prefetch manager that coordinates background prefetching.
pub struct PrefetchManager {
    /// Warm segment cache.
    cache: Arc<WarmSegmentCache>,

    /// Request sender.
    tx: mpsc::Sender<PrefetchRequest>,

    /// Shutdown flag.
    shutdown: Arc<AtomicBool>,

    /// Active worker count.
    active_count: Arc<AtomicUsize>,

    /// Configuration.
    config: PrefetchConfig,

    /// Query history for prediction.
    query_history: Mutex<VecDeque<QueryRecord>>,

    /// Segment access patterns.
    access_patterns: RwLock<HashMap<String, SegmentAccessPattern>>,
}

/// Record of a query for pattern learning.
#[derive(Debug, Clone)]
struct QueryRecord {
    /// Segments accessed.
    segments: Vec<String>,

    /// Timestamp.
    timestamp: Instant,
}

/// Access pattern for a segment.
#[derive(Debug, Clone, Default)]
struct SegmentAccessPattern {
    /// Co-accessed segments (segment_id -> count).
    co_accessed: HashMap<String, usize>,

    /// Total access count.
    access_count: usize,
}

impl PrefetchManager {
    /// Create a new prefetch manager.
    pub fn new(config: PrefetchConfig) -> Self {
        let cache = Arc::new(WarmSegmentCache::new(config.clone()));
        let (tx, _rx) = mpsc::channel(config.queue_size);

        Self {
            cache,
            tx,
            shutdown: Arc::new(AtomicBool::new(false)),
            active_count: Arc::new(AtomicUsize::new(0)),
            config,
            query_history: Mutex::new(VecDeque::with_capacity(100)),
            access_patterns: RwLock::new(HashMap::new()),
        }
    }

    /// Start background workers.
    pub fn start_workers(&self) -> Vec<tokio::task::JoinHandle<()>> {
        let mut handles = Vec::new();
        let (tx, mut rx) = mpsc::channel::<PrefetchRequest>(self.config.queue_size);

        for _ in 0..self.config.num_workers {
            let cache = self.cache.clone();
            let shutdown = self.shutdown.clone();
            let active_count = self.active_count.clone();

            // Create a new receiver for each worker by cloning from a shared channel
            let (worker_tx, worker_rx) = mpsc::channel(self.config.queue_size);

            let worker = PrefetchWorker {
                cache,
                rx: worker_rx,
                shutdown,
                active_count,
            };

            handles.push(tokio::spawn(worker.run()));
        }

        handles
    }

    /// Request prefetch for segments.
    pub async fn prefetch(&self, paths: Vec<PathBuf>, priority: u32) {
        let request = PrefetchRequest { paths, priority };
        let _ = self.tx.send(request).await;
    }

    /// Get a segment (from cache or load).
    pub fn get_segment(&self, path: &Path) -> Option<Arc<LoadedSegment>> {
        self.cache.get_or_load(path)
    }

    /// Record segment access for pattern learning.
    pub fn record_access(&self, segments: Vec<String>) {
        let record = QueryRecord {
            segments: segments.clone(),
            timestamp: Instant::now(),
        };

        // Update query history
        {
            let mut history = self.query_history.lock();
            history.push_back(record);
            if history.len() > 100 {
                history.pop_front();
            }
        }

        // Update access patterns
        {
            let mut patterns = self.access_patterns.write();
            for (i, seg_id) in segments.iter().enumerate() {
                let pattern = patterns.entry(seg_id.clone()).or_default();
                pattern.access_count += 1;

                // Record co-accessed segments
                for (j, other_id) in segments.iter().enumerate() {
                    if i != j {
                        *pattern.co_accessed.entry(other_id.clone()).or_insert(0) += 1;
                    }
                }
            }
        }
    }

    /// Predict which segments to prefetch based on accessed segment.
    pub fn predict_prefetch(&self, accessed_segment: &str) -> Vec<String> {
        let patterns = self.access_patterns.read();

        if let Some(pattern) = patterns.get(accessed_segment) {
            // Get segments frequently co-accessed
            let mut co_accessed: Vec<_> = pattern.co_accessed.iter().collect();
            co_accessed.sort_by(|a, b| b.1.cmp(a.1));

            co_accessed
                .into_iter()
                .take(self.config.prefetch_ahead)
                .map(|(id, _)| id.clone())
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Get cache statistics.
    pub fn stats(&self) -> PrefetchStats {
        self.cache.stats()
    }

    /// Shutdown the manager.
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::Relaxed);
    }

    /// Evict stale cache entries.
    pub fn evict_stale(&self) {
        self.cache.evict_stale();
    }
}

/// Page advisor for mmap warming hints.
pub struct MmapAdvisor {
    /// Segments being advised.
    advised: Mutex<HashSet<PathBuf>>,
}

impl MmapAdvisor {
    /// Create a new mmap advisor.
    pub fn new() -> Self {
        Self {
            advised: Mutex::new(HashSet::new()),
        }
    }

    /// Advise the kernel about expected access patterns.
    #[cfg(unix)]
    pub fn advise_sequential(&self, path: &Path) {
        use std::os::unix::io::AsRawFd;

        let mut advised = self.advised.lock();
        if advised.contains(path) {
            return;
        }

        if let Ok(file) = std::fs::File::open(path) {
            let fd = file.as_raw_fd();
            let len = file.metadata().map(|m| m.len()).unwrap_or(0);

            unsafe {
                // POSIX_FADV_SEQUENTIAL = 2
                libc::posix_fadvise(fd, 0, len as i64, 2);
            }

            advised.insert(path.to_path_buf());
        }
    }

    /// Advise the kernel about random access patterns.
    #[cfg(unix)]
    pub fn advise_random(&self, path: &Path) {
        use std::os::unix::io::AsRawFd;

        if let Ok(file) = std::fs::File::open(path) {
            let fd = file.as_raw_fd();
            let len = file.metadata().map(|m| m.len()).unwrap_or(0);

            unsafe {
                // POSIX_FADV_RANDOM = 1
                libc::posix_fadvise(fd, 0, len as i64, 1);
            }
        }
    }

    /// Advise that data will be needed soon.
    #[cfg(unix)]
    pub fn advise_willneed(&self, path: &Path) {
        use std::os::unix::io::AsRawFd;

        if let Ok(file) = std::fs::File::open(path) {
            let fd = file.as_raw_fd();
            let len = file.metadata().map(|m| m.len()).unwrap_or(0);

            unsafe {
                // POSIX_FADV_WILLNEED = 3
                libc::posix_fadvise(fd, 0, len as i64, 3);
            }
        }
    }

    /// No-op implementations for non-Unix platforms.
    #[cfg(not(unix))]
    pub fn advise_sequential(&self, _path: &Path) {}

    #[cfg(not(unix))]
    pub fn advise_random(&self, _path: &Path) {}

    #[cfg(not(unix))]
    pub fn advise_willneed(&self, _path: &Path) {}
}

impl Default for MmapAdvisor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prefetch_config() {
        let config = PrefetchConfig::default();
        assert_eq!(config.prefetch_ahead, 3);
        assert_eq!(config.max_warm_segments, 10);
    }

    #[test]
    fn test_prefetch_manager_patterns() {
        let config = PrefetchConfig::default();
        let manager = PrefetchManager::new(config);

        // Record some access patterns
        manager.record_access(vec!["seg_a".to_string(), "seg_b".to_string()]);
        manager.record_access(vec!["seg_a".to_string(), "seg_b".to_string()]);
        manager.record_access(vec!["seg_a".to_string(), "seg_c".to_string()]);

        // Predict based on seg_a
        let predictions = manager.predict_prefetch("seg_a");

        assert!(!predictions.is_empty());
        // seg_b should be predicted (accessed 2x with seg_a)
        assert!(predictions.contains(&"seg_b".to_string()));
    }

    #[test]
    fn test_mmap_advisor() {
        let advisor = MmapAdvisor::new();
        // Just test that the methods don't panic
        advisor.advise_sequential(Path::new("/nonexistent"));
        advisor.advise_random(Path::new("/nonexistent"));
        advisor.advise_willneed(Path::new("/nonexistent"));
    }
}
