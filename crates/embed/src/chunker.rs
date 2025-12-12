//! Text chunking utilities using chunkr.
//!
//! Provides multiple chunking strategies for splitting documents
//! into smaller pieces suitable for embedding.

use chunkr::chunker::base::BaseChunker;
use chunkr::chunker::char::CharacterChunker;
use chunkr::chunker::word::WordChunker;
use serde::{Deserialize, Serialize};

/// Chunking strategy configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChunkingStrategy {
    /// Chunk by number of words.
    Words {
        /// Maximum words per chunk.
        chunk_size: usize,
        /// Overlap in words between chunks.
        overlap: usize,
    },
    /// Chunk by number of characters.
    Characters {
        /// Maximum characters per chunk.
        chunk_size: usize,
        /// Overlap in characters between chunks.
        overlap: usize,
    },
}

impl Default for ChunkingStrategy {
    fn default() -> Self {
        ChunkingStrategy::Words {
            chunk_size: 256,
            overlap: 32,
        }
    }
}

/// Result of chunking a document.
#[derive(Debug, Clone)]
pub struct ChunkResult {
    /// The chunk text.
    pub text: String,
    /// Chunk index (0-based).
    pub index: usize,
}

/// Puffer text chunker wrapping chunkr.
pub struct PufferChunker {
    strategy: ChunkingStrategy,
}

impl PufferChunker {
    /// Create a new chunker with the given strategy.
    pub fn new(strategy: ChunkingStrategy) -> Self {
        Self { strategy }
    }

    /// Create a chunker with default word-based strategy.
    pub fn default_words(chunk_size: usize, overlap: usize) -> Self {
        Self {
            strategy: ChunkingStrategy::Words { chunk_size, overlap },
        }
    }

    /// Create a chunker with character-based strategy.
    pub fn by_characters(chunk_size: usize, overlap: usize) -> Self {
        Self {
            strategy: ChunkingStrategy::Characters { chunk_size, overlap },
        }
    }

    /// Chunk the given text.
    pub fn chunk(&self, text: &str) -> Vec<ChunkResult> {
        if text.is_empty() {
            return Vec::new();
        }

        match &self.strategy {
            ChunkingStrategy::Words { chunk_size, overlap } => {
                self.chunk_by_words(text, *chunk_size, *overlap)
            }
            ChunkingStrategy::Characters { chunk_size, overlap } => {
                self.chunk_by_chars(text, *chunk_size, *overlap)
            }
        }
    }

    /// Chunk text by words using chunkr.
    fn chunk_by_words(&self, text: &str, chunk_size: usize, overlap: usize) -> Vec<ChunkResult> {
        let chunker = WordChunker::new();
        match chunker.chunk_text(text, chunk_size, overlap) {
            Ok(docs) => docs
                .into_iter()
                .enumerate()
                .map(|(idx, doc)| ChunkResult {
                    text: doc.content,
                    index: idx,
                })
                .collect(),
            Err(_) => {
                // Fallback: return original text as single chunk
                vec![ChunkResult {
                    text: text.to_string(),
                    index: 0,
                }]
            }
        }
    }

    /// Chunk text by characters.
    fn chunk_by_chars(&self, text: &str, chunk_size: usize, overlap: usize) -> Vec<ChunkResult> {
        let chunker = CharacterChunker::new();
        match chunker.chunk_text(text, chunk_size, overlap) {
            Ok(docs) => docs
                .into_iter()
                .enumerate()
                .map(|(idx, doc)| ChunkResult {
                    text: doc.content,
                    index: idx,
                })
                .collect(),
            Err(_) => {
                // Fallback: return original text as single chunk
                vec![ChunkResult {
                    text: text.to_string(),
                    index: 0,
                }]
            }
        }
    }

    /// Get the current strategy.
    pub fn strategy(&self) -> &ChunkingStrategy {
        &self.strategy
    }
}

impl Default for PufferChunker {
    fn default() -> Self {
        Self::new(ChunkingStrategy::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_by_words() {
        let text = "This is a test sentence. Another sentence here. And one more for good measure.";
        let chunker = PufferChunker::default_words(5, 1);
        let chunks = chunker.chunk(text);

        assert!(!chunks.is_empty());
        for chunk in &chunks {
            assert!(!chunk.text.is_empty());
        }
    }

    #[test]
    fn test_chunk_by_characters() {
        let text = "This is a test sentence. Another sentence here.";
        let chunker = PufferChunker::by_characters(20, 5);
        let chunks = chunker.chunk(text);

        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_empty_text() {
        let chunker = PufferChunker::default();
        let chunks = chunker.chunk("");
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_short_text() {
        let text = "Short text.";
        let chunker = PufferChunker::default_words(100, 0);
        let chunks = chunker.chunk(text);

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, text);
    }
}
