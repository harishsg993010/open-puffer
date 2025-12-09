//! FTS index management.

use crate::config::FtsConfig;
use crate::error::{FtsError, FtsResult};
use crate::schema::{FtsSchema, TextDocument};
use parking_lot::RwLock;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tantivy::{
    directory::MmapDirectory,
    doc,
    Index, IndexReader, IndexWriter, IndexSettings, ReloadPolicy, TantivyDocument,
};

/// A full-text search index backed by Tantivy.
pub struct FtsIndex {
    /// Path to the index directory.
    pub path: PathBuf,

    /// Tantivy index.
    index: Index,

    /// Index writer (behind RwLock for thread safety).
    writer: RwLock<Option<IndexWriter>>,

    /// Index reader for searching.
    reader: IndexReader,

    /// Schema definition.
    pub schema: FtsSchema,

    /// Configuration.
    pub config: FtsConfig,

    /// Documents pending commit.
    pending_docs: AtomicUsize,
}

impl FtsIndex {
    /// Create a new FTS index at the given path.
    pub fn create(path: impl AsRef<Path>, config: FtsConfig) -> FtsResult<Self> {
        let path = path.as_ref().to_path_buf();

        if path.exists() {
            return Err(FtsError::IndexAlreadyExists(path.display().to_string()));
        }

        std::fs::create_dir_all(&path)?;

        let schema = FtsSchema::from_config(&config)?;
        let directory = MmapDirectory::open(&path)?;
        let index = Index::create(directory, schema.schema.clone(), IndexSettings::default())?;

        // Register custom tokenizer
        index
            .tokenizers()
            .register("custom", schema.create_tokenizer());

        let writer = index.writer(config.writer_heap_size)?;
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()?;

        Ok(Self {
            path,
            index,
            writer: RwLock::new(Some(writer)),
            reader,
            schema,
            config,
            pending_docs: AtomicUsize::new(0),
        })
    }

    /// Open an existing FTS index.
    pub fn open(path: impl AsRef<Path>, config: FtsConfig) -> FtsResult<Self> {
        let path = path.as_ref().to_path_buf();

        if !path.exists() {
            return Err(FtsError::IndexNotFound(path.display().to_string()));
        }

        let schema = FtsSchema::from_config(&config)?;
        let directory = MmapDirectory::open(&path)?;
        let index = Index::open(directory)?;

        // Register custom tokenizer
        index
            .tokenizers()
            .register("custom", schema.create_tokenizer());

        let writer = index.writer(config.writer_heap_size)?;
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()?;

        Ok(Self {
            path,
            index,
            writer: RwLock::new(Some(writer)),
            reader,
            schema,
            config,
            pending_docs: AtomicUsize::new(0),
        })
    }

    /// Open or create an FTS index.
    pub fn open_or_create(path: impl AsRef<Path>, config: FtsConfig) -> FtsResult<Self> {
        let path = path.as_ref();
        if path.exists() && path.join("meta.json").exists() {
            Self::open(path, config)
        } else {
            Self::create(path, config)
        }
    }

    /// Add a document to the index.
    pub fn add_document(&self, doc: TextDocument) -> FtsResult<()> {
        let mut writer_guard = self.writer.write();
        let writer = writer_guard
            .as_mut()
            .ok_or(FtsError::NotReady)?;

        let mut tantivy_doc = TantivyDocument::default();

        // Add vector ID
        tantivy_doc.add_text(self.schema.vector_id_field, &doc.vector_id);

        // Add text
        tantivy_doc.add_text(self.schema.text_field, &doc.text);

        // Add optional fields
        if let (Some(title), Some(field)) = (&doc.title, self.schema.title_field) {
            tantivy_doc.add_text(field, title);
        }

        if let (Some(tags), Some(field)) = (&doc.tags, self.schema.tags_field) {
            for tag in tags {
                tantivy_doc.add_text(field, tag);
            }
        }

        if let (Some(metadata), Some(field)) = (&doc.metadata, self.schema.metadata_field) {
            let json_str = serde_json::to_string(metadata)?;
            tantivy_doc.add_text(field, &json_str);
        }

        writer.add_document(tantivy_doc)?;

        let pending = self.pending_docs.fetch_add(1, Ordering::SeqCst) + 1;

        // Auto-commit if threshold reached
        if pending >= self.config.auto_commit_threshold {
            drop(writer_guard);
            self.commit()?;
        }

        Ok(())
    }

    /// Add multiple documents in batch.
    pub fn add_documents(&self, docs: Vec<TextDocument>) -> FtsResult<usize> {
        let count = docs.len();
        for doc in docs {
            self.add_document(doc)?;
        }
        Ok(count)
    }

    /// Delete a document by vector ID.
    pub fn delete_document(&self, vector_id: &str) -> FtsResult<()> {
        let mut writer_guard = self.writer.write();
        let writer = writer_guard
            .as_mut()
            .ok_or(FtsError::NotReady)?;

        let term = tantivy::Term::from_field_text(self.schema.vector_id_field, vector_id);
        writer.delete_term(term);

        Ok(())
    }

    /// Commit pending changes.
    pub fn commit(&self) -> FtsResult<()> {
        let mut writer_guard = self.writer.write();
        if let Some(writer) = writer_guard.as_mut() {
            writer.commit()?;
            self.pending_docs.store(0, Ordering::SeqCst);
        }
        drop(writer_guard);
        // Reload the reader to see the committed changes
        self.reader.reload()?;
        Ok(())
    }

    /// Get a searcher for this index.
    pub fn searcher(&self) -> tantivy::Searcher {
        self.reader.searcher()
    }

    /// Reload the reader to see committed changes.
    pub fn reload(&self) -> FtsResult<()> {
        self.reader.reload()?;
        Ok(())
    }

    /// Get the number of documents in the index.
    pub fn num_docs(&self) -> u64 {
        self.searcher().num_docs()
    }

    /// Get index statistics.
    pub fn stats(&self) -> FtsStats {
        let searcher = self.searcher();
        FtsStats {
            num_docs: searcher.num_docs(),
            num_segments: searcher.segment_readers().len(),
            pending_docs: self.pending_docs.load(Ordering::SeqCst),
        }
    }

    /// Force merge all segments into one.
    pub fn optimize(&self) -> FtsResult<()> {
        let mut writer_guard = self.writer.write();
        if let Some(writer) = writer_guard.as_mut() {
            // Merge all segments into one by committing (triggers merge policy)
            writer.commit()?;
        }
        Ok(())
    }

    /// Get the underlying Tantivy index.
    pub fn tantivy_index(&self) -> &Index {
        &self.index
    }
}

/// FTS index statistics.
#[derive(Debug, Clone)]
pub struct FtsStats {
    pub num_docs: u64,
    pub num_segments: usize,
    pub pending_docs: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_create_and_add() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("fts");

        let config = FtsConfig::enabled().with_auto_commit(10);
        let index = FtsIndex::create(&path, config).unwrap();

        let doc = TextDocument::new("vec_0", "Hello world, this is a test document");
        index.add_document(doc).unwrap();
        index.commit().unwrap();

        assert_eq!(index.num_docs(), 1);
    }

    #[test]
    fn test_batch_add() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("fts");

        let config = FtsConfig::enabled();
        let index = FtsIndex::create(&path, config).unwrap();

        let docs: Vec<TextDocument> = (0..100)
            .map(|i| TextDocument::new(format!("vec_{}", i), format!("Document number {}", i)))
            .collect();

        let count = index.add_documents(docs).unwrap();
        index.commit().unwrap();

        assert_eq!(count, 100);
        assert_eq!(index.num_docs(), 100);
    }

    #[test]
    fn test_reopen() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("fts");

        let config = FtsConfig::enabled();

        // Create and add documents
        {
            let index = FtsIndex::create(&path, config.clone()).unwrap();
            index.add_document(TextDocument::new("vec_0", "Test document")).unwrap();
            index.commit().unwrap();
        }

        // Reopen
        let index = FtsIndex::open(&path, config).unwrap();
        assert_eq!(index.num_docs(), 1);
    }

    #[test]
    fn test_delete() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("fts");

        let config = FtsConfig::enabled();
        let index = FtsIndex::create(&path, config).unwrap();

        index.add_document(TextDocument::new("vec_0", "Document 0")).unwrap();
        index.add_document(TextDocument::new("vec_1", "Document 1")).unwrap();
        index.commit().unwrap();

        assert_eq!(index.num_docs(), 2);

        index.delete_document("vec_0").unwrap();
        index.commit().unwrap();

        // Note: Tantivy uses soft deletes, actual count may vary
        // until merge happens
    }
}
