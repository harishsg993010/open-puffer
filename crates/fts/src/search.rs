//! FTS search operations.

use crate::error::{FtsError, FtsResult};
use crate::index::FtsIndex;
use serde::{Deserialize, Serialize};
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::Value;
use tantivy::{DocAddress, Score, TantivyDocument};

/// Result of an FTS search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FtsSearchResult {
    /// Vector ID of the matching document.
    pub vector_id: String,

    /// BM25 score.
    pub score: f32,

    /// Matched text (if stored).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,

    /// Matched title (if stored).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,

    /// Document metadata (if stored).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

/// Search options for BM25 queries.
#[derive(Debug, Clone, Default)]
pub struct SearchOptions {
    /// Fields to search (empty = default field).
    pub fields: Vec<String>,

    /// Include text in results.
    pub include_text: bool,

    /// Include metadata in results.
    pub include_metadata: bool,

    /// Use fuzzy matching.
    pub fuzzy: bool,

    /// Fuzzy distance (edit distance).
    pub fuzzy_distance: u8,

    /// Minimum score threshold.
    pub min_score: Option<f32>,
}

impl SearchOptions {
    /// Create options with text included.
    pub fn with_text(mut self) -> Self {
        self.include_text = true;
        self
    }

    /// Create options with metadata included.
    pub fn with_metadata(mut self) -> Self {
        self.include_metadata = true;
        self
    }

    /// Enable fuzzy matching.
    pub fn with_fuzzy(mut self, distance: u8) -> Self {
        self.fuzzy = true;
        self.fuzzy_distance = distance;
        self
    }

    /// Set minimum score threshold.
    pub fn with_min_score(mut self, score: f32) -> Self {
        self.min_score = Some(score);
        self
    }
}

/// Perform a BM25 search on the index.
pub fn search_bm25(
    index: &FtsIndex,
    query_str: &str,
    top_k: usize,
) -> FtsResult<Vec<FtsSearchResult>> {
    search_bm25_with_options(index, query_str, top_k, &SearchOptions::default())
}

/// Perform a BM25 search with custom options.
pub fn search_bm25_with_options(
    index: &FtsIndex,
    query_str: &str,
    top_k: usize,
    options: &SearchOptions,
) -> FtsResult<Vec<FtsSearchResult>> {
    if query_str.trim().is_empty() {
        return Ok(Vec::new());
    }

    let searcher = index.searcher();

    // Build query parser with appropriate fields
    let fields = index.schema.searchable_fields();
    let query_parser = QueryParser::for_index(index.tantivy_index(), fields);

    // Parse the query
    let query = query_parser.parse_query(query_str)?;

    // Execute search
    let top_docs = searcher.search(&query, &TopDocs::with_limit(top_k))?;

    // Convert results
    let mut results = Vec::with_capacity(top_docs.len());

    for (score, doc_address) in top_docs {
        if let Some(min_score) = options.min_score {
            if score < min_score {
                continue;
            }
        }

        let doc: TantivyDocument = searcher.doc(doc_address)?;

        let vector_id = doc
            .get_first(index.schema.vector_id_field)
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| FtsError::DocumentNotFound("Missing vector_id".into()))?;

        let text = if options.include_text && index.config.store_text {
            doc.get_first(index.schema.text_field)
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        } else {
            None
        };

        let title = if options.include_text {
            index.schema.title_field.and_then(|f| {
                doc.get_first(f)
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
            })
        } else {
            None
        };

        let metadata = if options.include_metadata {
            index.schema.metadata_field.and_then(|f| {
                doc.get_first(f)
                    .and_then(|v| v.as_str())
                    .and_then(|s| serde_json::from_str(s).ok())
            })
        } else {
            None
        };

        results.push(FtsSearchResult {
            vector_id,
            score,
            text,
            title,
            metadata,
        });
    }

    Ok(results)
}

/// Search with term-based query (no parsing).
pub fn search_term(
    index: &FtsIndex,
    field: &str,
    term: &str,
    top_k: usize,
) -> FtsResult<Vec<FtsSearchResult>> {
    let searcher = index.searcher();

    let field = match field {
        "text" => index.schema.text_field,
        "title" => index.schema.title_field.ok_or_else(|| {
            FtsError::SchemaError("Title field not indexed".into())
        })?,
        "tags" => index.schema.tags_field.ok_or_else(|| {
            FtsError::SchemaError("Tags field not indexed".into())
        })?,
        _ => return Err(FtsError::SchemaError(format!("Unknown field: {}", field))),
    };

    let term = tantivy::Term::from_field_text(field, term);
    let query = tantivy::query::TermQuery::new(term, tantivy::schema::IndexRecordOption::Basic);

    let top_docs = searcher.search(&query, &TopDocs::with_limit(top_k))?;

    let mut results = Vec::with_capacity(top_docs.len());

    for (score, doc_address) in top_docs {
        let doc: TantivyDocument = searcher.doc(doc_address)?;

        let vector_id = doc
            .get_first(index.schema.vector_id_field)
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| FtsError::DocumentNotFound("Missing vector_id".into()))?;

        results.push(FtsSearchResult {
            vector_id,
            score,
            text: None,
            title: None,
            metadata: None,
        });
    }

    Ok(results)
}

/// Search with phrase query.
pub fn search_phrase(
    index: &FtsIndex,
    phrase: &str,
    top_k: usize,
) -> FtsResult<Vec<FtsSearchResult>> {
    let searcher = index.searcher();

    // Build phrase query
    let terms: Vec<_> = phrase
        .split_whitespace()
        .map(|t| tantivy::Term::from_field_text(index.schema.text_field, t))
        .collect();

    if terms.is_empty() {
        return Ok(Vec::new());
    }

    let query = tantivy::query::PhraseQuery::new(terms);
    let top_docs = searcher.search(&query, &TopDocs::with_limit(top_k))?;

    let mut results = Vec::with_capacity(top_docs.len());

    for (score, doc_address) in top_docs {
        let doc: TantivyDocument = searcher.doc(doc_address)?;

        let vector_id = doc
            .get_first(index.schema.vector_id_field)
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| FtsError::DocumentNotFound("Missing vector_id".into()))?;

        results.push(FtsSearchResult {
            vector_id,
            score,
            text: None,
            title: None,
            metadata: None,
        });
    }

    Ok(results)
}

/// Get all documents matching a query (for hybrid search).
pub fn search_all_matching(
    index: &FtsIndex,
    query_str: &str,
    limit: usize,
) -> FtsResult<Vec<(String, f32)>> {
    if query_str.trim().is_empty() {
        return Ok(Vec::new());
    }

    let searcher = index.searcher();
    let fields = index.schema.searchable_fields();
    let query_parser = QueryParser::for_index(index.tantivy_index(), fields);
    let query = query_parser.parse_query(query_str)?;

    let top_docs = searcher.search(&query, &TopDocs::with_limit(limit))?;

    let mut results = Vec::with_capacity(top_docs.len());

    for (score, doc_address) in top_docs {
        let doc: TantivyDocument = searcher.doc(doc_address)?;

        if let Some(vector_id) = doc
            .get_first(index.schema.vector_id_field)
            .and_then(|v| v.as_str())
        {
            results.push((vector_id.to_string(), score));
        }
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::TextDocument;
    use tempfile::tempdir;

    fn create_test_index() -> (tempfile::TempDir, FtsIndex) {
        let dir = tempdir().unwrap();
        let path = dir.path().join("fts");

        let config = crate::config::FtsConfig::enabled()
            .with_fields(vec!["text".to_string(), "title".to_string()]);

        let index = FtsIndex::create(&path, config).unwrap();

        // Add test documents
        let docs = vec![
            TextDocument::new("vec_0", "Machine learning is a subset of artificial intelligence")
                .with_title("ML Intro"),
            TextDocument::new("vec_1", "Deep learning uses neural networks with many layers")
                .with_title("Deep Learning"),
            TextDocument::new("vec_2", "Natural language processing helps computers understand text")
                .with_title("NLP Basics"),
            TextDocument::new("vec_3", "Computer vision enables machines to interpret images")
                .with_title("Computer Vision"),
            TextDocument::new("vec_4", "Reinforcement learning trains agents through rewards")
                .with_title("RL Overview"),
        ];

        for doc in docs {
            index.add_document(doc).unwrap();
        }
        index.commit().unwrap();

        (dir, index)
    }

    #[test]
    fn test_basic_search() {
        let (_dir, index) = create_test_index();

        let results = search_bm25(&index, "machine learning", 5).unwrap();

        assert!(!results.is_empty());
        // Results should contain vec_0 (the machine learning document)
        let ids: Vec<&str> = results.iter().map(|r| r.vector_id.as_str()).collect();
        assert!(ids.contains(&"vec_0"), "Should find the machine learning document");
    }

    #[test]
    fn test_search_with_options() {
        let (_dir, index) = create_test_index();

        let options = SearchOptions::default().with_text().with_min_score(0.1);
        let results = search_bm25_with_options(&index, "neural networks", 5, &options).unwrap();

        assert!(!results.is_empty());
        // Text should be included
        assert!(results[0].text.is_some());
    }

    #[test]
    fn test_search_all_matching() {
        let (_dir, index) = create_test_index();

        let results = search_all_matching(&index, "learning", 10).unwrap();

        // Should match multiple documents with "learning"
        assert!(results.len() >= 2);
    }

    #[test]
    fn test_empty_query() {
        let (_dir, index) = create_test_index();

        let results = search_bm25(&index, "", 5).unwrap();
        assert!(results.is_empty());

        let results = search_bm25(&index, "   ", 5).unwrap();
        assert!(results.is_empty());
    }
}
