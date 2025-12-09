//! FTS schema definitions.

use crate::config::FtsConfig;
use crate::error::{FtsError, FtsResult};
use serde::{Deserialize, Serialize};
use tantivy::schema::{
    Field, IndexRecordOption, Schema, SchemaBuilder, TextFieldIndexing, TextOptions, STORED,
};
use tantivy::tokenizer::{
    LowerCaser, RemoveLongFilter, SimpleTokenizer, Stemmer, TextAnalyzer,
};

/// A document to be indexed for full-text search.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TextDocument {
    /// The vector ID this document corresponds to.
    pub vector_id: String,

    /// Main text content.
    pub text: String,

    /// Optional title field.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,

    /// Optional tags/keywords.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tags: Option<Vec<String>>,

    /// Optional metadata as JSON.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

impl TextDocument {
    /// Create a new document with just text.
    pub fn new(vector_id: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            vector_id: vector_id.into(),
            text: text.into(),
            ..Default::default()
        }
    }

    /// Add a title.
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Add tags.
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = Some(tags);
        self
    }

    /// Add metadata.
    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = Some(metadata);
        self
    }
}

/// Schema definition for the FTS index.
pub struct FtsSchema {
    /// Tantivy schema.
    pub schema: Schema,

    /// Vector ID field (stored, not indexed).
    pub vector_id_field: Field,

    /// Main text field.
    pub text_field: Field,

    /// Title field (optional).
    pub title_field: Option<Field>,

    /// Tags field (optional).
    pub tags_field: Option<Field>,

    /// Metadata field (stored JSON).
    pub metadata_field: Option<Field>,

    /// Configuration used to build this schema.
    pub config: FtsConfig,
}

impl FtsSchema {
    /// Build a schema from configuration.
    pub fn from_config(config: &FtsConfig) -> FtsResult<Self> {
        config.validate().map_err(FtsError::Config)?;

        let mut builder = SchemaBuilder::new();

        // Vector ID field - stored but not indexed
        let vector_id_field = builder.add_text_field("vector_id", STORED);

        // Build text field options
        let text_options = Self::build_text_options(config);

        // Main text field
        let text_field = builder.add_text_field("text", text_options.clone());

        // Optional title field
        let title_field = if config.indexed_fields.contains(&"title".to_string()) {
            Some(builder.add_text_field("title", text_options.clone()))
        } else {
            None
        };

        // Optional tags field
        let tags_field = if config.indexed_fields.contains(&"tags".to_string()) {
            Some(builder.add_text_field("tags", text_options.clone()))
        } else {
            None
        };

        // Metadata field - stored JSON
        let metadata_field = Some(builder.add_text_field("metadata", STORED));

        let schema = builder.build();

        Ok(Self {
            schema,
            vector_id_field,
            text_field,
            title_field,
            tags_field,
            metadata_field,
            config: config.clone(),
        })
    }

    /// Build text field options based on config.
    fn build_text_options(config: &FtsConfig) -> TextOptions {
        let indexing = TextFieldIndexing::default()
            .set_tokenizer("custom")
            .set_index_option(if config.enable_positions {
                IndexRecordOption::WithFreqsAndPositions
            } else {
                IndexRecordOption::WithFreqs
            });

        let mut options = TextOptions::default().set_indexing_options(indexing);

        if config.store_text {
            options = options.set_stored();
        }

        options
    }

    /// Create a custom tokenizer for Tantivy.
    pub fn create_tokenizer(&self) -> TextAnalyzer {
        let tokenizer = SimpleTokenizer::default();

        if self.config.enable_stemming {
            let stemmer = match self.config.stemmer_language.as_str() {
                "english" => Stemmer::new(tantivy::tokenizer::Language::English),
                "german" => Stemmer::new(tantivy::tokenizer::Language::German),
                "french" => Stemmer::new(tantivy::tokenizer::Language::French),
                "spanish" => Stemmer::new(tantivy::tokenizer::Language::Spanish),
                "italian" => Stemmer::new(tantivy::tokenizer::Language::Italian),
                "portuguese" => Stemmer::new(tantivy::tokenizer::Language::Portuguese),
                "russian" => Stemmer::new(tantivy::tokenizer::Language::Russian),
                _ => Stemmer::new(tantivy::tokenizer::Language::English),
            };

            TextAnalyzer::builder(tokenizer)
                .filter(RemoveLongFilter::limit(40))
                .filter(LowerCaser)
                .filter(stemmer)
                .build()
        } else {
            TextAnalyzer::builder(tokenizer)
                .filter(RemoveLongFilter::limit(40))
                .filter(LowerCaser)
                .build()
        }
    }

    /// Get all searchable fields for query parser.
    pub fn searchable_fields(&self) -> Vec<Field> {
        let mut fields = vec![self.text_field];

        if let Some(f) = self.title_field {
            fields.push(f);
        }
        if let Some(f) = self.tags_field {
            fields.push(f);
        }

        fields
    }

    /// Get the default search field.
    pub fn default_field(&self) -> Field {
        match self.config.default_field.as_str() {
            "title" => self.title_field.unwrap_or(self.text_field),
            "tags" => self.tags_field.unwrap_or(self.text_field),
            _ => self.text_field,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_creation() {
        let config = FtsConfig::enabled();
        let schema = FtsSchema::from_config(&config).unwrap();

        assert!(schema.title_field.is_none()); // Not in indexed_fields by default
        assert!(schema.metadata_field.is_some());
    }

    #[test]
    fn test_schema_with_all_fields() {
        let config = FtsConfig::enabled()
            .with_fields(vec![
                "text".to_string(),
                "title".to_string(),
                "tags".to_string(),
            ]);

        let schema = FtsSchema::from_config(&config).unwrap();

        assert!(schema.title_field.is_some());
        assert!(schema.tags_field.is_some());
    }

    #[test]
    fn test_text_document() {
        let doc = TextDocument::new("vec_0", "Hello world")
            .with_title("Greeting")
            .with_tags(vec!["hello".to_string(), "world".to_string()]);

        assert_eq!(doc.vector_id, "vec_0");
        assert_eq!(doc.text, "Hello world");
        assert_eq!(doc.title, Some("Greeting".to_string()));
    }
}
