//! HTTP request handlers.

use crate::state::AppState;
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use puffer_core::{Metric, VectorId, VectorRecord};
use puffer_embed::ModelType;
use puffer_storage::CollectionConfig;
use serde::{Deserialize, Serialize};

/// Error response.
#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
}

impl ErrorResponse {
    pub fn new(msg: impl Into<String>) -> Self {
        Self { error: msg.into() }
    }
}

/// Health check response.
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
}

/// Health check handler.
pub async fn health_check() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

/// Create collection request.
#[derive(Debug, Deserialize)]
pub struct CreateCollectionRequest {
    pub name: String,
    pub dimension: usize,
    #[serde(default)]
    pub metric: Option<Metric>,
    #[serde(default)]
    pub staging_threshold: Option<usize>,
    #[serde(default)]
    pub num_clusters: Option<usize>,
    #[serde(default)]
    pub router_top_m: Option<usize>,
    #[serde(default)]
    pub l0_max_segments: Option<usize>,
    #[serde(default)]
    pub segment_target_size: Option<usize>,
}

/// Create collection response.
#[derive(Debug, Serialize)]
pub struct CreateCollectionResponse {
    pub name: String,
    pub dimension: usize,
    pub metric: Metric,
}

/// Create a new collection.
pub async fn create_collection(
    State(state): State<AppState>,
    Json(req): Json<CreateCollectionRequest>,
) -> impl IntoResponse {
    let config = CollectionConfig {
        name: req.name.clone(),
        dimension: req.dimension,
        metric: req.metric.unwrap_or(Metric::Cosine),
        staging_threshold: req.staging_threshold.unwrap_or(10_000),
        num_clusters: req.num_clusters.unwrap_or(100),
        router_top_m: req.router_top_m.unwrap_or(5),
        l0_max_segments: req.l0_max_segments.unwrap_or(10),
        segment_target_size: req.segment_target_size.unwrap_or(100_000),
    };

    match state.catalog.create_collection(config.clone()) {
        Ok(()) => (
            StatusCode::CREATED,
            Json(CreateCollectionResponse {
                name: config.name,
                dimension: config.dimension,
                metric: config.metric,
            }),
        )
            .into_response(),
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new(e.to_string())),
        )
            .into_response(),
    }
}

/// Collection info.
#[derive(Debug, Serialize)]
pub struct CollectionInfo {
    pub name: String,
    pub dimension: usize,
    pub metric: Metric,
}

/// List all collections.
pub async fn list_collections(State(state): State<AppState>) -> impl IntoResponse {
    let names = state.catalog.list_collections();

    let collections: Vec<CollectionInfo> = names
        .into_iter()
        .filter_map(|name| {
            state.catalog.get_collection(&name).ok().map(|c| {
                let coll = c.read().unwrap();
                CollectionInfo {
                    name: coll.meta.config.name.clone(),
                    dimension: coll.meta.config.dimension,
                    metric: coll.meta.config.metric,
                }
            })
        })
        .collect();

    Json(collections)
}

/// Point in insert request.
#[derive(Debug, Deserialize)]
pub struct PointInput {
    pub id: String,
    pub vector: Vec<f32>,
    #[serde(default)]
    pub payload: Option<serde_json::Value>,
}

/// Insert points request.
#[derive(Debug, Deserialize)]
pub struct InsertPointsRequest {
    pub points: Vec<PointInput>,
}

/// Insert points response.
#[derive(Debug, Serialize)]
pub struct InsertPointsResponse {
    pub inserted: usize,
}

/// Insert points into a collection.
pub async fn insert_points(
    State(state): State<AppState>,
    Path(collection_name): Path<String>,
    Json(req): Json<InsertPointsRequest>,
) -> impl IntoResponse {
    let records: Vec<VectorRecord> = req
        .points
        .into_iter()
        .map(|p| {
            let mut record = VectorRecord::new(VectorId::new(p.id), p.vector);
            if let Some(payload) = p.payload {
                record = record.with_payload(payload);
            }
            record
        })
        .collect();

    match state.engine.insert(&collection_name, records) {
        Ok(count) => (StatusCode::OK, Json(InsertPointsResponse { inserted: count })).into_response(),
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new(e.to_string())),
        )
            .into_response(),
    }
}

/// Search request.
#[derive(Debug, Deserialize)]
pub struct SearchRequest {
    pub vector: Vec<f32>,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    #[serde(default = "default_nprobe")]
    pub nprobe: usize,
    #[serde(default)]
    pub include_payload: bool,
}

fn default_top_k() -> usize {
    10
}

fn default_nprobe() -> usize {
    4
}

/// Search result item.
#[derive(Debug, Serialize)]
pub struct SearchResultItem {
    pub id: String,
    pub distance: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payload: Option<serde_json::Value>,
}

/// Search response.
#[derive(Debug, Serialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResultItem>,
}

/// Search for similar vectors.
pub async fn search(
    State(state): State<AppState>,
    Path(collection_name): Path<String>,
    Json(req): Json<SearchRequest>,
) -> impl IntoResponse {
    match state.engine.search(
        &collection_name,
        &req.vector,
        req.top_k,
        req.nprobe,
        req.include_payload,
    ) {
        Ok(results) => {
            let items: Vec<SearchResultItem> = results
                .into_iter()
                .map(|r| SearchResultItem {
                    id: r.id.to_string(),
                    distance: r.distance,
                    payload: r.payload,
                })
                .collect();
            (StatusCode::OK, Json(SearchResponse { results: items })).into_response()
        }
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new(e.to_string())),
        )
            .into_response(),
    }
}

/// Get collection statistics.
pub async fn get_stats(
    State(state): State<AppState>,
    Path(collection_name): Path<String>,
) -> impl IntoResponse {
    match state.engine.stats(&collection_name) {
        Ok(stats) => (StatusCode::OK, Json(stats)).into_response(),
        Err(e) => (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse::new(e.to_string())),
        )
            .into_response(),
    }
}

/// Delete collection.
pub async fn delete_collection(
    State(state): State<AppState>,
    Path(collection_name): Path<String>,
) -> impl IntoResponse {
    match state.catalog.drop_collection(&collection_name) {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse::new(e.to_string())),
        )
            .into_response(),
    }
}

/// Flush staging buffer to segment.
pub async fn flush_collection(
    State(state): State<AppState>,
    Path(collection_name): Path<String>,
) -> impl IntoResponse {
    match state.engine.force_flush(&collection_name) {
        Ok(()) => StatusCode::OK.into_response(),
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new(e.to_string())),
        )
            .into_response(),
    }
}

/// Rebuild router response.
#[derive(Debug, Serialize)]
pub struct RebuildRouterResponse {
    pub segments_rebuilt: usize,
}

/// Rebuild router index for a collection.
pub async fn rebuild_router(
    State(state): State<AppState>,
    Path(collection_name): Path<String>,
) -> impl IntoResponse {
    match state.engine.rebuild_router(&collection_name) {
        Ok(count) => (
            StatusCode::OK,
            Json(RebuildRouterResponse { segments_rebuilt: count }),
        )
            .into_response(),
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new(e.to_string())),
        )
            .into_response(),
    }
}

/// Trigger compaction for a collection.
pub async fn compact_collection(
    State(state): State<AppState>,
    Path(collection_name): Path<String>,
) -> impl IntoResponse {
    match state.engine.compact(&collection_name) {
        Ok(result) => (StatusCode::OK, Json(result)).into_response(),
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new(e.to_string())),
        )
            .into_response(),
    }
}

// =============================================================================
// Full-Text Search Handlers
// =============================================================================

/// Text search request.
#[derive(Debug, Deserialize)]
pub struct TextSearchRequest {
    /// Search query string.
    pub query: String,
    /// Number of results to return.
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    /// Include text in results.
    #[serde(default)]
    pub include_text: bool,
    /// Include metadata in results.
    #[serde(default)]
    pub include_metadata: bool,
    /// Minimum BM25 score threshold.
    #[serde(default)]
    pub min_score: Option<f32>,
}

/// Text search result item.
#[derive(Debug, Serialize)]
pub struct TextSearchResultItem {
    pub id: String,
    pub score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

/// Text search response.
#[derive(Debug, Serialize)]
pub struct TextSearchResponse {
    pub results: Vec<TextSearchResultItem>,
    pub query_time_ms: u64,
}

/// Full-text search handler.
pub async fn text_search(
    State(state): State<AppState>,
    Path(collection_name): Path<String>,
    Json(req): Json<TextSearchRequest>,
) -> impl IntoResponse {
    let start = std::time::Instant::now();

    // Check if FTS is available for this collection
    let fts_index = match state.get_fts_index(&collection_name) {
        Some(index) => index,
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse::new("Full-text search not enabled for this collection")),
            )
                .into_response();
        }
    };

    use puffer_fts::search::{search_bm25_with_options, SearchOptions};

    let options = SearchOptions::default()
        .with_text()
        .with_metadata();

    match search_bm25_with_options(&fts_index, &req.query, req.top_k, &options) {
        Ok(results) => {
            let items: Vec<TextSearchResultItem> = results
                .into_iter()
                .filter(|r| req.min_score.map(|min| r.score >= min).unwrap_or(true))
                .map(|r| TextSearchResultItem {
                    id: r.vector_id,
                    score: r.score,
                    text: if req.include_text { r.text } else { None },
                    title: if req.include_text { r.title } else { None },
                    metadata: if req.include_metadata { r.metadata } else { None },
                })
                .collect();

            (
                StatusCode::OK,
                Json(TextSearchResponse {
                    results: items,
                    query_time_ms: start.elapsed().as_millis() as u64,
                }),
            )
                .into_response()
        }
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new(e.to_string())),
        )
            .into_response(),
    }
}

// =============================================================================
// Hybrid Search Handlers
// =============================================================================

/// Hybrid search request.
#[derive(Debug, Deserialize)]
pub struct HybridSearchRequest {
    /// Text query for BM25 search.
    pub text_query: String,
    /// Vector for similarity search.
    pub vector: Vec<f32>,
    /// Number of results to return.
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    /// Lambda weight for text vs vector (0=vector only, 1=text only).
    #[serde(default = "default_lambda")]
    pub lambda: f32,
    /// Number of clusters to probe for vector search.
    #[serde(default = "default_nprobe")]
    pub nprobe: usize,
    /// Fusion method.
    #[serde(default)]
    pub fusion_method: Option<String>,
    /// RRF k parameter (only for RRF fusion).
    #[serde(default)]
    pub rrf_k: Option<f32>,
    /// Number of candidates from each source.
    #[serde(default = "default_candidates")]
    pub candidates_per_source: usize,
    /// Include payload in results.
    #[serde(default)]
    pub include_payload: bool,
}

fn default_lambda() -> f32 {
    0.5
}

fn default_candidates() -> usize {
    100
}

/// Hybrid search result item.
#[derive(Debug, Serialize)]
pub struct HybridSearchResultItem {
    pub id: String,
    pub score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text_score: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector_score: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text_rank: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector_rank: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payload: Option<serde_json::Value>,
}

/// Hybrid search response.
#[derive(Debug, Serialize)]
pub struct HybridSearchResponse {
    pub results: Vec<HybridSearchResultItem>,
    pub query_time_ms: u64,
    pub text_candidates: usize,
    pub vector_candidates: usize,
}

/// Hybrid vector + text search handler.
pub async fn hybrid_search(
    State(state): State<AppState>,
    Path(collection_name): Path<String>,
    Json(req): Json<HybridSearchRequest>,
) -> impl IntoResponse {
    let start = std::time::Instant::now();

    // Get FTS index
    let fts_index = match state.get_fts_index(&collection_name) {
        Some(index) => index,
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse::new("Full-text search not enabled for this collection")),
            )
                .into_response();
        }
    };

    // Get text results
    use puffer_fts::search::search_all_matching;
    let text_results = match search_all_matching(&fts_index, &req.text_query, req.candidates_per_source) {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse::new(format!("Text search failed: {}", e))),
            )
                .into_response();
        }
    };

    // Get vector results
    let vector_results = match state.engine.search(
        &collection_name,
        &req.vector,
        req.candidates_per_source,
        req.nprobe,
        req.include_payload,
    ) {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse::new(format!("Vector search failed: {}", e))),
            )
                .into_response();
        }
    };

    // Convert vector results for hybrid fusion
    use puffer_fts::hybrid::{convert_vector_results, hybrid_search as fuse_results, FusionMethod, HybridConfig, VectorResult};

    let vector_for_fusion: Vec<VectorResult> = vector_results
        .iter()
        .map(|r| VectorResult {
            id: r.id.to_string(),
            distance: r.distance,
        })
        .collect();

    // Build fusion config
    let fusion = match req.fusion_method.as_deref() {
        Some("rrf") | Some("reciprocal_rank_fusion") => {
            FusionMethod::ReciprocalRankFusion { k: req.rrf_k.unwrap_or(60.0) }
        }
        Some("normalized") | Some("normalized_weighted_sum") => {
            FusionMethod::NormalizedWeightedSum
        }
        Some("softmax") => {
            FusionMethod::SoftmaxFusion { temperature: 1.0 }
        }
        _ => FusionMethod::WeightedSum,
    };

    let config = HybridConfig {
        lambda: req.lambda,
        fusion,
        candidates_per_source: req.candidates_per_source,
        min_text_score: None,
        max_vector_distance: None,
    };

    // Fuse results
    let hybrid_results = fuse_results(&text_results, &vector_for_fusion, &config, req.top_k);

    // Build response with payloads
    let payload_map: std::collections::HashMap<String, Option<serde_json::Value>> = vector_results
        .into_iter()
        .map(|r| (r.id.to_string(), r.payload))
        .collect();

    let items: Vec<HybridSearchResultItem> = hybrid_results
        .into_iter()
        .map(|r| HybridSearchResultItem {
            id: r.id.clone(),
            score: r.score,
            text_score: r.text_score,
            vector_score: r.vector_score,
            text_rank: r.text_rank,
            vector_rank: r.vector_rank,
            payload: if req.include_payload {
                payload_map.get(&r.id).cloned().flatten()
            } else {
                None
            },
        })
        .collect();

    (
        StatusCode::OK,
        Json(HybridSearchResponse {
            results: items,
            query_time_ms: start.elapsed().as_millis() as u64,
            text_candidates: text_results.len(),
            vector_candidates: vector_for_fusion.len(),
        }),
    )
        .into_response()
}

// =============================================================================
// Index Text Documents
// =============================================================================

/// Add text document request.
#[derive(Debug, Deserialize)]
pub struct AddTextDocumentRequest {
    pub documents: Vec<TextDocumentInput>,
}

/// Input for a text document.
#[derive(Debug, Deserialize)]
pub struct TextDocumentInput {
    /// Vector ID this document corresponds to.
    pub vector_id: String,
    /// Main text content.
    pub text: String,
    /// Optional title.
    #[serde(default)]
    pub title: Option<String>,
    /// Optional tags.
    #[serde(default)]
    pub tags: Option<Vec<String>>,
    /// Optional metadata.
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
}

/// Add text documents response.
#[derive(Debug, Serialize)]
pub struct AddTextDocumentsResponse {
    pub indexed: usize,
}

/// Add text documents to FTS index.
pub async fn add_text_documents(
    State(state): State<AppState>,
    Path(collection_name): Path<String>,
    Json(req): Json<AddTextDocumentRequest>,
) -> impl IntoResponse {
    let fts_index = match state.get_fts_index(&collection_name) {
        Some(index) => index,
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse::new("Full-text search not enabled for this collection")),
            )
                .into_response();
        }
    };

    use puffer_fts::schema::TextDocument;

    let mut indexed = 0;
    for doc_input in req.documents {
        let mut doc = TextDocument::new(doc_input.vector_id, doc_input.text);

        if let Some(title) = doc_input.title {
            doc = doc.with_title(title);
        }
        if let Some(tags) = doc_input.tags {
            doc = doc.with_tags(tags);
        }
        if let Some(metadata) = doc_input.metadata {
            doc = doc.with_metadata(metadata);
        }

        if fts_index.add_document(doc).is_ok() {
            indexed += 1;
        }
    }

    // Commit the changes
    if let Err(e) = fts_index.commit() {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse::new(format!("Failed to commit: {}", e))),
        )
            .into_response();
    }

    (
        StatusCode::OK,
        Json(AddTextDocumentsResponse { indexed }),
    )
        .into_response()
}

// =============================================================================
// Embedding Handlers
// =============================================================================

/// Configure embedder request.
#[derive(Debug, Deserialize)]
pub struct ConfigureEmbedderRequest {
    /// Model type: jina, bert, sentence_transformer, clip, openai, cohere.
    #[serde(default)]
    pub model_type: Option<String>,
    /// Custom model ID (optional).
    #[serde(default)]
    pub model_id: Option<String>,
    /// API key for cloud models (openai, cohere).
    #[serde(default)]
    pub api_key: Option<String>,
    /// Batch size for embedding.
    #[serde(default)]
    pub batch_size: Option<usize>,
}

/// Embedder info response.
#[derive(Debug, Serialize)]
pub struct EmbedderInfoResponse {
    pub model_type: String,
    pub model_id: String,
    pub dimension: usize,
    pub initialized: bool,
}

/// Configure the embedder model.
pub async fn configure_embedder(
    State(state): State<AppState>,
    Json(req): Json<ConfigureEmbedderRequest>,
) -> impl IntoResponse {
    use puffer_embed::EmbedConfig;

    let model_type = match req.model_type.as_deref() {
        Some("jina") | None => ModelType::Jina,
        Some("bert") => ModelType::Bert,
        Some("sentence_transformer") | Some("sentence-transformer") => ModelType::SentenceTransformer,
        Some("clip") => ModelType::Clip,
        Some("openai") => ModelType::OpenAI,
        Some("cohere") => ModelType::Cohere,
        Some(other) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse::new(format!("Unknown model type: {}", other))),
            )
                .into_response();
        }
    };

    let mut config = EmbedConfig::new(model_type.clone());

    if let Some(model_id) = req.model_id {
        config = config.with_model_id(model_id);
    }
    if let Some(api_key) = req.api_key {
        config = config.with_api_key(api_key);
    }
    if let Some(batch_size) = req.batch_size {
        config = config.with_batch_size(batch_size);
    }

    state.configure_embedder(config);

    (
        StatusCode::OK,
        Json(serde_json::json!({
            "status": "configured",
            "model_type": format!("{:?}", model_type),
        })),
    )
        .into_response()
}

/// Get embedder info.
pub async fn get_embedder_info(State(state): State<AppState>) -> impl IntoResponse {
    match state.get_embedder().await {
        Ok(embedder) => {
            let config = embedder.config();
            (
                StatusCode::OK,
                Json(EmbedderInfoResponse {
                    model_type: format!("{:?}", config.model_type),
                    model_id: config.get_model_id().to_string(),
                    dimension: embedder.dimension(),
                    initialized: true,
                }),
            )
                .into_response()
        }
        Err(e) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorResponse::new(e)),
        )
            .into_response(),
    }
}

/// Embed texts request.
#[derive(Debug, Deserialize)]
pub struct EmbedTextsRequest {
    /// Texts to embed.
    pub texts: Vec<String>,
}

/// Embed texts response.
#[derive(Debug, Serialize)]
pub struct EmbedTextsResponse {
    pub embeddings: Vec<Vec<f32>>,
    pub dimension: usize,
    pub count: usize,
}

/// Embed multiple texts.
pub async fn embed_texts(
    State(state): State<AppState>,
    Json(req): Json<EmbedTextsRequest>,
) -> impl IntoResponse {
    if req.texts.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new("No texts provided")),
        )
            .into_response();
    }

    let embedder = match state.get_embedder().await {
        Ok(e) => e,
        Err(e) => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(ErrorResponse::new(e)),
            )
                .into_response();
        }
    };

    match embedder.embed_texts(&req.texts).await {
        Ok(results) => {
            let embeddings: Vec<Vec<f32>> = results.into_iter().map(|r| r.embedding).collect();
            let dimension = embedder.dimension();
            let count = embeddings.len();

            (
                StatusCode::OK,
                Json(EmbedTextsResponse {
                    embeddings,
                    dimension,
                    count,
                }),
            )
                .into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse::new(e.to_string())),
        )
            .into_response(),
    }
}

/// Embed and insert request (combined operation).
#[derive(Debug, Deserialize)]
pub struct EmbedAndInsertRequest {
    /// Documents to embed and insert.
    pub documents: Vec<DocumentInput>,
}

/// Document input for embedding.
#[derive(Debug, Deserialize)]
pub struct DocumentInput {
    /// Document ID.
    pub id: String,
    /// Text content to embed.
    pub text: String,
    /// Optional metadata/payload.
    #[serde(default)]
    pub payload: Option<serde_json::Value>,
}

/// Embed and insert response.
#[derive(Debug, Serialize)]
pub struct EmbedAndInsertResponse {
    pub embedded: usize,
    pub inserted: usize,
    pub dimension: usize,
}

/// Embed texts and insert into collection.
pub async fn embed_and_insert(
    State(state): State<AppState>,
    Path(collection_name): Path<String>,
    Json(req): Json<EmbedAndInsertRequest>,
) -> impl IntoResponse {
    if req.documents.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new("No documents provided")),
        )
            .into_response();
    }

    // Get embedder
    let embedder = match state.get_embedder().await {
        Ok(e) => e,
        Err(e) => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(ErrorResponse::new(e)),
            )
                .into_response();
        }
    };

    // Extract texts
    let texts: Vec<String> = req.documents.iter().map(|d| d.text.clone()).collect();

    // Generate embeddings
    let embeddings = match embedder.embed_texts(&texts).await {
        Ok(results) => results,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse::new(format!("Embedding failed: {}", e))),
            )
                .into_response();
        }
    };

    // Build vector records
    let records: Vec<VectorRecord> = req
        .documents
        .into_iter()
        .zip(embeddings.into_iter())
        .map(|(doc, emb)| {
            let mut record = VectorRecord::new(VectorId::new(doc.id), emb.embedding);
            if let Some(payload) = doc.payload {
                record = record.with_payload(payload);
            }
            record
        })
        .collect();

    let dimension = embedder.dimension();
    let embedded = records.len();

    // Insert into collection
    match state.engine.insert(&collection_name, records) {
        Ok(inserted) => (
            StatusCode::OK,
            Json(EmbedAndInsertResponse {
                embedded,
                inserted,
                dimension,
            }),
        )
            .into_response(),
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new(e.to_string())),
        )
            .into_response(),
    }
}

/// Semantic search request (text query -> embedding -> search).
#[derive(Debug, Deserialize)]
pub struct SemanticSearchRequest {
    /// Text query to search for.
    pub query: String,
    /// Number of results.
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    /// Number of clusters to probe.
    #[serde(default = "default_nprobe")]
    pub nprobe: usize,
    /// Include payload in results.
    #[serde(default)]
    pub include_payload: bool,
}

/// Semantic search: embed query text and search.
pub async fn semantic_search(
    State(state): State<AppState>,
    Path(collection_name): Path<String>,
    Json(req): Json<SemanticSearchRequest>,
) -> impl IntoResponse {
    let start = std::time::Instant::now();

    // Get embedder
    let embedder = match state.get_embedder().await {
        Ok(e) => e,
        Err(e) => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(ErrorResponse::new(e)),
            )
                .into_response();
        }
    };

    // Embed query
    let query_vector = match embedder.embed_text(&req.query).await {
        Ok(v) => v,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse::new(format!("Failed to embed query: {}", e))),
            )
                .into_response();
        }
    };

    // Search
    match state.engine.search(
        &collection_name,
        &query_vector,
        req.top_k,
        req.nprobe,
        req.include_payload,
    ) {
        Ok(results) => {
            let items: Vec<SearchResultItem> = results
                .into_iter()
                .map(|r| SearchResultItem {
                    id: r.id.to_string(),
                    distance: r.distance,
                    payload: r.payload,
                })
                .collect();

            (
                StatusCode::OK,
                Json(serde_json::json!({
                    "results": items,
                    "query_time_ms": start.elapsed().as_millis() as u64,
                })),
            )
                .into_response()
        }
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new(e.to_string())),
        )
            .into_response(),
    }
}
