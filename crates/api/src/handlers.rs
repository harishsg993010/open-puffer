//! HTTP request handlers.

use crate::state::AppState;
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use puffer_core::{Metric, VectorId, VectorRecord};
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
