//! Route definitions.

use crate::handlers;
use crate::state::AppState;
use axum::{
    extract::DefaultBodyLimit,
    routing::{delete, get, post},
    Router,
};
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;

/// Create the API router.
pub fn create_router(state: AppState) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        // Health check
        .route("/healthz", get(handlers::health_check))
        // Collections
        .route("/v1/collections", post(handlers::create_collection))
        .route("/v1/collections", get(handlers::list_collections))
        .route(
            "/v1/collections/:name",
            delete(handlers::delete_collection),
        )
        .route("/v1/collections/:name/stats", get(handlers::get_stats))
        .route("/v1/collections/:name/flush", post(handlers::flush_collection))
        // Points
        .route(
            "/v1/collections/:name/points",
            post(handlers::insert_points),
        )
        .route("/v1/collections/:name/search", post(handlers::search))
        .route("/v1/collections/:name/rebuild-router", post(handlers::rebuild_router))
        .route("/v1/collections/:name/compact", post(handlers::compact_collection))
        // Middleware
        .layer(DefaultBodyLimit::max(100 * 1024 * 1024)) // 100MB limit
        .layer(TraceLayer::new_for_http())
        .layer(cors)
        .with_state(state)
}
