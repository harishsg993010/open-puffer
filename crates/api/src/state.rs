//! Application state.

use puffer_query::QueryEngine;
use puffer_storage::Catalog;
use std::sync::Arc;

/// Shared application state.
#[derive(Clone)]
pub struct AppState {
    pub catalog: Arc<Catalog>,
    pub engine: Arc<QueryEngine>,
}

impl AppState {
    pub fn new(catalog: Arc<Catalog>) -> Self {
        let engine = Arc::new(QueryEngine::new(catalog.clone()));
        Self { catalog, engine }
    }
}
