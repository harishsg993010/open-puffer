//! Puffer vector database server.

use clap::Parser;
use puffer_api::{create_router, AppState};
use puffer_storage::Catalog;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::net::TcpListener;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Parser, Debug)]
#[command(name = "puffer-server")]
#[command(about = "Puffer vector database server")]
struct Args {
    /// Data directory for storing collections and segments
    #[arg(long, default_value = "./data")]
    data_dir: PathBuf,

    /// Address to bind the server to
    #[arg(long, default_value = "0.0.0.0:8080")]
    bind_addr: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "puffer_api=info,puffer_query=info,tower_http=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let args = Args::parse();

    // Create data directory if it doesn't exist
    std::fs::create_dir_all(&args.data_dir)?;

    tracing::info!("Opening catalog at {:?}", args.data_dir);
    let catalog = Arc::new(Catalog::open(&args.data_dir)?);

    let state = AppState::new(catalog);
    let app = create_router(state);

    let addr: SocketAddr = args.bind_addr.parse()?;
    tracing::info!("Starting server on {}", addr);

    let listener = TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
