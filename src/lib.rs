//! QENEX Financial Operating System
//! 
//! Production-grade banking infrastructure with quantum-resistant security,
//! real-time transaction processing, and comprehensive compliance framework.

pub mod core;
pub mod network;
pub mod storage;
pub mod api;
pub mod consensus;
pub mod compliance;
pub mod monitoring;
pub mod ai;
pub mod protocols;
pub mod cluster;
pub mod testing;
pub mod kernel;
pub mod crypto;
pub mod transaction;

pub use core::{BankingCore, SystemConfig, SystemHealth, CoreError, Result};

/// Re-export commonly used types
pub use core::transaction::{Transaction, TransactionType, TransactionStatus, AccountTier};
pub use core::crypto::{CryptoProvider, Signature, KeyType};

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const BUILD_INFO: &str = concat!(
    env!("CARGO_PKG_VERSION"), 
    " (", 
    env!("VERGEN_GIT_SHA_SHORT"),
    ")"
);

/// Initialize logging and monitoring
pub fn init_telemetry() -> Result<()> {
    use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
    
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "qenex_os=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();
    
    // Initialize metrics
    let builder = metrics_exporter_prometheus::PrometheusBuilder::new();
    builder
        .install()
        .map_err(|e| CoreError::NetworkError(format!("Failed to install metrics: {}", e)))?;
    
    Ok(())
}