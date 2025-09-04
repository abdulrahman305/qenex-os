//! QENEX Banking Operating System - Main Entry Point
//! 
//! Production-ready banking OS with complete implementation

use std::sync::Arc;
use tokio::runtime::Runtime;
use tracing::{info, error};

mod core;
mod kernel;
mod transaction;
mod security;
mod compliance;
mod protocols;
mod ai;
mod api;
mod monitoring;

use crate::core::{BankingCore, SystemConfig};
use crate::api::ApiServer;
use crate::monitoring::MetricsCollector;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("qenex_os=info")
        .with_target(false)
        .init();

    info!("QENEX Banking OS starting...");

    // Load configuration
    let config = SystemConfig::from_file("/etc/qenex/config.toml")
        .unwrap_or_default();

    // Create async runtime
    let runtime = Runtime::new()?;
    
    runtime.block_on(async {
        // Initialize banking core
        let banking_core = Arc::new(
            BankingCore::new(config.clone()).await?
        );

        // Start metrics collector
        let metrics = Arc::new(MetricsCollector::new());
        tokio::spawn({
            let metrics = metrics.clone();
            async move {
                metrics.start_collection().await;
            }
        });

        // Initialize API server
        let api_server = ApiServer::new(
            banking_core.clone(),
            metrics.clone(),
            config.api_config.clone()
        );

        // Start API server
        tokio::spawn(async move {
            if let Err(e) = api_server.start().await {
                error!("API server error: {}", e);
            }
        });

        // Initialize protocol handlers
        initialize_protocols(banking_core.clone()).await?;

        // Start compliance monitoring
        initialize_compliance(banking_core.clone()).await?;

        // Initialize AI/ML engine
        initialize_ai(banking_core.clone()).await?;

        info!("QENEX Banking OS initialized successfully");
        info!("System ready for banking operations");

        // Keep running
        tokio::signal::ctrl_c().await?;
        info!("Shutting down QENEX Banking OS...");

        Ok::<(), Box<dyn std::error::Error>>(())
    })?;

    Ok(())
}

async fn initialize_protocols(core: Arc<BankingCore>) -> Result<(), Box<dyn std::error::Error>> {
    use crate::protocols::{SwiftGateway, ISO20022Processor, SEPAHandler};

    info!("Initializing banking protocols...");

    // SWIFT
    let swift = SwiftGateway::new(core.clone()).await?;
    tokio::spawn(async move {
        swift.start_processing().await;
    });

    // ISO 20022
    let iso = ISO20022Processor::new(core.clone()).await?;
    tokio::spawn(async move {
        iso.start_processing().await;
    });

    // SEPA
    let sepa = SEPAHandler::new(core.clone()).await?;
    tokio::spawn(async move {
        sepa.start_processing().await;
    });

    info!("Banking protocols initialized");
    Ok(())
}

async fn initialize_compliance(core: Arc<BankingCore>) -> Result<(), Box<dyn std::error::Error>> {
    use crate::compliance::{ComplianceEngine, RealTimeScreening};

    info!("Initializing compliance engine...");

    let compliance = ComplianceEngine::new(core.clone()).await?;
    
    // Start real-time screening
    let screening = RealTimeScreening::new(compliance.clone()).await?;
    tokio::spawn(async move {
        screening.start_monitoring().await;
    });

    info!("Compliance engine initialized");
    Ok(())
}

async fn initialize_ai(core: Arc<BankingCore>) -> Result<(), Box<dyn std::error::Error>> {
    use crate::ai::{FraudDetector, RiskAssessor};

    info!("Initializing AI/ML engine...");

    // Fraud detection
    let fraud_detector = FraudDetector::new(core.clone()).await?;
    tokio::spawn(async move {
        fraud_detector.start_monitoring().await;
    });

    // Risk assessment
    let risk_assessor = RiskAssessor::new(core.clone()).await?;
    tokio::spawn(async move {
        risk_assessor.start_assessment().await;
    });

    info!("AI/ML engine initialized");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_system_initialization() {
        let config = SystemConfig::default();
        let core = BankingCore::new(config).await.unwrap();
        assert!(core.is_operational());
    }
}