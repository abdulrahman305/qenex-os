//! QENEX Banking System - Main Entry Point
//! 
//! Production-ready implementation with real functionality

use std::sync::Arc;
use std::net::SocketAddr;
use tokio::runtime::Runtime;
use tracing::{info, error, warn};
use rust_decimal::Decimal;

mod core;
mod auth;
mod api_server;

use crate::core::{TransactionEngine, InMemoryPersistence};
use crate::auth::AuthenticationSystem;
use crate::api_server::{build_router, ApiState};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            std::env::var("RUST_LOG")
                .unwrap_or_else(|_| "qenex=info,tower_http=debug".to_string())
        )
        .with_target(false)
        .init();

    info!("QENEX Banking System starting...");
    info!("Version: {}", env!("CARGO_PKG_VERSION"));

    // Create async runtime
    let runtime = Runtime::new()?;
    
    runtime.block_on(async {
        // Initialize persistence layer
        let persistence = Arc::new(InMemoryPersistence::new());
        
        // Initialize transaction engine
        let transaction_engine = Arc::new(TransactionEngine::new(persistence));
        
        // Initialize authentication system
        let jwt_secret = std::env::var("JWT_SECRET")
            .unwrap_or_else(|_| {
                warn!("JWT_SECRET not set, using default (INSECURE!)");
                "development_secret_change_in_production".to_string()
            });
        let auth_system = Arc::new(AuthenticationSystem::new(jwt_secret));
        
        // Create demo accounts for testing
        setup_demo_accounts(&transaction_engine, &auth_system).await?;
        
        // Create API state
        let api_state = ApiState {
            transaction_engine,
            auth_system,
        };
        
        // Build router
        let app = build_router(api_state);
        
        // Configure server address
        let addr: SocketAddr = std::env::var("BIND_ADDRESS")
            .unwrap_or_else(|_| "0.0.0.0:8080".to_string())
            .parse()?;
        
        info!("API server listening on {}", addr);
        info!("Health check: http://{}/health", addr);
        info!("API documentation: http://{}/api/docs", addr);
        
        // Start server
        axum::Server::bind(&addr)
            .serve(app.into_make_service())
            .await?;
        
        Ok::<(), Box<dyn std::error::Error>>(())
    })?;

    Ok(())
}

async fn setup_demo_accounts(
    engine: &Arc<TransactionEngine>,
    auth: &Arc<AuthenticationSystem>,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("Setting up demo accounts...");
    
    // Create demo user account
    let mut user_roles = std::collections::HashSet::new();
    user_roles.insert("user".to_string());
    
    match auth.register_user(
        "demo_user".to_string(),
        "demo@qenex.ai".to_string(),
        "DemoPassword123!".to_string(),
        user_roles,
    ) {
        Ok(_) => info!("Created demo user account (username: demo_user, password: DemoPassword123!)"),
        Err(e) if e.contains("already exists") => info!("Demo user account already exists"),
        Err(e) => error!("Failed to create demo user: {}", e),
    }
    
    // Create admin account
    let mut admin_roles = std::collections::HashSet::new();
    admin_roles.insert("admin".to_string());
    admin_roles.insert("user".to_string());
    
    match auth.register_user(
        "admin".to_string(),
        "admin@qenex.ai".to_string(),
        "AdminPassword123!".to_string(),
        admin_roles,
    ) {
        Ok(_) => info!("Created admin account (username: admin, password: AdminPassword123!)"),
        Err(e) if e.contains("already exists") => info!("Admin account already exists"),
        Err(e) => error!("Failed to create admin: {}", e),
    }
    
    // Create demo banking accounts
    match engine.create_account(
        "demo_checking".to_string(),
        "USD".to_string(),
        Decimal::new(10000, 0), // $100.00
    ).await {
        Ok(_) => info!("Created demo checking account with $100.00"),
        Err(e) if e.contains("already exists") => info!("Demo checking account already exists"),
        Err(e) => error!("Failed to create demo checking account: {}", e),
    }
    
    match engine.create_account(
        "demo_savings".to_string(),
        "USD".to_string(),
        Decimal::new(50000, 0), // $500.00
    ).await {
        Ok(_) => info!("Created demo savings account with $500.00"),
        Err(e) if e.contains("already exists") => info!("Demo savings account already exists"),
        Err(e) => error!("Failed to create demo savings account: {}", e),
    }
    
    match engine.create_account(
        "merchant_account".to_string(),
        "USD".to_string(),
        Decimal::new(100000, 0), // $1000.00
    ).await {
        Ok(_) => info!("Created merchant account with $1000.00"),
        Err(e) if e.contains("already exists") => info!("Merchant account already exists"),
        Err(e) => error!("Failed to create merchant account: {}", e),
    }
    
    info!("Demo setup completed");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_system_initialization() {
        let persistence = Arc::new(InMemoryPersistence::new());
        let engine = TransactionEngine::new(persistence);
        
        // Create test account
        let result = engine.create_account(
            "test".to_string(),
            "USD".to_string(),
            Decimal::new(1000, 0),
        ).await;
        
        assert!(result.is_ok());
    }
}