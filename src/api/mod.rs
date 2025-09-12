//! Banking API Server Implementation

use std::sync::Arc;
use axum::{
    Router,
    routing::{get, post},
    extract::{State, Json, Path, Query},
    response::{IntoResponse, Response},
    http::StatusCode,
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use rust_decimal::Decimal;

use crate::core::BankingCore;
use crate::transaction::{TransactionRequest, TransactionStatus};
use crate::compliance::{ScreeningResult, ComplianceRule};
use crate::monitoring::MetricsCollector;

pub struct ApiServer {
    banking_core: Arc<BankingCore>,
    metrics: Arc<MetricsCollector>,
    config: ApiConfig,
}

#[derive(Clone)]
pub struct ApiConfig {
    pub host: String,
    pub port: u16,
    pub tls_enabled: bool,
    pub rate_limit: u32,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8080,
            tls_enabled: true,
            rate_limit: 1000,
        }
    }
}

#[derive(Clone)]
struct AppState {
    banking_core: Arc<BankingCore>,
    metrics: Arc<MetricsCollector>,
}

impl ApiServer {
    pub fn new(
        banking_core: Arc<BankingCore>,
        metrics: Arc<MetricsCollector>,
        config: ApiConfig,
    ) -> Self {
        Self {
            banking_core,
            metrics,
            config,
        }
    }

    pub async fn start(self) -> Result<(), Box<dyn std::error::Error>> {
        let state = AppState {
            banking_core: self.banking_core,
            metrics: self.metrics,
        };

        let app = Router::new()
            // Health endpoints
            .route("/health", get(health_check))
            .route("/status", get(system_status))
            
            // Transaction endpoints
            .route("/api/v1/transactions", post(create_transaction))
            .route("/api/v1/transactions/:id", get(get_transaction))
            .route("/api/v1/transactions/:id/status", get(transaction_status))
            
            // Account endpoints
            .route("/api/v1/accounts", post(create_account))
            .route("/api/v1/accounts/:id", get(get_account))
            .route("/api/v1/accounts/:id/balance", get(get_balance))
            
            // Compliance endpoints
            .route("/api/v1/compliance/screen", post(compliance_screening))
            .route("/api/v1/compliance/report", post(generate_report))
            
            // Risk endpoints
            .route("/api/v1/risk/assess", post(risk_assessment))
            .route("/api/v1/risk/score/:entity_id", get(get_risk_score))
            
            // Protocol endpoints
            .route("/api/v1/swift/send", post(send_swift))
            .route("/api/v1/sepa/transfer", post(sepa_transfer))
            
            // Metrics endpoints
            .route("/metrics", get(get_metrics))
            .route("/metrics/transactions", get(transaction_metrics))
            
            .with_state(state);

        let addr = format!("{}:{}", self.config.host, self.config.port);
        tracing::info!("API server listening on {}", addr);

        let listener = tokio::net::TcpListener::bind(&addr).await?;
        axum::serve(listener, app.into_make_service()).await?;

        Ok(())
    }
}

// Health check endpoint
async fn health_check() -> impl IntoResponse {
    Json(HealthResponse {
        status: "healthy".to_string(),
        timestamp: chrono::Utc::now(),
    })
}

// System status endpoint
async fn system_status(State(state): State<AppState>) -> impl IntoResponse {
    let status = state.banking_core.get_system_status().await;
    Json(status)
}

// Create transaction
async fn create_transaction(
    State(state): State<AppState>,
    Json(req): Json<TransactionRequest>,
) -> Response {
    match state.banking_core.create_transaction(req).await {
        Ok(tx_id) => {
            state.metrics.increment_transaction_count();
            (StatusCode::CREATED, Json(TransactionResponse {
                id: tx_id,
                status: "created".to_string(),
            })).into_response()
        }
        Err(e) => {
            (StatusCode::BAD_REQUEST, Json(ErrorResponse {
                error: e.to_string(),
            })).into_response()
        }
    }
}

// Get transaction
async fn get_transaction(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> impl IntoResponse {
    match state.banking_core.get_transaction(id).await {
        Ok(tx) => (StatusCode::OK, Json(tx)),
        Err(_) => (StatusCode::NOT_FOUND, Json(ErrorResponse {
            error: "Transaction not found".to_string(),
        })),
    }
}

// Transaction status
async fn transaction_status(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> impl IntoResponse {
    match state.banking_core.get_transaction_status(id).await {
        Ok(status) => (StatusCode::OK, Json(status)),
        Err(_) => (StatusCode::NOT_FOUND, Json(ErrorResponse {
            error: "Transaction not found".to_string(),
        })),
    }
}

// Create account
async fn create_account(
    State(state): State<AppState>,
    Json(req): Json<CreateAccountRequest>,
) -> Response {
    match state.banking_core.create_account(req).await {
        Ok(account_id) => {
            (StatusCode::CREATED, Json(AccountResponse {
                id: account_id,
                status: "active".to_string(),
            })).into_response()
        }
        Err(e) => {
            (StatusCode::BAD_REQUEST, Json(ErrorResponse {
                error: e.to_string(),
            })).into_response()
        }
    }
}

// Get account
async fn get_account(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    match state.banking_core.get_account(&id).await {
        Ok(account) => (StatusCode::OK, Json(account)),
        Err(_) => (StatusCode::NOT_FOUND, Json(ErrorResponse {
            error: "Account not found".to_string(),
        })),
    }
}

// Get balance
async fn get_balance(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Response {
    match state.banking_core.get_balance(&id).await {
        Ok(balance) => (StatusCode::OK, Json(BalanceResponse {
            account_id: id,
            balance,
            currency: "USD".to_string(),
        })).into_response(),
        Err(_) => (StatusCode::NOT_FOUND, Json(ErrorResponse {
            error: "Account not found".to_string(),
        })).into_response(),
    }
}

// Compliance screening
async fn compliance_screening(
    State(state): State<AppState>,
    Json(req): Json<ScreeningRequest>,
) -> impl IntoResponse {
    match state.banking_core.screen_entity(req).await {
        Ok(result) => (StatusCode::OK, Json(result)),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse {
            error: e.to_string(),
        })),
    }
}

// Generate compliance report
async fn generate_report(
    State(state): State<AppState>,
    Json(req): Json<ReportRequest>,
) -> impl IntoResponse {
    match state.banking_core.generate_compliance_report(req).await {
        Ok(report) => (StatusCode::OK, Json(report)),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse {
            error: e.to_string(),
        })),
    }
}

// Risk assessment
async fn risk_assessment(
    State(state): State<AppState>,
    Json(req): Json<RiskAssessmentRequest>,
) -> impl IntoResponse {
    match state.banking_core.assess_risk(req).await {
        Ok(assessment) => (StatusCode::OK, Json(assessment)),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse {
            error: e.to_string(),
        })),
    }
}

// Get risk score
async fn get_risk_score(
    State(state): State<AppState>,
    Path(entity_id): Path<String>,
) -> Response {
    match state.banking_core.get_risk_score(&entity_id).await {
        Ok(score) => (StatusCode::OK, Json(RiskScoreResponse {
            entity_id,
            score,
            category: categorize_risk(score),
        })).into_response(),
        Err(_) => (StatusCode::NOT_FOUND, Json(ErrorResponse {
            error: "Entity not found".to_string(),
        })).into_response(),
    }
}

// Send SWIFT message
async fn send_swift(
    State(state): State<AppState>,
    Json(req): Json<SwiftMessageRequest>,
) -> Response {
    match state.banking_core.send_swift_message(req).await {
        Ok(msg_id) => (StatusCode::OK, Json(SwiftResponse {
            message_id: msg_id,
            status: "sent".to_string(),
        })).into_response(),
        Err(e) => (StatusCode::BAD_REQUEST, Json(ErrorResponse {
            error: e.to_string(),
        })).into_response(),
    }
}

// SEPA transfer
async fn sepa_transfer(
    State(state): State<AppState>,
    Json(req): Json<SEPATransferRequest>,
) -> Response {
    match state.banking_core.initiate_sepa_transfer(req).await {
        Ok(transfer_id) => (StatusCode::OK, Json(SEPAResponse {
            transfer_id,
            status: "initiated".to_string(),
        })).into_response(),
        Err(e) => (StatusCode::BAD_REQUEST, Json(ErrorResponse {
            error: e.to_string(),
        })).into_response(),
    }
}

// Get metrics
async fn get_metrics(State(state): State<AppState>) -> impl IntoResponse {
    let metrics = state.metrics.get_all_metrics().await;
    Json(metrics)
}

// Transaction metrics
async fn transaction_metrics(State(state): State<AppState>) -> impl IntoResponse {
    let metrics = state.metrics.get_transaction_metrics().await;
    Json(metrics)
}

fn categorize_risk(score: f64) -> String {
    match score {
        s if s >= 0.8 => "HIGH".to_string(),
        s if s >= 0.5 => "MEDIUM".to_string(),
        _ => "LOW".to_string(),
    }
}

// Request/Response types
#[derive(Deserialize)]
pub struct CreateAccountRequest {
    pub account_type: String,
    pub customer_id: String,
    pub currency: String,
    pub initial_balance: Option<Decimal>,
}

#[derive(Deserialize)]
pub struct ReportRequest {
    pub report_type: String,
    pub start_date: chrono::DateTime<chrono::Utc>,
    pub end_date: chrono::DateTime<chrono::Utc>,
}

#[derive(Deserialize)]
pub struct RiskAssessmentRequest {
    pub entity_id: String,
    pub transaction_amount: Decimal,
    pub context: serde_json::Value,
}

#[derive(Deserialize)]
pub struct SwiftMessageRequest {
    pub message_type: String,
    pub receiver_bic: String,
    pub amount: Decimal,
    pub currency: String,
    pub reference: String,
}

#[derive(Deserialize)]
pub struct SEPATransferRequest {
    pub debtor_iban: String,
    pub creditor_iban: String,
    pub amount: Decimal,
    pub reference: String,
}

#[derive(Serialize)]
struct HealthResponse {
    status: String,
    timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Serialize)]
struct TransactionResponse {
    id: Uuid,
    status: String,
}

#[derive(Serialize)]
struct AccountResponse {
    id: String,
    status: String,
}

#[derive(Serialize)]
struct BalanceResponse {
    account_id: String,
    balance: Decimal,
    currency: String,
}

#[derive(Serialize)]
struct RiskScoreResponse {
    entity_id: String,
    score: f64,
    category: String,
}

#[derive(Serialize)]
struct SwiftResponse {
    message_id: String,
    status: String,
}

#[derive(Serialize)]
struct SEPAResponse {
    transfer_id: String,
    status: String,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}