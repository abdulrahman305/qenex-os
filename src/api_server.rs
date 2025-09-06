//! REST API Server with proper validation and error handling

use axum::{
    Router,
    routing::{get, post, put, delete},
    extract::{State, Json, Path, Query, Extension},
    response::{IntoResponse, Response},
    middleware,
    http::{StatusCode, header},
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tower::ServiceBuilder;
use tower_http::{
    cors::CorsLayer,
    trace::TraceLayer,
    compression::CompressionLayer,
    limit::RequestBodyLimitLayer,
};
use validator::Validate;
use uuid::Uuid;
use rust_decimal::Decimal;

use crate::core::{TransactionEngine, Transaction, TransactionStatus, InMemoryPersistence};
use crate::auth::{AuthenticationSystem, Claims};

/// API Server state
pub struct ApiState {
    pub transaction_engine: Arc<TransactionEngine>,
    pub auth_system: Arc<AuthenticationSystem>,
}

/// API Error response
#[derive(Debug, Serialize)]
pub struct ApiError {
    pub error: String,
    pub message: String,
    pub status_code: u16,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = StatusCode::from_u16(self.status_code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        (status, Json(self)).into_response()
    }
}

/// Create account request
#[derive(Debug, Deserialize, Validate)]
pub struct CreateAccountRequest {
    #[validate(length(min = 3, max = 50))]
    pub id: String,
    #[validate(length(equal = 3))]
    pub currency: String,
    #[validate(range(min = 0))]
    pub initial_balance: Decimal,
}

/// Create account response
#[derive(Debug, Serialize)]
pub struct CreateAccountResponse {
    pub account_id: String,
    pub status: String,
}

/// Process transaction request
#[derive(Debug, Deserialize, Validate)]
pub struct ProcessTransactionRequest {
    #[validate(length(min = 1, max = 50))]
    pub from_account: String,
    #[validate(length(min = 1, max = 50))]
    pub to_account: String,
    #[validate(range(min = 0.01))]
    pub amount: Decimal,
    #[validate(length(equal = 3))]
    pub currency: String,
    #[validate(length(max = 255))]
    pub reference: String,
}

/// Transaction response
#[derive(Debug, Serialize)]
pub struct TransactionResponse {
    pub transaction_id: Uuid,
    pub status: String,
}

/// Balance response
#[derive(Debug, Serialize)]
pub struct BalanceResponse {
    pub account_id: String,
    pub balance: Decimal,
    pub currency: String,
}

/// Login request
#[derive(Debug, Deserialize, Validate)]
pub struct LoginRequest {
    #[validate(length(min = 3, max = 50))]
    pub username: String,
    #[validate(length(min = 8, max = 128))]
    pub password: String,
    pub mfa_token: Option<String>,
}

/// Login response
#[derive(Debug, Serialize)]
pub struct LoginResponse {
    pub token: String,
    pub expires_in: u32,
}

/// Register request
#[derive(Debug, Deserialize, Validate)]
pub struct RegisterRequest {
    #[validate(length(min = 3, max = 50))]
    pub username: String,
    #[validate(email)]
    pub email: String,
    #[validate(length(min = 12, max = 128))]
    pub password: String,
}

/// Health check response
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub uptime: u64,
}

/// Build API router
pub fn build_router(state: ApiState) -> Router {
    Router::new()
        // Public endpoints
        .route("/health", get(health_check))
        .route("/api/v1/auth/register", post(register))
        .route("/api/v1/auth/login", post(login))
        
        // Protected endpoints
        .route("/api/v1/accounts", post(create_account))
        .route("/api/v1/accounts/:id/balance", get(get_balance))
        .route("/api/v1/transactions", post(process_transaction))
        .route("/api/v1/transactions/:id", get(get_transaction))
        .route("/api/v1/transactions/:id/reverse", post(reverse_transaction))
        
        // Add middleware
        .layer(
            ServiceBuilder::new()
                .layer(TraceLayer::new_for_http())
                .layer(CompressionLayer::new())
                .layer(CorsLayer::permissive())
                .layer(RequestBodyLimitLayer::new(1024 * 1024)) // 1MB limit
                .layer(middleware::from_fn_with_state(
                    Arc::new(state.auth_system.clone()),
                    auth_middleware,
                ))
        )
        .with_state(Arc::new(state))
}

/// Authentication middleware
async fn auth_middleware<B>(
    State(auth): State<Arc<Arc<AuthenticationSystem>>>,
    req: axum::http::Request<B>,
    next: middleware::Next<B>,
) -> Result<Response, ApiError> {
    // Skip authentication for public endpoints
    let path = req.uri().path();
    if path == "/health" || path.starts_with("/api/v1/auth/") {
        return Ok(next.run(req).await);
    }
    
    // Extract token from Authorization header
    let auth_header = req.headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| ApiError {
            error: "UNAUTHORIZED".to_string(),
            message: "Missing authorization header".to_string(),
            status_code: 401,
        })?;
    
    let token = auth_header
        .strip_prefix("Bearer ")
        .ok_or_else(|| ApiError {
            error: "UNAUTHORIZED".to_string(),
            message: "Invalid authorization header format".to_string(),
            status_code: 401,
        })?;
    
    // Validate token
    let claims = auth.validate_token(token)
        .map_err(|e| ApiError {
            error: "UNAUTHORIZED".to_string(),
            message: e,
            status_code: 401,
        })?;
    
    // Add claims to request extensions
    let mut req = req;
    req.extensions_mut().insert(claims);
    
    Ok(next.run(req).await)
}

/// Health check endpoint
async fn health_check() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "healthy".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        uptime: std::process::id() as u64, // Simplified
    })
}

/// Register endpoint
async fn register(
    State(state): State<Arc<ApiState>>,
    Json(req): Json<RegisterRequest>,
) -> Result<StatusCode, ApiError> {
    req.validate()
        .map_err(|e| ApiError {
            error: "VALIDATION_ERROR".to_string(),
            message: e.to_string(),
            status_code: 400,
        })?;
    
    let mut roles = std::collections::HashSet::new();
    roles.insert("user".to_string());
    
    state.auth_system.register_user(
        req.username,
        req.email,
        req.password,
        roles,
    ).map_err(|e| ApiError {
        error: "REGISTRATION_FAILED".to_string(),
        message: e,
        status_code: 400,
    })?;
    
    Ok(StatusCode::CREATED)
}

/// Login endpoint
async fn login(
    State(state): State<Arc<ApiState>>,
    Json(req): Json<LoginRequest>,
) -> Result<Json<LoginResponse>, ApiError> {
    req.validate()
        .map_err(|e| ApiError {
            error: "VALIDATION_ERROR".to_string(),
            message: e.to_string(),
            status_code: 400,
        })?;
    
    // Verify MFA if provided
    if let Some(mfa_token) = req.mfa_token {
        let valid = state.auth_system.verify_mfa(&req.username, &mfa_token)
            .map_err(|e| ApiError {
                error: "MFA_FAILED".to_string(),
                message: e,
                status_code: 401,
            })?;
        
        if !valid {
            return Err(ApiError {
                error: "MFA_FAILED".to_string(),
                message: "Invalid MFA token".to_string(),
                status_code: 401,
            });
        }
    }
    
    let token = state.auth_system.authenticate(
        &req.username,
        &req.password,
        "0.0.0.0".to_string(), // Should get from request
        "API".to_string(),
    ).map_err(|e| ApiError {
        error: "AUTH_FAILED".to_string(),
        message: e,
        status_code: 401,
    })?;
    
    Ok(Json(LoginResponse {
        token,
        expires_in: 28800, // 8 hours
    }))
}

/// Create account endpoint
async fn create_account(
    State(state): State<Arc<ApiState>>,
    Extension(claims): Extension<Claims>,
    Json(req): Json<CreateAccountRequest>,
) -> Result<Json<CreateAccountResponse>, ApiError> {
    req.validate()
        .map_err(|e| ApiError {
            error: "VALIDATION_ERROR".to_string(),
            message: e.to_string(),
            status_code: 400,
        })?;
    
    // Check permission
    if !claims.permissions.contains(&"create_account".to_string()) 
        && !claims.roles.contains(&"admin".to_string()) {
        return Err(ApiError {
            error: "FORBIDDEN".to_string(),
            message: "Insufficient permissions".to_string(),
            status_code: 403,
        });
    }
    
    let account_id = state.transaction_engine
        .create_account(req.id.clone(), req.currency, req.initial_balance)
        .await
        .map_err(|e| ApiError {
            error: "ACCOUNT_CREATION_FAILED".to_string(),
            message: e,
            status_code: 400,
        })?;
    
    Ok(Json(CreateAccountResponse {
        account_id,
        status: "created".to_string(),
    }))
}

/// Get balance endpoint
async fn get_balance(
    State(state): State<Arc<ApiState>>,
    Extension(_claims): Extension<Claims>,
    Path(account_id): Path<String>,
) -> Result<Json<BalanceResponse>, ApiError> {
    let balance = state.transaction_engine
        .get_balance(&account_id)
        .map_err(|e| ApiError {
            error: "ACCOUNT_NOT_FOUND".to_string(),
            message: e,
            status_code: 404,
        })?;
    
    Ok(Json(BalanceResponse {
        account_id,
        balance,
        currency: "USD".to_string(), // Should get from account
    }))
}

/// Process transaction endpoint
async fn process_transaction(
    State(state): State<Arc<ApiState>>,
    Extension(_claims): Extension<Claims>,
    Json(req): Json<ProcessTransactionRequest>,
) -> Result<Json<TransactionResponse>, ApiError> {
    req.validate()
        .map_err(|e| ApiError {
            error: "VALIDATION_ERROR".to_string(),
            message: e.to_string(),
            status_code: 400,
        })?;
    
    let transaction = Transaction {
        id: Uuid::new_v4(),
        from_account: req.from_account,
        to_account: req.to_account,
        amount: req.amount,
        currency: req.currency,
        status: TransactionStatus::Pending,
        created_at: chrono::Utc::now(),
        completed_at: None,
        reference: req.reference,
        metadata: std::collections::HashMap::new(),
    };
    
    let transaction_id = state.transaction_engine
        .process_transaction(transaction)
        .await
        .map_err(|e| ApiError {
            error: "TRANSACTION_FAILED".to_string(),
            message: e,
            status_code: 400,
        })?;
    
    Ok(Json(TransactionResponse {
        transaction_id,
        status: "completed".to_string(),
    }))
}

/// Get transaction endpoint
async fn get_transaction(
    State(state): State<Arc<ApiState>>,
    Extension(_claims): Extension<Claims>,
    Path(transaction_id): Path<Uuid>,
) -> Result<Json<TransactionStatus>, ApiError> {
    let status = state.transaction_engine
        .get_transaction_status(transaction_id)
        .map_err(|e| ApiError {
            error: "TRANSACTION_NOT_FOUND".to_string(),
            message: e,
            status_code: 404,
        })?;
    
    Ok(Json(status))
}

/// Reverse transaction endpoint
async fn reverse_transaction(
    State(state): State<Arc<ApiState>>,
    Extension(claims): Extension<Claims>,
    Path(transaction_id): Path<Uuid>,
) -> Result<Json<TransactionResponse>, ApiError> {
    // Check permission
    if !claims.permissions.contains(&"reverse_transaction".to_string()) 
        && !claims.roles.contains(&"admin".to_string()) {
        return Err(ApiError {
            error: "FORBIDDEN".to_string(),
            message: "Insufficient permissions".to_string(),
            status_code: 403,
        });
    }
    
    let reversal_id = state.transaction_engine
        .reverse_transaction(transaction_id)
        .await
        .map_err(|e| ApiError {
            error: "REVERSAL_FAILED".to_string(),
            message: e,
            status_code: 400,
        })?;
    
    Ok(Json(TransactionResponse {
        transaction_id: reversal_id,
        status: "reversed".to_string(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;
    
    async fn setup_test_app() -> Router {
        let persistence = Arc::new(InMemoryPersistence::new());
        let transaction_engine = Arc::new(TransactionEngine::new(persistence));
        let auth_system = Arc::new(AuthenticationSystem::new("test_secret".to_string()));
        
        let state = ApiState {
            transaction_engine,
            auth_system,
        };
        
        build_router(state)
    }
    
    #[tokio::test]
    async fn test_health_endpoint() {
        let app = setup_test_app().await;
        
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        
        assert_eq!(response.status(), StatusCode::OK);
    }
    
    #[tokio::test]
    async fn test_unauthorized_request() {
        let app = setup_test_app().await;
        
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/v1/accounts")
                    .method("POST")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"id":"test","currency":"USD","initial_balance":1000}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }
}