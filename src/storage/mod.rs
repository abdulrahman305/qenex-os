//! Storage Layer - Production Database Architecture
//! 
//! Distributed database system with ACID compliance,
//! high availability, and comprehensive audit trails

pub mod database;
pub mod migrations;
pub mod models;
pub mod cache;

use super::{CoreError, Result};
use sqlx::{PgPool, Pool, Postgres, Row};
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

/// Database connection configuration
#[derive(Debug, Clone)]
pub struct DatabaseConfig {
    pub url: String,
    pub max_connections: u32,
    pub min_connections: u32,
    pub connect_timeout: std::time::Duration,
    pub idle_timeout: std::time::Duration,
    pub max_lifetime: std::time::Duration,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            url: "postgresql://qenex:password@localhost/qenex".to_string(),
            max_connections: 20,
            min_connections: 5,
            connect_timeout: std::time::Duration::from_secs(30),
            idle_timeout: std::time::Duration::from_secs(600),
            max_lifetime: std::time::Duration::from_secs(1800),
        }
    }
}

/// Account information stored in database
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct Account {
    pub id: String,
    pub account_type: String,
    pub status: String,
    pub tier: String,
    pub daily_limit: Decimal,
    pub monthly_limit: Decimal,
    pub kyc_status: String,
    pub risk_rating: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub metadata: serde_json::Value,
}

/// Account balance information
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct AccountBalance {
    pub account_id: String,
    pub currency: String,
    pub available_balance: Decimal,
    pub pending_balance: Decimal,
    pub reserved_balance: Decimal,
    pub last_updated: DateTime<Utc>,
}

/// Transaction record in database
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct TransactionRecord {
    pub id: Uuid,
    pub sender: String,
    pub receiver: String,
    pub amount: Decimal,
    pub currency: String,
    pub transaction_type: String,
    pub status: String,
    pub priority: i32,
    pub reference: Option<String>,
    pub description: Option<String>,
    pub fee: Decimal,
    pub exchange_rate: Option<Decimal>,
    pub settlement_date: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub signature: Vec<u8>,
    pub metadata: serde_json::Value,
    pub compliance_flags: Vec<String>,
    pub risk_score: f64,
    pub block_hash: Option<String>,
    pub block_number: Option<i64>,
}

/// Audit log entry for compliance
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct AuditLog {
    pub id: Uuid,
    pub event_type: String,
    pub actor: String,
    pub resource: String,
    pub action: String,
    pub old_value: Option<serde_json::Value>,
    pub new_value: Option<serde_json::Value>,
    pub ip_address: Option<String>,
    pub user_agent: Option<String>,
    pub session_id: Option<String>,
    pub timestamp: DateTime<Utc>,
    pub severity: String,
}

/// Main storage manager
pub struct StorageManager {
    pool: Arc<PgPool>,
    cache: Arc<cache::CacheManager>,
    config: DatabaseConfig,
    metrics: Arc<RwLock<StorageMetrics>>,
}

/// Storage performance metrics
#[derive(Debug, Clone, Default)]
pub struct StorageMetrics {
    pub queries_executed: u64,
    pub average_query_time_ms: f64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub active_connections: u32,
    pub failed_queries: u64,
}

impl StorageManager {
    /// Create new storage manager
    pub async fn new(database_url: &str) -> Result<Self> {
        let config = DatabaseConfig {
            url: database_url.to_string(),
            ..Default::default()
        };
        
        // Create connection pool
        let pool = sqlx::postgres::PgPoolOptions::new()
            .max_connections(config.max_connections)
            .min_connections(config.min_connections)
            .acquire_timeout(config.connect_timeout)
            .idle_timeout(config.idle_timeout)
            .max_lifetime(config.max_lifetime)
            .connect(database_url)
            .await
            .map_err(|e| CoreError::StorageError(format!("Database connection failed: {}", e)))?;
        
        // Initialize cache
        let cache = Arc::new(cache::CacheManager::new().await?);
        
        Ok(Self {
            pool: Arc::new(pool),
            cache,
            config,
            metrics: Arc::new(RwLock::new(StorageMetrics::default())),
        })
    }
    
    /// Start storage manager
    pub async fn start(&self) -> Result<()> {
        // Run database migrations
        migrations::run_migrations(&self.pool).await?;
        
        // Initialize cache warming
        self.cache.warm_cache().await?;
        
        // Start background tasks
        self.start_maintenance_tasks().await;
        
        tracing::info!("Storage manager started successfully");
        Ok(())
    }
    
    /// Stop storage manager gracefully
    pub async fn stop(&self) -> Result<()> {
        // Close database connections
        self.pool.close().await;
        
        // Stop cache
        self.cache.stop().await?;
        
        tracing::info!("Storage manager stopped");
        Ok(())
    }
    
    /// Health check for storage system
    pub async fn health_check(&self) -> Result<()> {
        // Test database connectivity
        let result = sqlx::query("SELECT 1")
            .fetch_one(&*self.pool)
            .await
            .map_err(|e| CoreError::StorageError(format!("Database health check failed: {}", e)))?;
        
        // Verify cache connectivity
        self.cache.health_check().await?;
        
        Ok(())
    }
    
    /// Check if account exists
    pub async fn account_exists(&self, account_id: &str) -> Result<bool> {
        let start_time = std::time::Instant::now();
        
        // Check cache first
        if let Some(_) = self.cache.get_account(account_id).await {
            self.record_query_time(start_time, true).await;
            return Ok(true);
        }
        
        let exists = sqlx::query_scalar("SELECT EXISTS(SELECT 1 FROM accounts WHERE id = $1)")
            .bind(account_id)
            .fetch_one(&*self.pool)
            .await
            .map_err(|e| CoreError::StorageError(format!("Failed to check account existence: {}", e)))?
            .get::<bool, _>(0);
        
        self.record_query_time(start_time, false).await;
        Ok(exists)
    }
    
    /// Get account information
    pub async fn get_account_info(&self, account_id: &str) -> Result<Account> {
        let start_time = std::time::Instant::now();
        
        // Check cache first
        if let Some(account) = self.cache.get_account(account_id).await {
            self.record_query_time(start_time, true).await;
            return Ok(account);
        }
        
        let account = sqlx::query_as::<_, Account>(
            r#"
            SELECT id, account_type, status, tier, daily_limit, monthly_limit,
                   kyc_status, risk_rating, created_at, updated_at, metadata
            FROM accounts 
            WHERE id = $1
            "#
        )
        .bind(account_id)
        .fetch_one(&*self.pool)
        .await
        .map_err(|e| CoreError::StorageError(format!("Failed to get account info: {}", e)))?;
        
        // Cache the result
        self.cache.set_account(account_id, &account).await;
        
        self.record_query_time(start_time, false).await;
        Ok(account)
    }
    
    /// Get account balance for specific currency
    pub async fn get_balance(&self, account_id: &str) -> Result<Decimal> {
        let start_time = std::time::Instant::now();
        
        // Check cache first
        if let Some(balance) = self.cache.get_balance(account_id, "USD").await {
            self.record_query_time(start_time, true).await;
            return Ok(balance);
        }
        
        let balance = sqlx::query_scalar("SELECT available_balance FROM account_balances WHERE account_id = $1 AND currency = $2")
            .bind(account_id)
            .bind("USD")
            .fetch_optional(&*self.pool)
            .await
            .map_err(|e| CoreError::StorageError(format!("Failed to get balance: {}", e)))?
            .map(|row| row.get::<Decimal, _>(0))
            .unwrap_or(Decimal::ZERO);
        
        // Cache the result
        self.cache.set_balance(account_id, "USD", balance).await;
        
        self.record_query_time(start_time, false).await;
        Ok(balance)
    }
    
    /// Get all balances for an account
    pub async fn get_all_balances(&self, account_id: &str) -> Result<Vec<AccountBalance>> {
        let start_time = std::time::Instant::now();
        
        let balances = sqlx::query_as::<_, AccountBalance>(
            r#"
            SELECT account_id, currency, available_balance, pending_balance, 
                   reserved_balance, last_updated
            FROM account_balances 
            WHERE account_id = $1
            "#
        )
        .bind(account_id)
        .fetch_all(&*self.pool)
        .await
        .map_err(|e| CoreError::StorageError(format!("Failed to get all balances: {}", e)))?;
        
        self.record_query_time(start_time, false).await;
        Ok(balances)
    }
    
    /// Create new account
    pub async fn create_account(&self, account: &Account) -> Result<()> {
        let start_time = std::time::Instant::now();
        
        let mut tx = self.pool.begin()
            .await
            .map_err(|e| CoreError::StorageError(format!("Failed to begin transaction: {}", e)))?;
        
        // Insert account
        sqlx::query(
            r#"
            INSERT INTO accounts (id, account_type, status, tier, daily_limit, monthly_limit,
                                 kyc_status, risk_rating, created_at, updated_at, metadata)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            "#
        )
        .bind(&account.id)
        .bind(&account.account_type)
        .bind(&account.status)
        .bind(&account.tier)
        .bind(account.daily_limit)
        .bind(account.monthly_limit)
        .bind(&account.kyc_status)
        .bind(&account.risk_rating)
        .bind(account.created_at)
        .bind(account.updated_at)
        .bind(&account.metadata)
        .execute(&mut *tx)
        .await
        .map_err(|e| CoreError::StorageError(format!("Failed to create account: {}", e)))?;
        
        // Create default balances for major currencies
        let currencies = ["USD", "EUR", "GBP"];
        for currency in currencies {
            sqlx::query(
                r#"
                INSERT INTO account_balances (account_id, currency, available_balance, 
                                            pending_balance, reserved_balance, last_updated)
                VALUES ($1, $2, $3, $4, $5, $6)
                "#
            )
            .bind(&account.id)
            .bind(currency)
            .bind(Decimal::ZERO)
            .bind(Decimal::ZERO)
            .bind(Decimal::ZERO)
            .bind(Utc::now())
            .execute(&mut *tx)
            .await
            .map_err(|e| CoreError::StorageError(format!("Failed to create balance record: {}", e)))?;
        }
        
        tx.commit()
            .await
            .map_err(|e| CoreError::StorageError(format!("Failed to commit transaction: {}", e)))?;
        
        // Log audit trail
        self.log_audit_event(
            "account_created",
            "system",
            &account.id,
            "create",
            None,
            Some(serde_json::to_value(account).unwrap_or_default()),
        ).await?;
        
        // Invalidate cache
        self.cache.invalidate_account(&account.id).await;
        
        self.record_query_time(start_time, false).await;
        Ok(())
    }
    
    /// Update account balance (with transaction support)
    pub async fn update_balance(
        &self,
        account_id: &str,
        currency: &str,
        amount_change: Decimal,
        transaction_id: Uuid,
    ) -> Result<Decimal> {
        let start_time = std::time::Instant::now();
        
        let mut tx = self.pool.begin()
            .await
            .map_err(|e| CoreError::StorageError(format!("Failed to begin transaction: {}", e)))?;
        
        // Get current balance with row locking
        let current_balance = sqlx::query_scalar(
            r#"
            SELECT available_balance 
            FROM account_balances 
            WHERE account_id = $1 AND currency = $2
            FOR UPDATE
            "#
        )
        .bind(account_id)
        .bind(currency)
        .fetch_one(&mut *tx)
        .await
        .map_err(|e| CoreError::StorageError(format!("Failed to get current balance: {}", e)))?
        .get::<Decimal, _>(0);
        
        let new_balance = current_balance + amount_change;
        
        // Prevent negative balances
        if new_balance < Decimal::ZERO {
            return Err(CoreError::StorageError("Insufficient funds".to_string()));
        }
        
        // Update balance
        sqlx::query(
            r#"
            UPDATE account_balances 
            SET available_balance = $1, last_updated = $2
            WHERE account_id = $3 AND currency = $4
            "#
        )
        .bind(new_balance)
        .bind(Utc::now())
        .bind(account_id)
        .bind(currency)
        .execute(&mut *tx)
        .await
        .map_err(|e| CoreError::StorageError(format!("Failed to update balance: {}", e)))?;
        
        // Record balance change
        sqlx::query(
            r#"
            INSERT INTO balance_changes (account_id, currency, amount_change, 
                                       balance_before, balance_after, transaction_id, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            "#
        )
        .bind(account_id)
        .bind(currency)
        .bind(amount_change)
        .bind(current_balance)
        .bind(new_balance)
        .bind(transaction_id)
        .bind(Utc::now())
        .execute(&mut *tx)
        .await
        .map_err(|e| CoreError::StorageError(format!("Failed to record balance change: {}", e)))?;
        
        tx.commit()
            .await
            .map_err(|e| CoreError::StorageError(format!("Failed to commit transaction: {}", e)))?;
        
        // Invalidate cache
        self.cache.invalidate_balance(account_id, currency).await;
        
        // Log audit event
        self.log_audit_event(
            "balance_updated",
            "system",
            account_id,
            "update",
            Some(serde_json::json!({"balance": current_balance})),
            Some(serde_json::json!({"balance": new_balance, "change": amount_change})),
        ).await?;
        
        self.record_query_time(start_time, false).await;
        Ok(new_balance)
    }
    
    /// Save transaction record
    pub async fn save_transaction(&self, tx: &TransactionRecord) -> Result<()> {
        let start_time = std::time::Instant::now();
        
        sqlx::query(
            r#"
            INSERT INTO transactions (
                id, sender, receiver, amount, currency, transaction_type, status, priority,
                reference, description, fee, exchange_rate, settlement_date, created_at, 
                updated_at, signature, metadata, compliance_flags, risk_score, block_hash, block_number
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21)
            "#
        )
        .bind(tx.id)
        .bind(&tx.sender)
        .bind(&tx.receiver)
        .bind(tx.amount)
        .bind(&tx.currency)
        .bind(&tx.transaction_type)
        .bind(&tx.status)
        .bind(tx.priority)
        .bind(&tx.reference)
        .bind(&tx.description)
        .bind(tx.fee)
        .bind(tx.exchange_rate)
        .bind(tx.settlement_date)
        .bind(tx.created_at)
        .bind(tx.updated_at)
        .bind(&tx.signature)
        .bind(&tx.metadata)
        .bind(&tx.compliance_flags)
        .bind(tx.risk_score)
        .bind(&tx.block_hash)
        .bind(tx.block_number)
        .execute(&*self.pool)
        .await
        .map_err(|e| CoreError::StorageError(format!("Failed to save transaction: {}", e)))?;
        
        self.record_query_time(start_time, false).await;
        Ok(())
    }
    
    /// Get transaction by ID
    pub async fn get_transaction(&self, tx_id: &Uuid) -> Result<Option<TransactionRecord>> {
        let start_time = std::time::Instant::now();
        
        let tx = sqlx::query_as::<_, TransactionRecord>(
            r#"
            SELECT id, sender, receiver, amount, currency, transaction_type, status, priority,
                   reference, description, fee, exchange_rate, settlement_date, created_at,
                   updated_at, signature, metadata, compliance_flags, risk_score, block_hash, block_number
            FROM transactions 
            WHERE id = $1
            "#
        )
        .bind(tx_id)
        .fetch_optional(&*self.pool)
        .await
        .map_err(|e| CoreError::StorageError(format!("Failed to get transaction: {}", e)))?;
        
        self.record_query_time(start_time, false).await;
        Ok(tx)
    }
    
    /// Get transactions for account with pagination
    pub async fn get_account_transactions(
        &self,
        account_id: &str,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<TransactionRecord>> {
        let start_time = std::time::Instant::now();
        
        let transactions = sqlx::query_as::<_, TransactionRecord>(
            r#"
            SELECT id, sender, receiver, amount, currency, transaction_type, status, priority,
                   reference, description, fee, exchange_rate, settlement_date, created_at,
                   updated_at, signature, metadata, compliance_flags, risk_score, block_hash, block_number
            FROM transactions 
            WHERE sender = $1 OR receiver = $1
            ORDER BY created_at DESC
            LIMIT $2 OFFSET $3
            "#
        )
        .bind(account_id)
        .bind(limit)
        .bind(offset)
        .fetch_all(&*self.pool)
        .await
        .map_err(|e| CoreError::StorageError(format!("Failed to get account transactions: {}", e)))?;
        
        self.record_query_time(start_time, false).await;
        Ok(transactions)
    }
    
    /// Log audit event for compliance
    pub async fn log_audit_event(
        &self,
        event_type: &str,
        actor: &str,
        resource: &str,
        action: &str,
        old_value: Option<serde_json::Value>,
        new_value: Option<serde_json::Value>,
    ) -> Result<()> {
        let audit_log = AuditLog {
            id: Uuid::new_v4(),
            event_type: event_type.to_string(),
            actor: actor.to_string(),
            resource: resource.to_string(),
            action: action.to_string(),
            old_value,
            new_value,
            ip_address: None,
            user_agent: None,
            session_id: None,
            timestamp: Utc::now(),
            severity: "info".to_string(),
        };
        
        sqlx::query(
            r#"
            INSERT INTO audit_logs (
                id, event_type, actor, resource, action, old_value, new_value,
                ip_address, user_agent, session_id, timestamp, severity
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            "#
        )
        .bind(audit_log.id)
        .bind(&audit_log.event_type)
        .bind(&audit_log.actor)
        .bind(&audit_log.resource)
        .bind(&audit_log.action)
        .bind(&audit_log.old_value)
        .bind(&audit_log.new_value)
        .bind(&audit_log.ip_address)
        .bind(&audit_log.user_agent)
        .bind(&audit_log.session_id)
        .bind(audit_log.timestamp)
        .bind(&audit_log.severity)
        .execute(&*self.pool)
        .await
        .map_err(|e| CoreError::StorageError(format!("Failed to log audit event: {}", e)))?;
        
        Ok(())
    }
    
    /// Get storage metrics
    pub async fn get_metrics(&self) -> StorageMetrics {
        let metrics = self.metrics.read().await;
        metrics.clone()
    }
    
    /// Record query execution time
    async fn record_query_time(&self, start_time: std::time::Instant, cache_hit: bool) {
        let mut metrics = self.metrics.write().await;
        let elapsed = start_time.elapsed().as_millis() as f64;
        
        metrics.queries_executed += 1;
        metrics.average_query_time_ms = 
            (metrics.average_query_time_ms * (metrics.queries_executed - 1) as f64 + elapsed) / 
            metrics.queries_executed as f64;
        
        if cache_hit {
            metrics.cache_hits += 1;
        } else {
            metrics.cache_misses += 1;
        }
    }
    
    /// Start background maintenance tasks
    async fn start_maintenance_tasks(&self) {
        let pool = Arc::clone(&self.pool);
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(3600)); // Every hour
            
            loop {
                interval.tick().await;
                
                // Cleanup old audit logs (keep 1 year)
                let cutoff = Utc::now() - chrono::Duration::days(365);
                if let Err(e) = sqlx::query("DELETE FROM audit_logs WHERE timestamp < $1")
                    .bind(cutoff)
                .execute(&*pool)
                .await {
                    tracing::warn!("Failed to cleanup old audit logs: {}", e);
                }
                
                // Update database statistics
                if let Err(e) = sqlx::query("ANALYZE")
                    .execute(&*pool)
                    .await {
                    tracing::warn!("Failed to update database statistics: {}", e);
                }
                
                tracing::debug!("Database maintenance completed");
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sqlx::PgPool;
    use testcontainers::{clients::Cli, images::postgres::Postgres, RunnableImage};
    
    async fn setup_test_db() -> PgPool {
        // This would use testcontainers to spin up a test PostgreSQL instance
        // For now, we'll skip actual database tests in the template
        todo!("Implement test database setup")
    }
    
    #[tokio::test]
    async fn test_account_operations() {
        // Test account creation, balance updates, etc.
        // Implementation would go here
    }
    
    #[tokio::test] 
    async fn test_transaction_storage() {
        // Test transaction saving and retrieval
        // Implementation would go here
    }
}