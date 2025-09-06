pub mod engine;
pub mod acid_engine;

#[cfg(test)]
pub mod acid_tests;

// Export both engines - ACID engine for production, legacy for compatibility  
pub use acid_engine::{
    ACIDTransactionEngine,
    ACIDEngineConfig,
    DistributedTransaction,
    DistributedTransactionStatus,
    TransactionOperation,
    OperationType,
    IsolationLevel,
    LockMode,
    ACIDError,
};

pub use engine::{
    TransactionEngine,
    TransactionEngineConfig,
    TransactionRequest,
    TransactionDetails,
    TransactionType,
    TransactionStatus,
    TransactionPriority,
    TransactionEngineError,
    TransactionEngineMetrics,
};

// Re-export commonly used types for compatibility
pub use crate::core::transaction::{Transaction, AccountTier};

use std::sync::Arc;
use sqlx::PgPool;

/// High-level transaction processor interface
pub struct TransactionProcessor {
    engine: Arc<TransactionEngine>,
}

impl TransactionProcessor {
    /// Create new transaction processor
    pub async fn new(
        db_pool: Arc<PgPool>,
        config: Option<TransactionEngineConfig>,
    ) -> Result<Self, TransactionEngineError> {
        let config = config.unwrap_or_default();
        let engine = Arc::new(TransactionEngine::new(db_pool, config).await?);
        
        Ok(Self { engine })
    }
    
    /// Process a transaction request
    pub async fn process_transaction(
        &self,
        request: TransactionRequest,
    ) -> Result<uuid::Uuid, TransactionEngineError> {
        let transaction_id = self.engine.submit_transaction(request).await?;
        
        // The engine will process the transaction asynchronously
        // In a real implementation, you might want to wait for completion
        // or return immediately and let the client poll for status
        
        Ok(transaction_id)
    }
    
    /// Get transaction status
    pub async fn get_status(
        &self,
        transaction_id: uuid::Uuid,
    ) -> Result<TransactionStatus, TransactionEngineError> {
        self.engine.get_transaction_status(transaction_id).await
    }
    
    /// Get detailed transaction information
    pub async fn get_details(
        &self,
        transaction_id: uuid::Uuid,
    ) -> Result<TransactionDetails, TransactionEngineError> {
        self.engine.get_transaction_details(transaction_id).await
    }
    
    /// Cancel a pending transaction
    pub async fn cancel_transaction(
        &self,
        transaction_id: uuid::Uuid,
        reason: String,
    ) -> Result<(), TransactionEngineError> {
        self.engine.cancel_transaction(transaction_id, reason).await
    }
    
    /// Get performance metrics
    pub async fn get_metrics(&self) -> TransactionEngineMetrics {
        self.engine.get_metrics().await
    }
}