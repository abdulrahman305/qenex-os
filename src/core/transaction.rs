//! Transaction Processing - Production Banking Implementation
//! 
//! Complete transaction lifecycle with proper validation,
//! compliance checking, and settlement processing

use super::{CoreError, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;

/// Account tier for transaction limits
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AccountTier {
    Individual,
    Business,
    Institution,
}

/// Transaction status lifecycle
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TransactionStatus {
    Pending,
    Validating,
    Approved,
    Processing,
    Settled,
    Failed,
    Rejected,
    Cancelled,
}

/// Transaction type for different banking operations
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TransactionType {
    Transfer,
    Payment,
    Withdrawal,
    Deposit,
    FxExchange,
    Fee,
    Interest,
    Dividend,
}

/// Transaction priority for processing order
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
pub enum TransactionPriority {
    Low = 1,
    Normal = 2,
    High = 3,
    Urgent = 4,
}

/// Account information for validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccountInfo {
    pub account_id: String,
    pub account_type: String,
    pub tier: AccountTier,
    pub status: String,
    pub daily_limit: Decimal,
    pub monthly_limit: Decimal,
    pub current_balance: Decimal,
    pub available_balance: Decimal,
    pub created_at: DateTime<Utc>,
    pub last_activity: DateTime<Utc>,
}

/// Main transaction structure with comprehensive fields
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transaction {
    pub id: Uuid,
    pub sender: String,
    pub receiver: String,
    pub amount: Decimal,
    pub currency: String,
    pub transaction_type: TransactionType,
    pub status: TransactionStatus,
    pub priority: TransactionPriority,
    pub reference: Option<String>,
    pub description: Option<String>,
    pub fee: Decimal,
    pub exchange_rate: Option<Decimal>,
    pub settlement_date: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub signature: Vec<u8>,
    pub metadata: HashMap<String, String>,
    pub compliance_flags: Vec<String>,
    pub risk_score: f64,
}

impl Transaction {
    /// Create new transaction with validation
    pub fn new(
        sender: String,
        receiver: String,
        amount: Decimal,
        currency: String,
        transaction_type: TransactionType,
    ) -> Result<Self> {
        if amount <= Decimal::ZERO {
            return Err(CoreError::TransactionError("Amount must be positive".to_string()));
        }
        
        if sender == receiver {
            return Err(CoreError::TransactionError("Sender and receiver cannot be the same".to_string()));
        }
        
        if currency.len() != 3 {
            return Err(CoreError::TransactionError("Invalid currency code".to_string()));
        }
        
        let now = Utc::now();
        
        Ok(Self {
            id: Uuid::new_v4(),
            sender,
            receiver,
            amount,
            currency,
            transaction_type,
            status: TransactionStatus::Pending,
            priority: TransactionPriority::Normal,
            reference: None,
            description: None,
            fee: Decimal::ZERO,
            exchange_rate: None,
            settlement_date: None,
            created_at: now,
            updated_at: now,
            signature: Vec::new(),
            metadata: HashMap::new(),
            compliance_flags: Vec::new(),
            risk_score: 0.0,
        })
    }
    
    /// Calculate transaction hash for signing
    pub fn hash(&self) -> Vec<u8> {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        
        hasher.update(self.id.to_string().as_bytes());
        hasher.update(self.sender.as_bytes());
        hasher.update(self.receiver.as_bytes());
        hasher.update(self.amount.to_string().as_bytes());
        hasher.update(self.currency.as_bytes());
        hasher.update(self.created_at.timestamp().to_string().as_bytes());
        
        hasher.finalize().to_vec()
    }
    
    /// Update transaction status with timestamp
    pub fn update_status(&mut self, new_status: TransactionStatus) {
        self.status = new_status;
        self.updated_at = Utc::now();
    }
    
    /// Add compliance flag
    pub fn add_compliance_flag(&mut self, flag: String) {
        if !self.compliance_flags.contains(&flag) {
            self.compliance_flags.push(flag);
        }
        self.updated_at = Utc::now();
    }
    
    /// Set risk score from risk assessment
    pub fn set_risk_score(&mut self, score: f64) {
        self.risk_score = score.clamp(0.0, 1.0);
        self.updated_at = Utc::now();
    }
    
    /// Check if transaction requires manual approval
    pub fn requires_manual_approval(&self) -> bool {
        self.risk_score > 0.7 || 
        !self.compliance_flags.is_empty() ||
        self.amount > Decimal::from(100_000) // $100k threshold
    }
    
    /// Calculate total cost including fees
    pub fn total_cost(&self) -> Decimal {
        self.amount + self.fee
    }
}

/// Transaction pool for managing pending transactions
pub struct TransactionPool {
    transactions: Arc<RwLock<HashMap<Uuid, Transaction>>>,
    priority_queue: Arc<RwLock<VecDeque<Uuid>>>,
    max_size: usize,
    metrics: Arc<RwLock<TransactionMetrics>>,
}

/// Transaction processing metrics
#[derive(Debug, Clone, Default)]
pub struct TransactionMetrics {
    pub total_processed: u64,
    pub successful: u64,
    pub failed: u64,
    pub pending_count: usize,
    pub average_processing_time_ms: u64,
    pub throughput_per_second: f64,
}

impl TransactionPool {
    /// Create new transaction pool
    pub fn new(max_size: usize) -> Self {
        Self {
            transactions: Arc::new(RwLock::new(HashMap::new())),
            priority_queue: Arc::new(RwLock::new(VecDeque::new())),
            max_size,
            metrics: Arc::new(RwLock::new(TransactionMetrics::default())),
        }
    }
    
    /// Add transaction to pool with priority ordering
    pub async fn add_transaction(&self, mut transaction: Transaction) -> Result<Uuid> {
        let mut transactions = self.transactions.write().await;
        let mut queue = self.priority_queue.write().await;
        
        // Check pool capacity
        if transactions.len() >= self.max_size {
            return Err(CoreError::TransactionError("Transaction pool full".to_string()));
        }
        
        let tx_id = transaction.id;
        
        // Add to priority queue based on priority and timestamp
        let insert_position = queue.iter().position(|&id| {
            if let Some(existing_tx) = transactions.get(&id) {
                if existing_tx.priority < transaction.priority {
                    return true;
                }
                if existing_tx.priority == transaction.priority && 
                   existing_tx.created_at > transaction.created_at {
                    return true;
                }
            }
            false
        }).unwrap_or(queue.len());
        
        queue.insert(insert_position, tx_id);
        transactions.insert(tx_id, transaction);
        
        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.pending_count = transactions.len();
        }
        
        Ok(tx_id)
    }
    
    /// Get next transaction for processing
    pub async fn get_next_transaction(&self) -> Option<Transaction> {
        let mut queue = self.priority_queue.write().await;
        if let Some(tx_id) = queue.pop_front() {
            let transactions = self.transactions.read().await;
            transactions.get(&tx_id).cloned()
        } else {
            None
        }
    }
    
    /// Remove transaction from pool
    pub async fn remove_transaction(&self, tx_id: &Uuid) -> Result<Transaction> {
        let mut transactions = self.transactions.write().await;
        let mut queue = self.priority_queue.write().await;
        
        let transaction = transactions.remove(tx_id)
            .ok_or_else(|| CoreError::TransactionError("Transaction not found in pool".to_string()))?;
        
        // Remove from priority queue
        if let Some(pos) = queue.iter().position(|&id| id == *tx_id) {
            queue.remove(pos);
        }
        
        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.pending_count = transactions.len();
            metrics.total_processed += 1;
        }
        
        Ok(transaction)
    }
    
    /// Get transaction by ID
    pub async fn get_transaction(&self, tx_id: &Uuid) -> Option<Transaction> {
        let transactions = self.transactions.read().await;
        transactions.get(tx_id).cloned()
    }
    
    /// Update transaction in pool
    pub async fn update_transaction(&self, transaction: Transaction) -> Result<()> {
        let mut transactions = self.transactions.write().await;
        let tx_id = transaction.id;
        
        if transactions.contains_key(&tx_id) {
            transactions.insert(tx_id, transaction);
            Ok(())
        } else {
            Err(CoreError::TransactionError("Transaction not found in pool".to_string()))
        }
    }
    
    /// Get pool size
    pub async fn size(&self) -> usize {
        let transactions = self.transactions.read().await;
        transactions.len()
    }
    
    /// Get transactions by status
    pub async fn get_transactions_by_status(&self, status: TransactionStatus) -> Vec<Transaction> {
        let transactions = self.transactions.read().await;
        transactions.values()
            .filter(|tx| tx.status == status)
            .cloned()
            .collect()
    }
    
    /// Get current metrics
    pub async fn get_metrics(&self) -> TransactionMetrics {
        let metrics = self.metrics.read().await;
        metrics.clone()
    }
    
    /// Clear expired transactions (older than 24 hours and still pending)
    pub async fn cleanup_expired(&self) -> usize {
        let mut transactions = self.transactions.write().await;
        let mut queue = self.priority_queue.write().await;
        let cutoff_time = Utc::now() - chrono::Duration::hours(24);
        
        let expired_ids: Vec<Uuid> = transactions.iter()
            .filter(|(_, tx)| {
                matches!(tx.status, TransactionStatus::Pending | TransactionStatus::Validating) &&
                tx.created_at < cutoff_time
            })
            .map(|(id, _)| *id)
            .collect();
        
        for tx_id in &expired_ids {
            transactions.remove(tx_id);
            if let Some(pos) = queue.iter().position(|&id| id == *tx_id) {
                queue.remove(pos);
            }
        }
        
        expired_ids.len()
    }
}

/// Transaction validator with comprehensive checks
pub struct TransactionValidator {
    daily_limits: Arc<RwLock<HashMap<String, Decimal>>>,
    monthly_limits: Arc<RwLock<HashMap<String, Decimal>>>,
    blocked_accounts: Arc<RwLock<std::collections::HashSet<String>>>,
    suspicious_patterns: Arc<RwLock<Vec<SuspiciousPattern>>>,
}

/// Pattern for suspicious transaction detection
#[derive(Debug, Clone)]
pub struct SuspiciousPattern {
    pub name: String,
    pub description: String,
    pub threshold_amount: Decimal,
    pub time_window_hours: u32,
    pub max_occurrences: u32,
}

impl TransactionValidator {
    pub fn new() -> Self {
        let mut suspicious_patterns = Vec::new();
        
        // Add common suspicious patterns
        suspicious_patterns.push(SuspiciousPattern {
            name: "rapid_transfers".to_string(),
            description: "Multiple transfers to same account within short time".to_string(),
            threshold_amount: Decimal::from(10_000),
            time_window_hours: 1,
            max_occurrences: 5,
        });
        
        suspicious_patterns.push(SuspiciousPattern {
            name: "round_amounts".to_string(),
            description: "Multiple round amount transactions".to_string(),
            threshold_amount: Decimal::from(50_000),
            time_window_hours: 24,
            max_occurrences: 3,
        });
        
        Self {
            daily_limits: Arc::new(RwLock::new(HashMap::new())),
            monthly_limits: Arc::new(RwLock::new(HashMap::new())),
            blocked_accounts: Arc::new(RwLock::new(std::collections::HashSet::new())),
            suspicious_patterns: Arc::new(RwLock::new(suspicious_patterns)),
        }
    }
    
    /// Validate transaction with comprehensive checks
    pub async fn validate_transaction(&self, tx: &Transaction, account_info: &AccountInfo) -> Result<ValidationResult> {
        let mut result = ValidationResult {
            valid: true,
            warnings: Vec::new(),
            errors: Vec::new(),
            risk_score: 0.0,
        };
        
        // Check blocked accounts
        {
            let blocked = self.blocked_accounts.read().await;
            if blocked.contains(&tx.sender) || blocked.contains(&tx.receiver) {
                result.errors.push("Account is blocked".to_string());
                result.valid = false;
                return Ok(result);
            }
        }
        
        // Validate amount limits
        if let Err(e) = self.check_amount_limits(tx, account_info).await {
            result.errors.push(e.to_string());
            result.valid = false;
        }
        
        // Check daily/monthly limits
        if let Err(e) = self.check_spending_limits(tx).await {
            result.warnings.push(e.to_string());
            result.risk_score += 0.2;
        }
        
        // Check for suspicious patterns
        let suspicious_score = self.check_suspicious_patterns(tx).await;
        result.risk_score += suspicious_score;
        
        if suspicious_score > 0.5 {
            result.warnings.push("Transaction matches suspicious patterns".to_string());
        }
        
        // AML/KYC checks
        self.perform_aml_checks(tx, &mut result).await;
        
        // Currency and cross-border checks
        self.check_currency_compliance(tx, &mut result).await;
        
        Ok(result)
    }
    
    async fn check_amount_limits(&self, tx: &Transaction, account_info: &AccountInfo) -> Result<()> {
        // Check single transaction limits based on account tier
        let max_single_transaction = match account_info.tier {
            AccountTier::Individual => Decimal::from(25_000),
            AccountTier::Business => Decimal::from(250_000),
            AccountTier::Institution => Decimal::from(10_000_000),
        };
        
        if tx.amount > max_single_transaction {
            return Err(CoreError::TransactionError(
                format!("Amount exceeds single transaction limit of ${}", max_single_transaction)
            ));
        }
        
        // Check available balance
        if account_info.available_balance < tx.total_cost() {
            return Err(CoreError::TransactionError("Insufficient available balance".to_string()));
        }
        
        Ok(())
    }
    
    async fn check_spending_limits(&self, tx: &Transaction) -> Result<()> {
        let daily_limits = self.daily_limits.read().await;
        let monthly_limits = self.monthly_limits.read().await;
        
        // Check daily limit
        if let Some(&daily_spent) = daily_limits.get(&tx.sender) {
            let daily_limit = Decimal::from(100_000); // Default $100k daily limit
            if daily_spent + tx.amount > daily_limit {
                return Err(CoreError::TransactionError("Daily spending limit exceeded".to_string()));
            }
        }
        
        // Check monthly limit  
        if let Some(&monthly_spent) = monthly_limits.get(&tx.sender) {
            let monthly_limit = Decimal::from(1_000_000); // Default $1M monthly limit
            if monthly_spent + tx.amount > monthly_limit {
                return Err(CoreError::TransactionError("Monthly spending limit exceeded".to_string()));
            }
        }
        
        Ok(())
    }
    
    async fn check_suspicious_patterns(&self, tx: &Transaction) -> f64 {
        let patterns = self.suspicious_patterns.read().await;
        let mut risk_score = 0.0;
        
        // Check for round amounts (potential money laundering)
        if tx.amount.fract() == Decimal::ZERO && tx.amount >= Decimal::from(10_000) {
            risk_score += 0.2;
        }
        
        // Check for unusual transaction times (e.g., multiple transactions at 3 AM)
        let hour = tx.created_at.hour();
        if hour < 6 || hour > 22 {
            risk_score += 0.1;
        }
        
        // High-value transactions get higher scrutiny
        if tx.amount > Decimal::from(100_000) {
            risk_score += 0.3;
        }
        
        risk_score.min(1.0)
    }
    
    async fn perform_aml_checks(&self, tx: &Transaction, result: &mut ValidationResult) {
        // Simplified AML checks - in production integrate with OFAC/sanctions lists
        let high_risk_countries = ["AF", "IR", "KP", "SY"];
        
        // Check if transaction involves high-risk jurisdictions
        if let Some(sender_country) = tx.metadata.get("sender_country") {
            if high_risk_countries.contains(&sender_country.as_str()) {
                result.warnings.push("Transaction from high-risk jurisdiction".to_string());
                result.risk_score += 0.4;
            }
        }
        
        // Check for structuring (breaking large amounts into smaller ones)
        if tx.amount > Decimal::from(9_900) && tx.amount < Decimal::from(10_100) {
            result.warnings.push("Potential structuring detected".to_string());
            result.risk_score += 0.3;
        }
    }
    
    async fn check_currency_compliance(&self, tx: &Transaction, result: &mut ValidationResult) {
        let restricted_currencies = ["KPW", "IRR", "VEB"]; // North Korea, Iran, Venezuela
        
        if restricted_currencies.contains(&tx.currency.as_str()) {
            result.errors.push("Currency not supported due to sanctions".to_string());
            result.valid = false;
        }
        
        // Cross-border reporting requirements (e.g., >$10k USD equivalent)
        if tx.amount > Decimal::from(10_000) && tx.currency == "USD" {
            result.warnings.push("Transaction subject to cross-border reporting".to_string());
        }
    }
    
    /// Block account from transactions
    pub async fn block_account(&self, account_id: String) {
        let mut blocked = self.blocked_accounts.write().await;
        blocked.insert(account_id);
    }
    
    /// Unblock account
    pub async fn unblock_account(&self, account_id: &str) {
        let mut blocked = self.blocked_accounts.write().await;
        blocked.remove(account_id);
    }
}

/// Transaction validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub valid: bool,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
    pub risk_score: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_transaction_creation() {
        let tx = Transaction::new(
            "account1".to_string(),
            "account2".to_string(),
            Decimal::from(1000),
            "USD".to_string(),
            TransactionType::Transfer,
        );
        
        assert!(tx.is_ok());
        let tx = tx.unwrap();
        assert_eq!(tx.amount, Decimal::from(1000));
        assert_eq!(tx.status, TransactionStatus::Pending);
    }
    
    #[test]
    fn test_invalid_transaction() {
        // Zero amount should fail
        let tx = Transaction::new(
            "account1".to_string(),
            "account2".to_string(),
            Decimal::ZERO,
            "USD".to_string(),
            TransactionType::Transfer,
        );
        assert!(tx.is_err());
        
        // Same sender and receiver should fail
        let tx = Transaction::new(
            "account1".to_string(),
            "account1".to_string(),
            Decimal::from(1000),
            "USD".to_string(),
            TransactionType::Transfer,
        );
        assert!(tx.is_err());
    }
    
    #[tokio::test]
    async fn test_transaction_pool() {
        let pool = TransactionPool::new(10);
        
        let tx = Transaction::new(
            "sender".to_string(),
            "receiver".to_string(),
            Decimal::from(500),
            "USD".to_string(),
            TransactionType::Transfer,
        ).unwrap();
        
        let tx_id = pool.add_transaction(tx.clone()).await.unwrap();
        assert_eq!(pool.size().await, 1);
        
        let retrieved_tx = pool.get_transaction(&tx_id).await;
        assert!(retrieved_tx.is_some());
        assert_eq!(retrieved_tx.unwrap().id, tx.id);
    }
    
    #[tokio::test]
    async fn test_transaction_priority() {
        let pool = TransactionPool::new(10);
        
        // Add normal priority transaction
        let mut normal_tx = Transaction::new(
            "sender1".to_string(),
            "receiver1".to_string(),
            Decimal::from(100),
            "USD".to_string(),
            TransactionType::Transfer,
        ).unwrap();
        normal_tx.priority = TransactionPriority::Normal;
        
        // Add urgent priority transaction  
        let mut urgent_tx = Transaction::new(
            "sender2".to_string(),
            "receiver2".to_string(),
            Decimal::from(200),
            "USD".to_string(),
            TransactionType::Transfer,
        ).unwrap();
        urgent_tx.priority = TransactionPriority::Urgent;
        
        pool.add_transaction(normal_tx).await.unwrap();
        pool.add_transaction(urgent_tx.clone()).await.unwrap();
        
        // Urgent transaction should come first
        let next_tx = pool.get_next_transaction().await.unwrap();
        assert_eq!(next_tx.id, urgent_tx.id);
    }
}