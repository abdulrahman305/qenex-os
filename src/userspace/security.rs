//! Fixed Security Module for Userspace
//! Addresses the original compilation errors

use crate::types::*;
use crate::error::*;
use chrono::{DateTime, Utc, Timelike};
use rust_decimal::Decimal;
use std::collections::HashMap;

pub struct SecurityEngine {
    risk_thresholds: HashMap<String, f64>,
}

impl SecurityEngine {
    pub fn new() -> Self {
        let mut thresholds = HashMap::new();
        thresholds.insert("high_value".to_string(), 0.8);
        thresholds.insert("time_based".to_string(), 0.6);
        thresholds.insert("recipient".to_string(), 0.7);

        Self {
            risk_thresholds: thresholds,
        }
    }

    pub async fn assess_transaction_risk(&self, transaction: &Transaction) -> Result<f64, QenexError> {
        let mut risk_score = 0.0f64;

        // Fix 1: Convert Decimal to u64 for struct compatibility
        let risk_transaction = RiskTransaction {
            id: transaction.core.id.clone(),
            sender: transaction.core.sender.clone(),
            receiver: transaction.core.receiver.clone(),
            // FIXED: Convert Decimal to u64
            amount: transaction.core.amount.value.to_u64().unwrap_or(0),
            currency: transaction.core.amount.currency.clone(),
            created_at: transaction.created_at,
        };

        // Amount-based risk
        risk_score += self.assess_amount_risk(&risk_transaction).await;

        // Fix 2: Use created_at instead of timestamp (which doesn't exist)
        let time_risk_score = self.assess_time_risk(transaction.created_at).await;
        risk_score += time_risk_score;

        // Fix 3: Use receiver instead of recipient (which doesn't exist)
        let recipient_risk = self.assess_recipient(&transaction.core.receiver).await.unwrap_or(0.0);
        risk_score += recipient_risk;

        // Ensure risk score is between 0 and 1
        Ok(risk_score.min(1.0f64))
    }

    async fn assess_amount_risk(&self, transaction: &RiskTransaction) -> f64 {
        // Convert amount to f64 for risk calculation
        let amount_f64 = transaction.amount as f64;

        // High amount = higher risk
        let normalized_amount = (amount_f64 / 1_000_000.0).min(1.0);
        normalized_amount * 0.3 // 30% weight for amount risk
    }

    // Fix 2: Use DateTime<Utc> parameter name that exists in Transaction
    async fn assess_time_risk(&self, created_at: DateTime<Utc>) -> f64 {
        // Fix: Add missing import and use it properly
        let hour = created_at.hour();
        let minute = created_at.minute();

        // Higher risk during off-hours
        let risk_multiplier = if hour < 6 || hour > 22 {
            0.8 // Higher risk during night hours
        } else if hour >= 9 && hour <= 17 {
            0.2 // Lower risk during business hours
        } else {
            0.5 // Medium risk during evening/morning
        };

        risk_multiplier * 0.2 // 20% weight for time-based risk
    }

    // Fix 3: Use AccountId parameter that exists in Transaction
    async fn assess_recipient(&self, receiver: &AccountId) -> Result<f64, QenexError> {
        // Simple recipient risk assessment
        // In practice, this would check against blacklists, etc.

        // For demo, use a simple hash-based risk
        let receiver_bytes = &receiver.0;
        let risk_factor = (receiver_bytes[0] as f64) / 255.0;

        Ok(risk_factor * 0.3) // 30% weight for recipient risk
    }
}

impl Default for SecurityEngine {
    fn default() -> Self {
        Self::new()
    }
}

// Helper struct for risk assessment with compatible types
#[derive(Debug, Clone)]
struct RiskTransaction {
    id: TransactionId,
    sender: AccountId,
    receiver: AccountId,
    amount: u64,  // Converted from Decimal for compatibility
    currency: Currency,
    created_at: DateTime<Utc>,
}

// Additional security functions
pub async fn validate_transaction_signature(
    transaction: &Transaction,
    public_key: &[u8]
) -> Result<bool, QenexError> {
    // Placeholder signature validation
    if let Some(signature) = &transaction.core.signature {
        // In practice, use proper cryptographic verification
        Ok(signature.len() == 64 && public_key.len() >= 32)
    } else {
        Ok(false)
    }
}

pub fn calculate_transaction_fee(amount: &Amount) -> Amount {
    // Simple fee calculation: 0.1% of amount
    let fee_rate = Decimal::new(1, 3); // 0.001 = 0.1%
    let fee_value = amount.value * fee_rate;

    Amount {
        value: fee_value,
        currency: amount.currency.clone(),
    }
}