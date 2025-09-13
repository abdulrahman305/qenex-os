//! Common types used across kernel and userspace

#![cfg_attr(feature = "kernel", no_std)]

use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

#[cfg(feature = "userspace")]
use chrono::{DateTime, Utc};

// Core financial types that work in both kernel and userspace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionId(pub [u8; 32]);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccountId(pub [u8; 32]);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Amount {
    pub value: Decimal,
    pub currency: Currency,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Currency {
    USD,
    EUR,
    GBP,
    BTC,
    ETH,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransactionStatus {
    Pending,
    Processing,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransactionType {
    Transfer,
    Payment,
    Withdrawal,
    Deposit,
    Exchange,
}

// Core transaction structure (works in both kernel and userspace)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreTransaction {
    pub id: TransactionId,
    pub sender: AccountId,
    pub receiver: AccountId,
    pub amount: Amount,
    pub transaction_type: TransactionType,
    pub status: TransactionStatus,
    pub signature: Option<[u8; 64]>,
}

impl CoreTransaction {
    pub fn hash(&self) -> [u8; 32] {
        // Simple hash implementation that works in no_std
        use core::hash::{Hash, Hasher};

        struct SimpleHasher {
            state: u64,
        }

        impl Hasher for SimpleHasher {
            fn write(&mut self, bytes: &[u8]) {
                for &byte in bytes {
                    self.state = self.state.wrapping_mul(31).wrapping_add(byte as u64);
                }
            }

            fn finish(&self) -> u64 {
                self.state
            }
        }

        let mut hasher = SimpleHasher { state: 0 };
        self.id.0.hash(&mut hasher);

        let hash_value = hasher.finish();
        let mut result = [0u8; 32];
        result[..8].copy_from_slice(&hash_value.to_le_bytes());
        result
    }

    pub fn amount_as_u64(&self) -> u64 {
        self.amount.value.to_u64().unwrap_or(0)
    }
}

// Userspace-specific transaction with timestamps
#[cfg(feature = "userspace")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transaction {
    #[serde(flatten)]
    pub core: CoreTransaction,

    // Userspace-specific fields
    pub created_at: DateTime<Utc>,
    pub updated_at: Option<DateTime<Utc>>,
    pub reference: Option<String>,
    pub description: Option<String>,
    pub fee: Option<Amount>,
}

#[cfg(feature = "userspace")]
impl Transaction {
    pub fn from_core(core: CoreTransaction) -> Self {
        Self {
            core,
            created_at: Utc::now(),
            updated_at: None,
            reference: None,
            description: None,
            fee: None,
        }
    }

    // Compatibility methods for existing code
    pub fn id(&self) -> &TransactionId { &self.core.id }
    pub fn sender(&self) -> &AccountId { &self.core.sender }
    pub fn receiver(&self) -> &AccountId { &self.core.receiver }
    pub fn amount(&self) -> &Amount { &self.core.amount }
    pub fn status(&self) -> &TransactionStatus { &self.core.status }
}

// Account structures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Account {
    pub id: AccountId,
    pub account_type: AccountType,
    pub status: AccountStatus,
    pub balances: Vec<Amount>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AccountType {
    Checking,
    Savings,
    Business,
    Treasury,
    Escrow,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AccountStatus {
    Active,
    Suspended,
    Closed,
    Frozen,
}