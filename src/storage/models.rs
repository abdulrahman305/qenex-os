//! Data Models Module
//! 
//! Core data structures and models for banking operations
//! with serialization and validation support.

#![cfg_attr(not(feature = "std"), no_std)]

use serde::{Deserialize, Serialize};
use uuid::Uuid;
use rust_decimal::Decimal;

/// Account data model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Account {
    pub id: Uuid,
    pub account_number: String,
    pub account_type: AccountType,
    pub balance: Decimal,
    pub currency: String,
    pub created_at: u64,
    pub updated_at: u64,
    pub status: AccountStatus,
}

/// Account type enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AccountType {
    Checking,
    Savings,
    Credit,
    Investment,
    Business,
}

/// Account status enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AccountStatus {
    Active,
    Suspended,
    Closed,
    PendingActivation,
}

/// Transaction data model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transaction {
    pub id: Uuid,
    pub from_account: Uuid,
    pub to_account: Uuid,
    pub amount: Decimal,
    pub currency: String,
    pub transaction_type: TransactionType,
    pub status: TransactionStatus,
    pub created_at: u64,
    pub processed_at: Option<u64>,
    pub description: Option<String>,
}

/// Transaction type enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransactionType {
    Transfer,
    Deposit,
    Withdrawal,
    Payment,
    Fee,
}

/// Transaction status enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransactionStatus {
    Pending,
    Processing,
    Completed,
    Failed,
    Cancelled,
}

/// User data model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub id: Uuid,
    pub email: String,
    pub username: String,
    pub first_name: String,
    pub last_name: String,
    pub created_at: u64,
    pub last_login: Option<u64>,
    pub status: UserStatus,
}

/// User status enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UserStatus {
    Active,
    Inactive,
    Suspended,
    PendingVerification,
}

impl Default for Account {
    fn default() -> Self {
        Self {
            id: Uuid::new_v4(),
            account_number: String::new(),
            account_type: AccountType::Checking,
            balance: Decimal::ZERO,
            currency: "USD".to_string(),
            created_at: 0,
            updated_at: 0,
            status: AccountStatus::PendingActivation,
        }
    }
}

impl Default for Transaction {
    fn default() -> Self {
        Self {
            id: Uuid::new_v4(),
            from_account: Uuid::new_v4(),
            to_account: Uuid::new_v4(),
            amount: Decimal::ZERO,
            currency: "USD".to_string(),
            transaction_type: TransactionType::Transfer,
            status: TransactionStatus::Pending,
            created_at: 0,
            processed_at: None,
            description: None,
        }
    }
}

impl Default for User {
    fn default() -> Self {
        Self {
            id: Uuid::new_v4(),
            email: String::new(),
            username: String::new(),
            first_name: String::new(),
            last_name: String::new(),
            created_at: 0,
            last_login: None,
            status: UserStatus::PendingVerification,
        }
    }
}