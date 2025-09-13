//! Error types that work in both kernel and userspace

#![cfg_attr(feature = "kernel", no_std)]

use core::fmt;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QenexError {
    // Transaction errors
    InvalidAmount,
    InsufficientFunds,
    TransactionNotFound,
    InvalidTransaction,

    // Account errors
    AccountNotFound,
    AccountBlocked,
    InvalidAccount,

    // Security errors
    InvalidSignature,
    AuthenticationFailed,
    AuthorizationFailed,

    // System errors
    SystemError,
    DatabaseError,
    NetworkError,

    // Kernel-specific errors
    #[cfg(feature = "kernel")]
    MemoryAllocationError,
    #[cfg(feature = "kernel")]
    HardwareError,

    // Userspace-specific errors
    #[cfg(feature = "userspace")]
    IoError,
    #[cfg(feature = "userspace")]
    SerializationError,
}

impl fmt::Display for QenexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QenexError::InvalidAmount => write!(f, "Invalid transaction amount"),
            QenexError::InsufficientFunds => write!(f, "Insufficient funds"),
            QenexError::TransactionNotFound => write!(f, "Transaction not found"),
            QenexError::InvalidTransaction => write!(f, "Invalid transaction"),
            QenexError::AccountNotFound => write!(f, "Account not found"),
            QenexError::AccountBlocked => write!(f, "Account is blocked"),
            QenexError::InvalidAccount => write!(f, "Invalid account"),
            QenexError::InvalidSignature => write!(f, "Invalid signature"),
            QenexError::AuthenticationFailed => write!(f, "Authentication failed"),
            QenexError::AuthorizationFailed => write!(f, "Authorization failed"),
            QenexError::SystemError => write!(f, "System error"),
            QenexError::DatabaseError => write!(f, "Database error"),
            QenexError::NetworkError => write!(f, "Network error"),

            #[cfg(feature = "kernel")]
            QenexError::MemoryAllocationError => write!(f, "Memory allocation error"),
            #[cfg(feature = "kernel")]
            QenexError::HardwareError => write!(f, "Hardware error"),

            #[cfg(feature = "userspace")]
            QenexError::IoError => write!(f, "I/O error"),
            #[cfg(feature = "userspace")]
            QenexError::SerializationError => write!(f, "Serialization error"),
        }
    }
}

#[cfg(feature = "userspace")]
impl std::error::Error for QenexError {}

// Result type alias
pub type QenexResult<T> = Result<T, QenexError>;