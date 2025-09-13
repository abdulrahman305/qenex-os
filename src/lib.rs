//! QENEX Financial Operating System
//!
//! A modular financial OS with kernel and userspace components

#![cfg_attr(feature = "kernel", no_std)]
#![cfg_attr(feature = "kernel", no_main)]

// Core modules (always available)
pub mod types;
pub mod error;

// Conditional compilation based on features
#[cfg(feature = "kernel")]
pub mod kernel;

#[cfg(feature = "userspace")]
pub mod userspace;

#[cfg(feature = "networking")]
pub mod network;

#[cfg(feature = "database")]
pub mod storage;

#[cfg(feature = "crypto")]
pub mod crypto;

#[cfg(feature = "consensus")]
pub mod consensus;

// Re-export common types
pub use types::*;
pub use error::*;

// Conditional re-exports
#[cfg(feature = "userspace")]
pub use userspace::*;

#[cfg(feature = "kernel")]
pub use kernel::*;