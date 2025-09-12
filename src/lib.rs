//! QENEX Banking Operating System
//!
//! Production-ready operating system kernel and userspace components designed
//! specifically for financial institutions and banking operations.
//!
//! This system provides:
//! - Bootable kernel with real hardware support
//! - Post-quantum cryptographic security
//! - ACID-compliant distributed transactions
//! - Real-time regulatory compliance
//! - Self-improving AI/ML fraud detection
//! - High availability with Byzantine fault tolerance

#![cfg_attr(feature = "kernel", no_std)]
#![cfg_attr(feature = "kernel", no_main)]
#![cfg_attr(feature = "kernel", feature(lang_items))]
#![cfg_attr(feature = "kernel", feature(panic_info_message))]

#[cfg(feature = "std")]
extern crate std;

pub mod core;
pub mod network;
pub mod storage;
pub mod api;
pub mod consensus;
pub mod compliance;
pub mod monitoring;
pub mod ai;
pub mod protocols;
pub mod cluster;
pub mod testing;
pub mod kernel;
pub mod crypto;
pub mod transaction;

#[cfg(feature = "std")]
pub use core::{BankingCore, SystemConfig, SystemHealth, CoreError, Result};

// Kernel-specific exports
#[cfg(feature = "kernel")]
pub use kernel::{KernelMain, KernelPanic};

// Kernel panic handler
#[cfg(feature = "kernel")]
#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    use kernel::KernelPanic;
    
    if let Some(message) = info.message() {
        if let Some(args) = message.as_str() {
            KernelPanic::panic_with_message(args);
        } else {
            KernelPanic::panic_with_message("Kernel panic occurred");
        }
    } else {
        KernelPanic::panic_with_message("Unknown kernel panic");
    }
}

// Language items for no_std kernel
#[cfg(feature = "kernel")]
#[lang = "eh_personality"]
extern "C" fn eh_personality() {}

// Entry point for kernel mode
#[cfg(feature = "kernel")]
#[no_mangle]
pub extern "C" fn kernel_main_rust(magic: u32, multiboot_info: *const u8) {
    kernel::initialize_kernel(magic, multiboot_info as *const core::ffi::c_void);
}

/// Re-export commonly used types
pub use core::transaction::{Transaction, TransactionType, TransactionStatus, AccountTier};
pub use core::crypto::{CryptoProvider, Signature, KeyType};

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const BUILD_INFO: &str = concat!(
    env!("CARGO_PKG_VERSION"), 
    " (unknown)"
);

/// Initialize logging and monitoring
#[cfg(feature = "std")]
pub fn init_telemetry() -> Result<(), CoreError> {
    use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
    
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "qenex_os=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();
    
    // Initialize metrics
    let builder = metrics_exporter_prometheus::PrometheusBuilder::new();
    builder
        .install()
        .map_err(|e| CoreError::NetworkError(format!("Failed to install metrics: {}", e)))?;
    
    Ok(())
}