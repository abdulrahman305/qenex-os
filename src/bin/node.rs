//! QENEX Banking Node - Main Application Entry Point
//! 
//! Production banking node with complete transaction processing,
//! consensus participation, and network communication

use qenex_os::{BankingCore, SystemConfig, init_telemetry};
use clap::Parser;
use std::path::PathBuf;
use tracing::{info, error};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Configuration file path
    #[arg(short, long, default_value = "/etc/qenex/config.toml")]
    config: PathBuf,
    
    /// Node ID (overrides config)
    #[arg(long)]
    node_id: Option<String>,
    
    /// Network port (overrides config)  
    #[arg(short, long)]
    port: Option<u16>,
    
    /// Database URL (overrides config)
    #[arg(short, long)]
    database_url: Option<String>,
    
    /// Log level
    #[arg(long, default_value = "info")]
    log_level: String,
    
    /// Run in daemon mode
    #[arg(short, long)]
    daemon: bool,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    
    // Initialize telemetry
    std::env::set_var("RUST_LOG", &args.log_level);
    init_telemetry()?;
    
    info!("Starting QENEX Banking Node v{}", qenex_os::VERSION);
    
    // Load configuration
    let mut config = load_config(&args.config).unwrap_or_default();
    
    // Apply command line overrides
    if let Some(node_id) = &args.node_id {
        config.node_id = node_id.parse()
            .map_err(|e| format!("Invalid node ID: {}", e))?;
    }
    
    if let Some(port) = args.port {
        config.network_port = port;
    }
    
    if let Some(database_url) = &args.database_url {
        config.database_url = database_url.clone();
    }
    
    config.log_level = args.log_level;
    
    info!("Configuration loaded for node {}", config.node_id);
    info!("Network port: {}", config.network_port);
    info!("Database: {}", mask_database_url(&config.database_url));
    
    // Initialize banking core
    let banking_core = match BankingCore::new(config).await {
        Ok(core) => core,
        Err(e) => {
            error!("Failed to initialize banking core: {}", e);
            return Err(e.into());
        }
    };
    
    // Setup signal handling for graceful shutdown
    let shutdown_signal = setup_shutdown_signal();
    
    // Start the banking core
    if let Err(e) = banking_core.start().await {
        error!("Failed to start banking core: {}", e);
        return Err(e.into());
    }
    
    info!("QENEX Banking Node started successfully");
    
    // If daemon mode, detach from terminal
    if args.daemon {
        daemonize()?;
    }
    
    // Wait for shutdown signal
    shutdown_signal.await;
    
    info!("Received shutdown signal, stopping node...");
    
    // Graceful shutdown
    if let Err(e) = banking_core.shutdown().await {
        error!("Error during shutdown: {}", e);
    }
    
    info!("QENEX Banking Node stopped");
    Ok(())
}

/// Load configuration from file
fn load_config(config_path: &PathBuf) -> Result<SystemConfig, Box<dyn std::error::Error>> {
    if !config_path.exists() {
        info!("Configuration file not found: {:?}, using defaults", config_path);
        return Ok(SystemConfig::default());
    }
    
    let config_str = std::fs::read_to_string(config_path)?;
    let config: SystemConfig = toml::from_str(&config_str)
        .map_err(|e| format!("Failed to parse configuration: {}", e))?;
    
    Ok(config)
}

/// Mask sensitive information in database URL for logging
fn mask_database_url(url: &str) -> String {
    if let Ok(parsed) = url::Url::parse(url) {
        let mut masked = parsed.clone();
        if masked.password().is_some() {
            let _ = masked.set_password(Some("****"));
        }
        masked.to_string()
    } else {
        "invalid_url".to_string()
    }
}

/// Setup signal handling for graceful shutdown
async fn setup_shutdown_signal() {
    use tokio::signal;
    
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };
    
    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };
    
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    
    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}

/// Daemonize the process (Unix only)
#[cfg(unix)]
fn daemonize() -> Result<(), Box<dyn std::error::Error>> {
    use std::process;
    
    // Fork the process
    let pid = unsafe { libc::fork() };
    
    match pid {
        -1 => return Err("Failed to fork process".into()),
        0 => {
            // Child process continues
            
            // Create new session
            if unsafe { libc::setsid() } == -1 {
                return Err("Failed to create new session".into());
            }
            
            // Fork again to ensure we can't acquire a controlling terminal
            let pid = unsafe { libc::fork() };
            match pid {
                -1 => return Err("Failed to fork process again".into()),
                0 => {
                    // Grandchild process continues as daemon
                    
                    // Change working directory to root
                    std::env::set_current_dir("/")?;
                    
                    // Close standard file descriptors
                    unsafe {
                        libc::close(0);
                        libc::close(1);
                        libc::close(2);
                    }
                    
                    // Redirect to /dev/null
                    use std::os::unix::io::AsRawFd;
                    let dev_null = std::fs::File::open("/dev/null")?;
                    let fd = dev_null.as_raw_fd();
                    unsafe {
                        libc::dup2(fd, 0);
                        libc::dup2(fd, 1);
                        libc::dup2(fd, 2);
                    }
                    
                    info!("Daemonized successfully");
                    Ok(())
                }
                _ => {
                    // Second parent exits
                    process::exit(0);
                }
            }
        }
        _ => {
            // First parent exits
            process::exit(0);
        }
    }
}

#[cfg(not(unix))]
fn daemonize() -> Result<(), Box<dyn std::error::Error>> {
    Err("Daemon mode not supported on this platform".into())
}