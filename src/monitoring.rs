//! Monitoring and Metrics Collection Module
//! 
//! Comprehensive monitoring system for banking operations
//! with real-time metrics, alerting, and performance tracking.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "std")]
use std::collections::HashMap;
#[cfg(not(feature = "std"))]
use heapless::FnvIndexMap as HashMap;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Metrics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    pub collection_interval_seconds: u64,
    pub retention_days: u32,
    pub alert_thresholds: HashMap<String, f64>,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        let mut thresholds = HashMap::new();
        #[cfg(feature = "std")]
        {
            thresholds.insert("cpu_usage".to_string(), 80.0);
            thresholds.insert("memory_usage".to_string(), 85.0);
            thresholds.insert("transaction_latency_ms".to_string(), 100.0);
        }
        
        Self {
            collection_interval_seconds: 60,
            retention_days: 30,
            alert_thresholds: thresholds,
        }
    }
}

/// System metric data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricData {
    pub id: Uuid,
    pub metric_name: String,
    pub value: f64,
    pub timestamp: u64,
    pub tags: HashMap<String, String>,
}

/// Metrics collector for banking operations
pub struct MetricsCollector {
    config: MonitoringConfig,
    metrics: HashMap<String, Vec<MetricData>>,
}

impl MetricsCollector {
    pub fn new(config: MonitoringConfig) -> Self {
        Self {
            config,
            metrics: HashMap::new(),
        }
    }

    pub fn record_metric(&mut self, name: String, value: f64) {
        let metric = MetricData {
            id: Uuid::new_v4(),
            metric_name: name.clone(),
            value,
            timestamp: self.get_current_timestamp(),
            tags: HashMap::new(),
        };

        #[cfg(feature = "std")]
        {
            self.metrics.entry(name).or_insert_with(Vec::new).push(metric);
        }
    }

    pub fn get_metric_count(&self) -> usize {
        self.metrics.len()
    }

    fn get_current_timestamp(&self) -> u64 {
        #[cfg(feature = "std")]
        {
            use std::time::{SystemTime, UNIX_EPOCH};
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs()
        }
        #[cfg(not(feature = "std"))]
        {
            0 // Placeholder for no_std environments
        }
    }

    #[cfg(feature = "std")]
    pub async fn start_collection(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        tracing::info!("Starting metrics collection");
        Ok(())
    }

    #[cfg(feature = "std")]
    pub async fn stop_collection(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        tracing::info!("Stopping metrics collection");
        Ok(())
    }
}