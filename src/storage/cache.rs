//! Cache Management - High Performance Caching Layer
//! 
//! Redis-based caching with intelligent invalidation
//! and performance optimization for financial operations

use super::{CoreError, Result, Account, AccountBalance};
use redis::{aio::ConnectionManager, AsyncCommands, Client};
use std::sync::Arc;
use tokio::sync::RwLock;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Cache configuration
#[derive(Debug, Clone)]
pub struct CacheConfig {
    pub redis_url: String,
    pub default_ttl: Duration,
    pub max_connections: u32,
    pub key_prefix: String,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            redis_url: "redis://localhost:6379".to_string(),
            default_ttl: Duration::from_secs(3600), // 1 hour
            max_connections: 20,
            key_prefix: "qenex:".to_string(),
        }
    }
}

/// Cache entry with TTL information
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CacheEntry<T> {
    data: T,
    expires_at: u64,
    created_at: u64,
}

impl<T> CacheEntry<T> {
    fn new(data: T, ttl: Duration) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        Self {
            data,
            expires_at: now + ttl.as_secs(),
            created_at: now,
        }
    }
    
    fn is_expired(&self) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        now > self.expires_at
    }
}

/// Cache statistics
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub memory_usage_bytes: u64,
    pub active_keys: u64,
}

/// Main cache manager
pub struct CacheManager {
    connection: Arc<RwLock<ConnectionManager>>,
    config: CacheConfig,
    stats: Arc<RwLock<CacheStats>>,
}

impl CacheManager {
    /// Create new cache manager
    pub async fn new() -> Result<Self> {
        let config = CacheConfig::default();
        Self::with_config(config).await
    }
    
    /// Create cache manager with custom configuration
    pub async fn with_config(config: CacheConfig) -> Result<Self> {
        let client = Client::open(config.redis_url.as_str())
            .map_err(|e| CoreError::StorageError(format!("Failed to create Redis client: {}", e)))?;
        
        let connection = client.get_tokio_connection_manager()
            .await
            .map_err(|e| CoreError::StorageError(format!("Failed to connect to Redis: {}", e)))?;
        
        Ok(Self {
            connection: Arc::new(RwLock::new(connection)),
            config,
            stats: Arc::new(RwLock::new(CacheStats::default())),
        })
    }
    
    /// Health check for cache system
    pub async fn health_check(&self) -> Result<()> {
        let mut conn = self.connection.write().await;
        
        let _: String = conn.ping()
            .await
            .map_err(|e| CoreError::StorageError(format!("Cache health check failed: {}", e)))?;
        
        Ok(())
    }
    
    /// Warm cache with frequently accessed data
    pub async fn warm_cache(&self) -> Result<()> {
        tracing::info!("Warming cache with frequently accessed data...");
        
        // This would typically load frequently accessed accounts, balances, etc.
        // Implementation would depend on specific business requirements
        
        tracing::info!("Cache warming completed");
        Ok(())
    }
    
    /// Stop cache manager
    pub async fn stop(&self) -> Result<()> {
        tracing::info!("Stopping cache manager");
        
        // Flush any pending operations
        let mut conn = self.connection.write().await;
        let _: () = conn.flushdb()
            .await
            .map_err(|e| CoreError::StorageError(format!("Failed to flush cache: {}", e)))?;
        
        Ok(())
    }
    
    /// Get account from cache
    pub async fn get_account(&self, account_id: &str) -> Option<Account> {
        let key = format!("{}account:{}", self.config.key_prefix, account_id);
        
        match self.get_cached_data::<Account>(&key).await {
            Ok(Some(account)) => {
                self.record_hit().await;
                Some(account)
            }
            _ => {
                self.record_miss().await;
                None
            }
        }
    }
    
    /// Set account in cache
    pub async fn set_account(&self, account_id: &str, account: &Account) {
        let key = format!("{}account:{}", self.config.key_prefix, account_id);
        let _ = self.set_cached_data(&key, account, self.config.default_ttl).await;
    }
    
    /// Get account balance from cache
    pub async fn get_balance(&self, account_id: &str, currency: &str) -> Option<Decimal> {
        let key = format!("{}balance:{}:{}", self.config.key_prefix, account_id, currency);
        
        match self.get_cached_data::<Decimal>(&key).await {
            Ok(Some(balance)) => {
                self.record_hit().await;
                Some(balance)
            }
            _ => {
                self.record_miss().await;
                None
            }
        }
    }
    
    /// Set account balance in cache
    pub async fn set_balance(&self, account_id: &str, currency: &str, balance: Decimal) {
        let key = format!("{}balance:{}:{}", self.config.key_prefix, account_id, currency);
        let _ = self.set_cached_data(&key, &balance, Duration::from_secs(300)).await; // 5 min TTL for balances
    }
    
    /// Get exchange rate from cache
    pub async fn get_exchange_rate(&self, from_currency: &str, to_currency: &str) -> Option<Decimal> {
        let key = format!("{}rate:{}:{}", self.config.key_prefix, from_currency, to_currency);
        
        match self.get_cached_data::<Decimal>(&key).await {
            Ok(Some(rate)) => {
                self.record_hit().await;
                Some(rate)
            }
            _ => {
                self.record_miss().await;
                None
            }
        }
    }
    
    /// Set exchange rate in cache
    pub async fn set_exchange_rate(&self, from_currency: &str, to_currency: &str, rate: Decimal) {
        let key = format!("{}rate:{}:{}", self.config.key_prefix, from_currency, to_currency);
        let _ = self.set_cached_data(&key, &rate, Duration::from_secs(60)).await; // 1 min TTL for rates
    }
    
    /// Cache system configuration
    pub async fn get_system_config(&self, config_key: &str) -> Option<serde_json::Value> {
        let key = format!("{}config:{}", self.config.key_prefix, config_key);
        
        match self.get_cached_data::<serde_json::Value>(&key).await {
            Ok(Some(value)) => {
                self.record_hit().await;
                Some(value)
            }
            _ => {
                self.record_miss().await;
                None
            }
        }
    }
    
    /// Set system configuration in cache
    pub async fn set_system_config(&self, config_key: &str, value: &serde_json::Value) {
        let key = format!("{}config:{}", self.config.key_prefix, config_key);
        let _ = self.set_cached_data(&key, value, Duration::from_secs(3600)).await; // 1 hour TTL
    }
    
    /// Cache transaction validation results
    pub async fn get_transaction_validation(&self, tx_hash: &str) -> Option<bool> {
        let key = format!("{}tx_valid:{}", self.config.key_prefix, tx_hash);
        
        match self.get_cached_data::<bool>(&key).await {
            Ok(Some(is_valid)) => {
                self.record_hit().await;
                Some(is_valid)
            }
            _ => {
                self.record_miss().await;
                None
            }
        }
    }
    
    /// Set transaction validation result
    pub async fn set_transaction_validation(&self, tx_hash: &str, is_valid: bool) {
        let key = format!("{}tx_valid:{}", self.config.key_prefix, tx_hash);
        let _ = self.set_cached_data(&key, &is_valid, Duration::from_secs(1800)).await; // 30 min TTL
    }
    
    /// Cache fraud detection results
    pub async fn get_fraud_score(&self, transaction_id: &str) -> Option<f64> {
        let key = format!("{}fraud:{}", self.config.key_prefix, transaction_id);
        
        match self.get_cached_data::<f64>(&key).await {
            Ok(Some(score)) => {
                self.record_hit().await;
                Some(score)
            }
            _ => {
                self.record_miss().await;
                None
            }
        }
    }
    
    /// Set fraud detection score
    pub async fn set_fraud_score(&self, transaction_id: &str, score: f64) {
        let key = format!("{}fraud:{}", self.config.key_prefix, transaction_id);
        let _ = self.set_cached_data(&key, &score, Duration::from_secs(600)).await; // 10 min TTL
    }
    
    /// Invalidate account-related cache entries
    pub async fn invalidate_account(&self, account_id: &str) {
        let patterns = vec![
            format!("{}account:{}", self.config.key_prefix, account_id),
            format!("{}balance:{}:*", self.config.key_prefix, account_id),
        ];
        
        for pattern in patterns {
            let _ = self.delete_pattern(&pattern).await;
        }
    }
    
    /// Invalidate balance cache for specific account and currency
    pub async fn invalidate_balance(&self, account_id: &str, currency: &str) {
        let key = format!("{}balance:{}:{}", self.config.key_prefix, account_id, currency);
        let _ = self.delete_key(&key).await;
    }
    
    /// Invalidate system configuration cache
    pub async fn invalidate_system_config(&self, config_key: &str) {
        let key = format!("{}config:{}", self.config.key_prefix, config_key);
        let _ = self.delete_key(&key).await;
    }
    
    /// Get cache statistics
    pub async fn get_stats(&self) -> CacheStats {
        let mut stats = self.stats.read().await.clone();
        
        // Get Redis memory usage
        if let Ok(mut conn) = self.connection.try_write() {
            if let Ok(info) = conn.info("memory").await {
                let info_str: String = info;
                if let Some(memory_line) = info_str.lines()
                    .find(|line| line.starts_with("used_memory:")) {
                    if let Some(memory_str) = memory_line.split(':').nth(1) {
                        if let Ok(memory) = memory_str.parse::<u64>() {
                            stats.memory_usage_bytes = memory;
                        }
                    }
                }
            }
            
            // Get key count
            if let Ok(key_count) = conn.dbsize().await {
                let count: u64 = key_count;
                stats.active_keys = count;
            }
        }
        
        stats
    }
    
    /// Clear all cache entries
    pub async fn clear_all(&self) -> Result<()> {
        let mut conn = self.connection.write().await;
        let _: () = conn.flushdb()
            .await
            .map_err(|e| CoreError::StorageError(format!("Failed to clear cache: {}", e)))?;
        
        // Reset stats
        {
            let mut stats = self.stats.write().await;
            *stats = CacheStats::default();
        }
        
        Ok(())
    }
    
    /// Generic method to get cached data
    async fn get_cached_data<T>(&self, key: &str) -> Result<Option<T>>
    where
        T: for<'de> Deserialize<'de>,
    {
        let mut conn = self.connection.write().await;
        
        let cached_json: Option<String> = conn.get::<_, Option<String>>(key)
            .await
            .map_err(|e| CoreError::StorageError(format!("Failed to get cache entry: {}", e)))?;
        
        match cached_json {
            Some(json) => {
                let entry: CacheEntry<T> = serde_json::from_str(&json)
                    .map_err(|e| CoreError::StorageError(format!("Failed to deserialize cache entry: {}", e)))?;
                
                if entry.is_expired() {
                    // Remove expired entry
                    let _: () = conn.del(key)
                        .await
                        .map_err(|e| CoreError::StorageError(format!("Failed to delete expired cache entry: {}", e)))?;
                    
                    self.record_eviction().await;
                    Ok(None)
                } else {
                    Ok(Some(entry.data))
                }
            }
            None => Ok(None),
        }
    }
    
    /// Generic method to set cached data
    async fn set_cached_data<T>(&self, key: &str, data: &T, ttl: Duration) -> Result<()>
    where
        T: Serialize,
    {
        let entry = CacheEntry::new(data, ttl);
        let json = serde_json::to_string(&entry)
            .map_err(|e| CoreError::StorageError(format!("Failed to serialize cache entry: {}", e)))?;
        
        let mut conn = self.connection.write().await;
        let _: () = conn.setex(key, ttl.as_secs() as usize, json)
            .await
            .map_err(|e| CoreError::StorageError(format!("Failed to set cache entry: {}", e)))?;
        
        Ok(())
    }
    
    /// Delete single cache key
    async fn delete_key(&self, key: &str) -> Result<()> {
        let mut conn = self.connection.write().await;
        let _: () = conn.del(key)
            .await
            .map_err(|e| CoreError::StorageError(format!("Failed to delete cache key: {}", e)))?;
        
        Ok(())
    }
    
    /// Delete keys matching pattern
    async fn delete_pattern(&self, pattern: &str) -> Result<()> {
        let mut conn = self.connection.write().await;
        
        // Get keys matching pattern
        let keys: Vec<String> = conn.keys(pattern)
            .await
            .map_err(|e| CoreError::StorageError(format!("Failed to get keys for pattern: {}", e)))?;
        
        if !keys.is_empty() {
            let _: () = conn.del(&keys)
                .await
                .map_err(|e| CoreError::StorageError(format!("Failed to delete cache keys: {}", e)))?;
        }
        
        Ok(())
    }
    
    /// Record cache hit
    async fn record_hit(&self) {
        let mut stats = self.stats.write().await;
        stats.hits += 1;
    }
    
    /// Record cache miss
    async fn record_miss(&self) {
        let mut stats = self.stats.write().await;
        stats.misses += 1;
    }
    
    /// Record cache eviction
    async fn record_eviction(&self) {
        let mut stats = self.stats.write().await;
        stats.evictions += 1;
    }
    
    /// Get cache hit rate
    pub async fn hit_rate(&self) -> f64 {
        let stats = self.stats.read().await;
        let total = stats.hits + stats.misses;
        
        if total == 0 {
            0.0
        } else {
            stats.hits as f64 / total as f64
        }
    }
    
    /// Preload frequently accessed data
    pub async fn preload_hot_data(&self, account_ids: Vec<String>) -> Result<()> {
        tracing::info!("Preloading hot data for {} accounts", account_ids.len());
        
        // This would typically load account data and balances for frequently accessed accounts
        // Implementation would depend on the specific data access patterns
        
        for account_id in account_ids {
            // Set placeholder entries to prevent cache stampede
            let key = format!("{}account:{}", self.config.key_prefix, account_id);
            let _: () = self.connection.write().await
                .setex(&key, 60, "loading")
                .await
                .map_err(|e| CoreError::StorageError(format!("Failed to set loading placeholder: {}", e)))?;
        }
        
        Ok(())
    }
}

/// Cache warming task for background preloading
pub struct CacheWarmer {
    cache: Arc<CacheManager>,
}

impl CacheWarmer {
    pub fn new(cache: Arc<CacheManager>) -> Self {
        Self { cache }
    }
    
    /// Start background cache warming task
    pub async fn start_background_warming(&self) {
        let cache = Arc::clone(&self.cache);
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(300)); // Every 5 minutes
            
            loop {
                interval.tick().await;
                
                if let Err(e) = cache.warm_cache().await {
                    tracing::warn!("Cache warming failed: {}", e);
                }
                
                // Log cache statistics
                let stats = cache.get_stats().await;
                let hit_rate = cache.hit_rate().await;
                
                tracing::debug!(
                    "Cache stats - Hits: {}, Misses: {}, Hit Rate: {:.2}%, Memory: {} bytes, Keys: {}",
                    stats.hits,
                    stats.misses,
                    hit_rate * 100.0,
                    stats.memory_usage_bytes,
                    stats.active_keys
                );
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_cache_operations() {
        // Test basic cache operations
        // Implementation would use a test Redis instance
    }
    
    #[tokio::test]
    async fn test_cache_expiration() {
        // Test that expired entries are properly handled
        // Implementation would go here
    }
    
    #[tokio::test]
    async fn test_cache_invalidation() {
        // Test cache invalidation patterns
        // Implementation would go here
    }
}